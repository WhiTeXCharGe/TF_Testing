# Azure Functions Usage Idea for Timefold Project  
  
## Purpose  
  
The current Timefold project is built and executed locally using Java and Maven.  
  
As the next step, Azure Functions can be considered to run the Timefold scheduling process through an HTTP API.    
The main idea is to allow a web application to send input files, trigger the Timefold solver, and receive or access the result after execution.  
  
---  
  
## 1. Deployment Idea  
  
Azure Functions supports Java projects and can be deployed using Maven.    
Because the current Timefold project already uses Maven, it may be possible to convert or wrap the project as an Azure Functions project and deploy it with Maven.  
  
However, the Java runtime version must be checked carefully.    
Maven can build and deploy the Azure Functions project, but the target Java version must be supported by Azure Functions.  
  
For Azure Functions Runtime 4.x, the Java developer reference lists Java 8, 11, 17, and 21 as supported Java versions for Windows and Linux Function Apps. The Maven project can specify the Java version used by the compiler and the Java version hosted by the Function App. These values should match the supported Azure Functions Java runtime versions. 
  
This is important for the Timefold project because Timefold Solver 2.x requires Java 21 or later.    
Therefore, if the project upgrades to Timefold Solver 2.x, the safest target version for Azure Functions is Java 21.  
  

Local Java version: Java 24.0.2  
Recommended Azure Functions Java runtime: Java 21  
  
Reason:  
- Azure Functions supports Java 21  
- Timefold Solver 2.x requires Java 21 or later  
- Java 22, 23, and 24 should not be used as the normal Azure Functions Java runtime target

Maven deployment itself is possible, but the deployment target should be Java 21, not Java 24.

Example Maven compiler setting:

<properties>  
  <maven.compiler.source>21</maven.compiler.source>  
  <maven.compiler.target>21</maven.compiler.target>  
</properties>

If Azure Functions Maven Plugin is used, the Function App runtime should also be set to Java 21.

<javaVersion>21</javaVersion>

In short:

|Item|Direction|
|---|---|
|Maven deploy to Azure Functions|Possible|
|Java 21 deploy|Possible|
|Java 22+ / Java 24 deploy|Not recommended as normal Azure Functions Java runtime target|
|Current local Java|Java 24.0.2|
|Recommended Azure Functions Java|Java 21|
|Timefold Solver 1.x|Java 17 or Java 21 can be considered|
|Timefold Solver 2.x|Java 21 should be used|

---

## 2. HTTP Request and Response

Azure Functions can use an HTTP Trigger.  
This means the solver can be executed when a web application sends an HTTP request.

Basic flow:

Web App  
  ↓ HTTP POST  
Azure Functions  
  ↓  
Read EnvConfig.yaml and Schedule.yaml  
  ↓  
Run Timefold Solver  
  ↓  
Save output  
  ↓  
Return response

The HTTP response can return either the full result or only the output file path.

For this project, returning only the output path is probably better because the scheduling result may become large.

Example response:

{  
  "status": "success",  
  "outputPath": "output/result_schedule.yaml"  
}

---

## 3. Trigger and Input File Handling

The simplest trigger is an HTTP Trigger.

There are several possible ways to send `EnvConfig.yaml` and `Schedule.yaml`.

|Method|Description|Advantage|Concern|
|---|---|---|---|
|Send YAML directly in HTTP body|Send the content of both YAML files in the request|Simple for small tests|Not good for large files|
|Send Blob Storage paths|Upload YAML files to Blob Storage first, then send file paths to the Function|Better for large files|Needs Blob read/write logic|
|Convert YAML to JSON|Send input data as JSON instead of YAML|Easier for web app integration|Requires changing current YAML-based logic|

For this project, using Azure Blob Storage for input files is probably the most realistic option.

Example request:

{  
  "envConfigPath": "input/EnvConfig.yaml",  
  "schedulePath": "input/Schedule.yaml",  
  "outputPath": "output/result_schedule.yaml"  
}

This way, Azure Functions does not need to receive large YAML content directly through the HTTP request.  
It only receives the file paths, then reads the actual files from Blob Storage.

---

## 4. Output Handling

The Timefold solver output can be saved to Azure Blob Storage.

Possible output methods:

| Output Method | Description                                  |
| ------------- | -------------------------------------------- |
| HTTP Response | Return small result data directly (for test) |
| Blob Storage  | Save result files such as YAML,  or Excel    |
| Database      | Log                                          |

Since the scheduling result may become large, Blob Storage is suitable for storing output files.

Example Blob Storage structure:
```

Blob Storage  
├─ input/  
│  ├─ EnvConfig.yaml  
│  └─ Schedule.yaml  
├─ output/  
│  ├─ result_schedule.yaml  
│  └─ result_summary.json
```
The Azure Function can return the output path after saving the result.
```
{  
  "status": "success",  
  "outputPath": "output/result_schedule.yaml",  
  "summaryPath": "output/result_summary.json"  
}
```
If needed, each execution can have a unique folder name.
```
output/  
├─ run-20260428-001/  
│  ├─ result_schedule.yaml  
│  └─ result_summary.json  
├─ run-20260428-002/  
│  ├─ result_schedule.yaml  
│  └─ result_summary.json
```
This makes it easier to avoid overwriting files when multiple users run the solver.

---

## 5. Web App Calling Idea

A web application can call Azure Functions after the user uploads input files.

Possible flow:
```
User  
  ↓  
Web App  
  ↓ Upload YAML files  
Azure Blob Storage  
  ↓  
Web App calls Azure Functions  
  ↓  
Azure Functions runs Timefold Solver  
  ↓  
Output is saved to Blob Storage  
  ↓  
Web App displays or downloads the result
```
Simple JavaScript example:
```

async function runSolver() {  
  const response = await fetch("https://example-function.azurewebsites.net/api/runSolver", {  
    method: "POST",  
    headers: {  
      "Content-Type": "application/json"  
    },  
    body: JSON.stringify({  
      envConfigPath: "input/EnvConfig.yaml",  
      schedulePath: "input/Schedule.yaml",  
      outputPath: "output/result_schedule.yaml"  
    })  
  });  
  
  const result = await response.json();  
  console.log(result);  
}
```
The web application does not need to execute Timefold directly.  
It only uploads files, calls the Azure Function, and displays the result.

---

## 6. Azure Functions Side Image

The Azure Function receives the request, reads the YAML files, runs Timefold, and saves the result.

Simple Java image:
```
@FunctionName("runSolver")  
public HttpResponseMessage runSolver(  
    @HttpTrigger(  
        name = "req",  
        methods = {HttpMethod.POST},  
        authLevel = AuthorizationLevel.FUNCTION  
    )  
    HttpRequestMessage<Optional<String>> request,  
    final ExecutionContext context  
) {  
    // 1. Read envConfigPath, schedulePath, and outputPath from request body  
    // 2. Read EnvConfig.yaml and Schedule.yaml from Blob Storage  
    // 3. Run Timefold Solver  
    // 4. Save result to Blob Storage  
    // 5. Return output path as HTTP response  
  
    return request.createResponseBuilder(HttpStatus.OK)  
        .body("{\"status\":\"success\"}")  
        .build();  
}
```
This is only an image of the structure.  
Actual implementation still needs Blob Storage reading/writing logic and solver execution logic.
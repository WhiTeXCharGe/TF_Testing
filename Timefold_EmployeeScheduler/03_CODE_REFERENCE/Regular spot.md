# Draft ideas: Regular / Spot worker type  
  
Just gathering ideas for now, not final design.  
  
Based on current solver flow, this feature probably does **not** need a big redesign.    
It looks like the main change would be around:  
  
- worker data in `EnvConfig`  
- employee fact fields  
- candidate ordering for each seat  
- maybe soft score later if needed  
  
The main meaning is:  
  
- **regular** should be tried first  
- **spot** should be used after that  
- both are still valid workers unless business says otherwise  
  
So this feels more like **priority** than a hard rule.  
  
---  
  
## Common implementation direction  
  
For all cases, the most natural idea is:  
  
1. build feasible candidates as usual    
2. among those candidates, sort:  
   - regular first  
   - spot second  
   - then manager / skill / preference / balance as today  
  
So in Timefold terms, this feels closer to:  
  
- comparator  
- ordered candidate list  
- maybe soft penalty  
  
and **not** really a ValueRangeFactory problem.  
  
Because the legal range is still the same.    
The issue is not “who is allowed”, but “who should be picked first”.  
  
---  
  
# 1. Global regular / spot per employee  
  
## Image  
  
Each worker has only one type for everything.  
  
Example:  
  
- A = regular  
- B = spot  
  
No fab split, no phase split.  
  
## EnvConfig draft  
  
```yaml  
worker_list:  
  - id: w1  
    name: A  
    worker_type: regular  
  
  - id: w2  
    name: B  
    worker_type: spot
```
Maybe missing field = `spot` by default.

## Code draft

In `EmployeeFact`:

public String workerType;

## Solver idea

Very simple.

When candidates are prepared, sort them so:

- regular first
- then spot

Sample idea:

int roleRank(EmployeeFact e) {  
    return "regular".equalsIgnoreCase(e.workerType) ? 0 : 1;  
}

Then use that in comparator.

## Comment

This is the easiest version.  
Good for testing the concept first.  
But probably too rough for real use.

---

# 2. Regular / spot depends on phase or operation

## Image

Worker can be regular for some tasks and spot for others.

Example:

- A = regular for p1, p2
- A = spot for p3, p4

This feels much more realistic.

## EnvConfig draft

worker_list:  
  - id: w1  
    name: A  
    worker_type_by_operation:  
      p1: regular  
      p2: regular  
      p3: spot  
      p4: spot

Or phase style:

worker_type_by_phase:  
  tool_p1: regular  
  tool_p2: regular  
  tool_p3: spot  
  tool_p4: spot

I think operation-based is a bit easier because your current model already uses `p1/p2/p3/p4` a lot.

## Code draft

public Map<String, String> workerTypeByOperation = new HashMap<>();

Helper idea:

String getTypeForOp(String opId) {  
    return workerTypeByOperation.getOrDefault(opId, "spot");  
}

## Solver idea

For each seat/block, check worker type for that operation.

Then order candidates:

- regular for this op first
- spot for this op second

## Comment

This is probably the best balance.

Not too hard, not too weak.

---

# 3. Regular depends on fab + task

## Image

Worker is regular only for some fab/task combinations.

Example:

- regular for `f3 + p1`
- regular for `f3 + p2`
- spot for most others

This is the detailed version.

Because fab count is large, storing both regular and spot everywhere would be too heavy.  
So your idea makes sense:

- store **regular only**
- everything else = spot by default

## EnvConfig draft

### Style A

worker_list:  
  - id: w1  
    name: A  
    regular_assignment_map:  
      f3: [p1, p2]  
      f10: [p4]

### Style B

worker_list:  
  - id: w1  
    name: A  
    regular_assignment_keys:  
      - "f3|p1"  
      - "f3|p2"  
      - "f10|p4"

I think Style B is easier in code.

## Code draft

public Set<String> regularAssignmentKeys = new HashSet<>();

Helper:

boolean isRegularFor(String fabId, String opId) {  
    return regularAssignmentKeys.contains(fabId + "|" + opId);  
}

## Solver idea

When building seat candidates:

- if worker matches regular key for this fab/op → treat as regular
- else → treat as spot

Then sort regular first.

## Comment

This is probably the best long-term version if the business really works this way.  
But data prep is heavier.

---

# Technique choice draft

## Comparator / ordered candidates

This feels like the main tool.

Use when:

- all workers are still allowed
- but regular should be tried before spot

This sounds like your case.

## ValueRangeProvider / ValueRangeFactory

Probably not the main tool.

Reason:  
the value range is still basically the same employees.  
What changes is priority, not legality.

## Soft score

Can be added later if comparator is not enough.

Example idea:

- penalize assigning spot when some regular candidate was possible

Not needed for first version, but can help.

## Hard rule

Only if business really says:

- do not assign spot while regular exists

I would not start with this.

---

# Small code style idea

Maybe candidate handling can be like this:

List<EmployeeFact> regulars = new ArrayList<>();  
List<EmployeeFact> spots = new ArrayList<>();  
  
for (EmployeeFact e : feasibleEmployees) {  
    if (isRegularForThisSeat(e, seat, block)) regulars.add(e);  
    else spots.add(e);  
}  
  
regulars.sort(existingComparator);  
spots.sort(existingComparator);  
  
List<EmployeeFact> ordered = new ArrayList<>();  
ordered.addAll(regulars);  
ordered.addAll(spots);

This is simple and easy to read.

---

# Other possible case

## Region + task instead of fab + task

Maybe real business is not fab-based but region-based.

Then data becomes smaller.

Example:

worker_list:  
  - id: w1  
    name: A  
    regular_region_op_keys:  
      - "r3|p1"  
      - "r3|p2"

This could be a middle option between case 2 and case 3.

---

## Customer + task

If same customer has many fabs, maybe this is more useful than fab-specific.

regular_customer_op_keys:  
  - "c3|p1"  
  - "c3|p2"

---

## Mandatory regular seat

Another separate case:

- at least one regular worker must be inside the task
- or seat 0 must be regular

This is no longer just priority.  
This becomes a constraint case.

---

# Rough recommendation

## If want fastest test

Use case 1 first.

## If want the most practical version

Case 2 looks best.

## If real business is strongly fab-based

Use case 3, but only store regular mapping.

---

# Simple conclusion draft

This feature looks more like a **candidate priority** feature than a domain/range feature.

So first idea would be:

- add regular/spot data to worker
- detect regular/spot relative to seat
- sort feasible candidates with regular first
- keep spot as fallback
- maybe add soft penalty later if needed

So for Timefold, I would first think about:

- comparator
- candidate ordering
- optional soft score later

not ValueRangeFactory as the main solution.
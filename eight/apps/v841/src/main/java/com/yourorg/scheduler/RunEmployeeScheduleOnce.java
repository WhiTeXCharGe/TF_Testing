package com.yourorg.scheduler;

public class RunEmployeeScheduleOnce {

    public static void main(String[] args) {
        // Default paths (same as you pass from Maven / Python)
        String envPath = "src/main/resource/EnvConfig.yaml";
        String schedPath = "src/main/resource/Schedule.yaml";

        if (args.length >= 2) {
            envPath = args[0];
            schedPath = args[1];
        }

        System.out.println("[RUN] EmployeeSchedule with:");
        System.out.println("      EnvConfig = " + envPath);
        System.out.println("      Schedule  = " + schedPath);

        // Directly call your existing solver main
        EmployeeSchedule.main(new String[] { envPath, schedPath });

        System.out.println("[DONE] EmployeeSchedule finished.");
    }
}

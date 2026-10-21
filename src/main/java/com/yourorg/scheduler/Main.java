package com.yourorg.scheduler;

import java.io.IOException;

import com.yourorg.scheduler.build.PlanBuilder;
import com.yourorg.scheduler.export.ScheduleExporter;
import com.yourorg.scheduler.solver.RunResult;
import com.yourorg.scheduler.solver.TimerConfig;
import com.yourorg.scheduler.solver.TwoStageRunner;

/**
 * SINGLE PASS scheduler (Timefold 1.27-compatible): reads EnvConfig.yaml + Schedule.yaml, solves in two stages
 * and writes the assignments back into the Schedule file.
 *
 * <pre>
 * mvn -DskipTests clean package
 * mvn -q exec:java -D"exec.args=EnvConfig.yaml Schedule.yaml"
 * </pre>
 */
public class Main {

    public static RunResult solveFromYaml(String envPath, String schedPath) throws IOException {
        PlanBuilder.BuiltProblem problem = PlanBuilder.buildProblem(envPath, schedPath);
        return TwoStageRunner.run(problem, TimerConfig.defaults());
    }

    public static void main(String[] args) throws Exception {
        String envPath = args.length > 0 ? args[0] : "EnvConfig.yaml";
        String schedPath = args.length > 1 ? args[1] : "Schedule.yaml";

        RunResult rr = solveFromYaml(envPath, schedPath);

        ScheduleExporter.overwriteScheduleWithAssignments(
                rr.plan, rr.planStart, schedPath, envPath);

        System.out.println("Done.");
    }
}

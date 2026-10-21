package com.yourorg.scheduler.solver;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.format.DateTimeFormatter;
import java.util.Locale;

// ---------------- Live status reporting (STATUS_FILE) ----------------
//
// entrypoint.sh writes status=Running once, right before launching this JVM,
// then only writes again (Completed/Failed/Cancelled) after the JVM exits.
// On Azure Batch that meant the status blob showed "queued" for the whole
// solve — Batch only uploads outputFiles once the task command line exits
// (see api-controller's uploadCondition: 'taskCompletion'), so there was
// never an intermediate write for it to pick up. This ticker fills that gap
// by updating STATUS_FILE itself every few seconds with the real stage and
// an elapsed-time-based progress estimate, so a live reader (the API
// controller reading the task's file straight off the node) has something
// fresh to show while the task is still active/running.

/** Writes the live Running status file; the progress budget comes from the {@link TimerConfig}. */
public final class StatusReporter {

    private final long stage1BudgetMs;
    private final long stage2BudgetMs;
    private final long totalBudgetMs;

    public StatusReporter(TimerConfig t) {
        this.stage1BudgetMs = java.time.Duration.ofMinutes(t.stage1Minutes()).toMillis();
        this.stage2BudgetMs = t.stage2Enabled()
                ? java.time.Duration.ofMinutes(t.stage2Minutes()).toMillis() : 0L;
        this.totalBudgetMs = stage1BudgetMs + stage2BudgetMs;
    }

    private static final DateTimeFormatter STATUS_TS_FMT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'").withZone(java.time.ZoneOffset.UTC);
    private static String nowIsoUtc() { return STATUS_TS_FMT.format(java.time.Instant.now()); }

    private volatile String runStartedAtIso = null;

    public void writeRunningStatus(int stage, double overallProgress) {
        try {
            String statusDir = System.getenv().getOrDefault("STATUS_DIR", "/work/status");
            String runId = System.getenv().getOrDefault("RUN_ID", "local");
            if (runStartedAtIso == null) runStartedAtIso = nowIsoUtc();
            double clamped = Math.max(0.0, Math.min(0.99, overallProgress));

            String json = String.format(Locale.ROOT,
                "{%n  \"runId\": \"%s\",%n  \"status\": \"Running\",%n  \"stage\": %d,%n" +
                "  \"progress\": %.4f,%n  \"startedAt\": \"%s\",%n  \"updatedAt\": \"%s\",%n" +
                "  \"finishedAt\": null,%n  \"error\": null,%n  \"output\": null%n}%n",
                runId, stage, clamped, runStartedAtIso, nowIsoUtc());

            Path dir = Paths.get(statusDir);
            Files.createDirectories(dir);
            Path statusFile = dir.resolve(runId + ".json");
            Path tmp = dir.resolve(runId + ".json.tmp");
            Files.writeString(tmp, json, java.nio.charset.StandardCharsets.UTF_8);
            Files.move(tmp, statusFile,
                    java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                    java.nio.file.StandardCopyOption.ATOMIC_MOVE);
        } catch (Exception ignored) {
            // best-effort — never let status reporting break the solve
        }
    }

    /** overall progress = time spent in earlier stages' budgets + fraction of the current stage's budget. */
    public double overallProgressForStage(int stage, long msSpentInStage) {
        if (totalBudgetMs <= 0) return 1.0;
        long priorBudgetMs = (stage <= 1) ? 0 : stage1BudgetMs;
        long stageBudgetMs = (stage <= 1) ? stage1BudgetMs : stage2BudgetMs;
        double stageFrac = stageBudgetMs <= 0 ? 1.0 : Math.min(1.0, msSpentInStage / (double) stageBudgetMs);
        return (priorBudgetMs + stageFrac * stageBudgetMs) / (double) totalBudgetMs;
    }
}

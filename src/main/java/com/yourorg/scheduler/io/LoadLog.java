package com.yourorg.scheduler.io;

import java.io.*;
import java.nio.file.*;
import java.util.*;

import com.yourorg.scheduler.model.*;

/** load_log.txt diagnostics written while parsing and building the problem. */
public class LoadLog {

    static final Path LOAD_LOG_FILE = Paths.get("load_log.txt");

    public static void appendLoadLog(String line) {
        try (BufferedWriter bw = Files.newBufferedWriter(
                LOAD_LOG_FILE,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND)) {
            bw.write(line);
            bw.newLine();
        } catch (IOException e) {
            // last resort: do nothing (or keep a minimal stderr)
            // System.err.println("Failed to write load_log.txt: " + e.getMessage());
        }
    }

    public static void logAllWindowsDetail(ParsedSchedule sch) {
        if (sch == null || sch.windows == null || sch.windows.isEmpty()) return;

        int daySlotsCount = (sch.daySlots == null) ? 0 : sch.daySlots.size();
        int minDay = (daySlotsCount == 0) ? Integer.MIN_VALUE : sch.daySlots.get(0).id;
        int maxDay = (daySlotsCount == 0) ? Integer.MAX_VALUE : sch.daySlots.get(daySlotsCount - 1).id;

        appendLoadLog("------------------------------------------------------------");
        appendLoadLog("TaskWindow details (after push+trim, before block build clamp)");
        appendLoadLog("solverDaySlots=" + daySlotsCount + " [" + minDay + ".." + maxDay + "]");

        // stable order
        List<TaskWindow> sorted = new ArrayList<>(sch.windows);
        sorted.sort(Comparator
            .comparing((TaskWindow w) -> YamlFiles.safeStr(w.module))
            .thenComparingInt(w -> w.phaseNum)
            .thenComparing(w -> YamlFiles.safeStr(w.opId)));

        int idx = 0;
        for (TaskWindow w : sorted) {
            if (w == null) continue;

            boolean exceeds = (w.startDayId < minDay) || (w.endDayId > maxDay);
            int clampStart = Math.max(w.startDayId, minDay);
            int clampEnd   = Math.min(w.endDayId,   maxDay);
            boolean outside = clampStart > clampEnd;

            appendLoadLog(String.format(
                Locale.ROOT,
                "[W%04d] module=%s wf=%s fab=%s phase=%s(#%d) op=%s " +
                "win=%d..%d clamp=%d..%d exceeds=%s outside=%s " +
                "workloadDays=%d min=%d max=%d allowed=%s recommendRange=%d..%d",
                (++idx),
                YamlFiles.safeStr(w.module), YamlFiles.safeStr(w.workflowId), YamlFiles.safeStr(w.factory),
                YamlFiles.safeStr(w.phaseId), w.phaseNum, YamlFiles.safeStr(w.opId),
                w.startDayId, w.endDayId,
                clampStart, clampEnd,
                String.valueOf(exceeds),
                String.valueOf(outside),
                w.workloadDays,
                w.minHeads, w.maxHeads,
                String.valueOf(w.allowed),
                w.recommendMinHeads, w.recommendMaxHeads
            ));
        }
    }

    public static void logModulePhaseCounts(ParsedSchedule sch) {
        try {
            if (sch == null) return;

            int daySlotsCount = (sch.daySlots == null) ? 0 : sch.daySlots.size();
            int minDay = (daySlotsCount == 0) ? Integer.MIN_VALUE : sch.daySlots.get(0).id;
            int maxDay = (daySlotsCount == 0) ? Integer.MAX_VALUE : sch.daySlots.get(daySlotsCount - 1).id;

            // ---------------- PLAN side (from windows) ----------------
            Set<String> planModules = new HashSet<>();
            Set<String> planPhases  = new HashSet<>(); // e.g., "e1_p3"

            if (sch.windows != null) {
                for (TaskWindow w : sch.windows) {
                    if (w == null) continue;
                    if (w.module == null || w.module.isBlank()) continue;

                    planModules.add(w.module);

                    int pn = w.phaseNum;
                    if (pn > 0) {
                        planPhases.add(w.module + "_p" + pn);
                    }
                }
            }

            // Helper: any positive hours inside solver range?
            java.util.function.Predicate<Map<Integer,Integer>> anyPositiveInRange = (byDay) -> {
                if (byDay == null || byDay.isEmpty()) return false;
                for (var e : byDay.entrySet()) {
                    int did = e.getKey();
                    int h   = (e.getValue() == null) ? 0 : e.getValue();
                    if (h <= 0) continue;
                    if (did >= minDay && did <= maxDay) return true;
                }
                return false;
            };

            // ---------------- FIXED side ----------------
            Set<String> fixedModules = new HashSet<>();
            Set<String> fixedPhases  = new HashSet<>();
            int fixedRowsInRange = 0;

            if (sch.fixedRows != null) {
                for (FixedAssign fa : sch.fixedRows) {
                    if (fa == null) continue;
                    if (fa.module == null || fa.module.isBlank()) continue;

                    if (!anyPositiveInRange.test(fa.hoursByDay)) continue;

                    fixedRowsInRange++;
                    fixedModules.add(fa.module);

                    int pn = fa.phaseNum;
                    if (pn > 0) fixedPhases.add(fa.module + "_p" + pn);
                }
            }

            // ---------------- FLEXIBLE side ----------------
            Set<String> flexModules = new HashSet<>();
            Set<String> flexPhases  = new HashSet<>();
            int flexRowsInRange = 0;

            if (sch.flexibleRows != null) {
                for (FlexibleAssign fa : sch.flexibleRows) {
                    if (fa == null) continue;
                    if (fa.module == null || fa.module.isBlank()) continue;

                    if (!anyPositiveInRange.test(fa.hoursByDay)) continue;

                    flexRowsInRange++;
                    flexModules.add(fa.module);

                    int pn = fa.phaseNum;
                    if (pn > 0) flexPhases.add(fa.module + "_p" + pn);
                }
            }

            // ---------------- TOTAL (fixed ∪ flexible) ----------------
            Set<String> totalAssignModules = new HashSet<>(fixedModules);
            totalAssignModules.addAll(flexModules);

            Set<String> totalAssignPhases = new HashSet<>(fixedPhases);
            totalAssignPhases.addAll(flexPhases);

            // overlap debug (optional)
            Set<String> bothModules = new HashSet<>(fixedModules);
            bothModules.retainAll(flexModules);

            // output
            appendLoadLog("------------------------------------------------------------");
            appendLoadLog("Module/Phase counts (modules=e1..., phases=e1_p1..)");
            appendLoadLog("solverDaySlots=" + daySlotsCount + " [" + minDay + ".." + maxDay + "]");

            appendLoadLog("[PLAN windows] modules=" + planModules.size()
                    + ", phases=" + planPhases.size());

            appendLoadLog("[ASSIGN fixed] modules=" + fixedModules.size()
                    + ", phases=" + fixedPhases.size()
                    + ", rowsInRange=" + fixedRowsInRange);

            appendLoadLog("[ASSIGN flexible] modules=" + flexModules.size()
                    + ", phases=" + flexPhases.size()
                    + ", rowsInRange=" + flexRowsInRange);

            appendLoadLog("[ASSIGN total(fixed+flex)] modules=" + totalAssignModules.size()
                    + ", phases=" + totalAssignPhases.size());

            appendLoadLog("[ASSIGN overlap] modulesInBothFixedAndFlexible=" + bothModules.size());

        } catch (Exception ex) {
            appendLoadLog("[WARN] Failed to log module/phase counts: " + ex.getMessage());
        }
    }

    public static void logParsedEnv(ParsedEnv env) {
        try {
            int workers = (env == null || env.employees == null) ? 0 : Math.max(0, env.employees.size() - 1); // exclude UNASSIGNED
            int opDefs  = (env == null || env.opdef == null) ? 0 : env.opdef.size();
            int byWid   = (env == null || env.byWid == null) ? 0 : env.byWid.size();

            appendLoadLog("============================================================");
            appendLoadLog("EnvConfig parse summary");
            appendLoadLog("workers=" + workers + ", opDefs=" + opDefs + ", byWid=" + byWid);

            if (env != null && env.byWid != null) {
                long blank = env.byWid.keySet().stream().filter(k -> k == null || k.isBlank()).count();
                if (blank > 0) appendLoadLog("[WARN] EnvConfig: blank worker id entries=" + blank);
            }
        } catch (Exception ex) {
            appendLoadLog("[WARN] Failed to log EnvConfig summary: " + ex.getMessage());
        }
    }

    public static void logParsedSchedule(ParsedSchedule sch) {
        try {
            if (sch == null) return;

            int daySlotsCount = (sch.daySlots == null) ? 0 : sch.daySlots.size();
            int minDay = (daySlotsCount == 0) ? -1 : sch.daySlots.get(0).id;
            int maxDay = (daySlotsCount == 0) ? -1 : sch.daySlots.get(daySlotsCount - 1).id;

            int horizonDays = (int) (sch.planEnd.toEpochDay() - sch.planStart.toEpochDay()) + 1;

            appendLoadLog("------------------------------------------------------------");
            appendLoadLog("Schedule parse summary");
            appendLoadLog("plan_range=" + sch.planStart + ".." + sch.planEnd
                    + ", cut_off=" + sch.cutOffDate + " (dayId=" + sch.cutOffDayId + ")"
                    + ", horizonDays=" + horizonDays);
            appendLoadLog("daySlots=" + daySlotsCount + " [" + minDay + ".." + maxDay + "]");
            appendLoadLog("windows=" + (sch.windows == null ? 0 : sch.windows.size())
                    + ", fixedRows=" + (sch.fixedRows == null ? 0 : sch.fixedRows.size())
                    + ", flexibleRows=" + (sch.flexibleRows == null ? 0 : sch.flexibleRows.size())
                    + ", requiredKeys=" + (sch.requiredByKey == null ? 0 : sch.requiredByKey.size())
                    + ", activeModules=" + (sch.activeModules == null ? 0 : sch.activeModules.size()));

            // warn if windows exceed solver daySlots
            if (sch.daySlots != null && !sch.daySlots.isEmpty() && sch.windows != null) {
                long outCount = sch.windows.stream()
                        .filter(w -> w != null && (w.startDayId < minDay || w.endDayId > maxDay))
                        .count();
                if (outCount > 0) {
                    appendLoadLog("[WARN] Schedule: " + outCount
                            + " TaskWindow(s) exceed solver daySlots; they will be clamped when building blocks.");
                }
            }
        } catch (Exception ex) {
            appendLoadLog("[WARN] Failed to log Schedule summary: " + ex.getMessage());
        }
    }
}

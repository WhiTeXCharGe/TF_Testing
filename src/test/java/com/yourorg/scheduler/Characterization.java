package com.yourorg.scheduler;

import com.yourorg.scheduler.build.PlanBuilder;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import java.util.TreeSet;

import ai.timefold.solver.core.api.solver.Solver;

import org.yaml.snakeyaml.LoaderOptions;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.SafeConstructor;

import com.yourorg.scheduler.model.SinglePassPlan;
import com.yourorg.scheduler.solver.ConstraintBreakdown;
import com.yourorg.scheduler.solver.SolverFactories;

/**
 * Characterisation harness: records what the CURRENT code computes for a fixed input, so that later
 * refactoring / upgrade steps can prove they change no behaviour.
 *
 * <p>Golden files are JSON under {@code src/test/resources/golden/}. They are (re)written only when the
 * system property {@code golden.write=true} is set; otherwise they are only compared.
 */
final class Characterization {

    private Characterization() {}

    /** Score and match count of one constraint. */
    record ConstraintTotal(String score, int matches) {}

    /** Total score plus the per-constraint totals of one solution. */
    record Snapshot(String score, Map<String, ConstraintTotal> constraints) {}

    static final Path GOLDEN_DIR = Path.of("src", "test", "resources", "golden");

    static boolean goldenWrite() {
        return Boolean.parseBoolean(System.getProperty("golden.write", "false"));
    }

    // ------------------------------------------------------------------ building

    /**
     * Copies EnvConfig.yaml and Schedule.yaml from {@code inputDir} into {@code workDir} and builds the plan from the
     * copies. The copy is required: preprocessScheduleYaml rewrites the schedule file in place.
     */
    static SinglePassPlan buildFrom(Path inputDir, Path workDir) throws IOException {
        Files.createDirectories(workDir);
        Path env = workDir.resolve("EnvConfig.yaml");
        Path sched = workDir.resolve("Schedule.yaml");
        Files.copy(inputDir.resolve("EnvConfig.yaml"), env, StandardCopyOption.REPLACE_EXISTING);
        Files.copy(inputDir.resolve("Schedule.yaml"), sched, StandardCopyOption.REPLACE_EXISTING);
        return PlanBuilder.buildPlanForTests(env.toString(), sched.toString());
    }

    // ------------------------------------------------------------------ measuring

    /**
     * Score analysis of {@code p} as it stands (no solving). Every registered constraint is listed. On Timefold 2.x
     * Community {@code SolutionManager.analyze} is Enterprise-only; {@link ConstraintBreakdown} gives the same
     * per-constraint scores and match counts (pinned by the goldens).
     */
    static Snapshot analyze(SinglePassPlan p) {
        ConstraintBreakdown.Result r = ConstraintBreakdown.of(SolverFactories.forTests(null, null)).analyze(p);
        Map<String, ConstraintTotal> totals = new TreeMap<>();
        r.totals().forEach((id, t) -> totals.put(id, new ConstraintTotal(String.valueOf(t.score()), t.matches())));
        return new Snapshot(String.valueOf(r.score()), totals);
    }

    /** Per-constraint score and match count of the initial (warm-start) solution. */
    static Map<String, ConstraintTotal> initialTotals(SinglePassPlan p) {
        return analyze(p).constraints();
    }

    /** Solves {@code p} with a step-count limit and a fixed random seed; returns the best solution. */
    static SinglePassPlan solveSeeded(SinglePassPlan p, int stepLimit, long seed) {
        Solver<SinglePassPlan> solver = SolverFactories.forTests(stepLimit, seed).buildSolver();
        return solver.solve(p);
    }

    /** Final score string of a seeded, step-limited run. */
    static String seededRun(SinglePassPlan p, int stepLimit, long seed) {
        return String.valueOf(solveSeeded(p, stepLimit, seed).getScore());
    }

    // ------------------------------------------------------------------ golden I/O

    static Map<String, Object> snapshotToJson(Snapshot s) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("score", s.score());
        Map<String, Object> cs = new LinkedHashMap<>();
        for (Map.Entry<String, ConstraintTotal> e : new TreeMap<>(s.constraints()).entrySet()) {
            Map<String, Object> one = new LinkedHashMap<>();
            one.put("score", e.getValue().score());
            one.put("matches", e.getValue().matches());
            cs.put(e.getKey(), one);
        }
        out.put("constraints", cs);
        return out;
    }

    @SuppressWarnings("unchecked")
    static Snapshot snapshotFromJson(Map<String, Object> json) {
        Map<String, ConstraintTotal> totals = new TreeMap<>();
        Map<String, Object> cs = (Map<String, Object>) json.getOrDefault("constraints", Map.of());
        for (Map.Entry<String, Object> e : cs.entrySet()) {
            Map<String, Object> one = (Map<String, Object>) e.getValue();
            totals.put(e.getKey(), new ConstraintTotal(String.valueOf(one.get("score")),
                    ((Number) one.get("matches")).intValue()));
        }
        return new Snapshot(String.valueOf(json.get("score")), totals);
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> readGolden(Path file) throws IOException {
        String text = Files.readString(file, StandardCharsets.UTF_8);
        // JSON is a subset of YAML flow style; SnakeYAML is already a dependency.
        Object root = new Yaml(new SafeConstructor(new LoaderOptions())).load(text);
        return (root instanceof Map) ? new LinkedHashMap<>((Map<String, Object>) root) : new LinkedHashMap<>();
    }

    static void writeGolden(Path file, Map<String, Object> root) throws IOException {
        Files.createDirectories(file.getParent());
        StringBuilder sb = new StringBuilder();
        writeJson(sb, root, 0);
        sb.append('\n');
        Files.writeString(file, sb.toString(), StandardCharsets.UTF_8);
    }

    @SuppressWarnings("unchecked")
    private static void writeJson(StringBuilder sb, Object v, int indent) {
        if (v instanceof Map<?, ?> m) {
            if (m.isEmpty()) { sb.append("{}"); return; }
            sb.append("{\n");
            int i = 0;
            for (Map.Entry<String, Object> e : ((Map<String, Object>) m).entrySet()) {
                sb.append("  ".repeat(indent + 1));
                writeString(sb, e.getKey());
                sb.append(": ");
                writeJson(sb, e.getValue(), indent + 1);
                if (++i < m.size()) sb.append(',');
                sb.append('\n');
            }
            sb.append("  ".repeat(indent)).append('}');
        } else if (v instanceof Number || v instanceof Boolean) {
            sb.append(v);
        } else if (v == null) {
            sb.append("null");
        } else {
            writeString(sb, String.valueOf(v));
        }
    }

    private static void writeString(StringBuilder sb, String s) {
        sb.append('"');
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> {
                    if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
                    else sb.append(c);
                }
            }
        }
        sb.append('"');
    }

    // ------------------------------------------------------------------ compare

    /**
     * Compares {@code actual} with the named section of the golden file, or (with -Dgolden.write=true) stores it there.
     * Fails with a readable, per-constraint diff when they differ.
     *
     * @param params extra values that identify how the section was produced (e.g. step_limit, seed); stored and
     *               compared as well
     */
    @SuppressWarnings("unchecked")
    static void checkSection(Path golden, String section, Map<String, Object> params, Snapshot actual)
            throws IOException {
        Map<String, Object> actualJson = new LinkedHashMap<>(params);
        actualJson.putAll(snapshotToJson(actual));

        if (goldenWrite()) {
            Map<String, Object> root = Files.exists(golden) ? readGolden(golden) : new LinkedHashMap<>();
            Map<String, Object> ordered = new LinkedHashMap<>();
            ordered.put("description", "Characterisation golden written by CharacterizationTest with "
                    + "-Dgolden.write=true. Do not edit by hand; regenerate only for an intended behaviour change.");
            root.remove("description");
            ordered.putAll(root);
            ordered.put(section, actualJson);
            writeGolden(golden, ordered);
            System.out.println("[golden] wrote section '" + section + "' of " + golden.toAbsolutePath());
            return;
        }

        if (!Files.exists(golden)) {
            throw new AssertionError("golden missing: " + golden.toAbsolutePath()
                    + " (generate it with: mvn test -Dgolden.write=true)");
        }
        Map<String, Object> root = readGolden(golden);
        Object sec = root.get(section);
        if (!(sec instanceof Map)) {
            throw new AssertionError("golden missing: section '" + section + "' not in " + golden.toAbsolutePath()
                    + " (generate it with: mvn test -Dgolden.write=true)");
        }
        Map<String, Object> expectedJson = (Map<String, Object>) sec;

        List<String> problems = new ArrayList<>();
        for (Map.Entry<String, Object> p : params.entrySet()) {
            Object exp = expectedJson.get(p.getKey());
            if (!Objects.equals(String.valueOf(exp), String.valueOf(p.getValue()))) {
                problems.add(String.format("  PARAM   %s: golden=%s actual=%s", p.getKey(), exp, p.getValue()));
            }
        }
        String diff = diff(snapshotFromJson(expectedJson), actual);
        if (!problems.isEmpty() || !diff.isEmpty()) {
            StringBuilder msg = new StringBuilder();
            msg.append("Characterisation mismatch in ").append(golden.getFileName())
               .append(" section '").append(section).append("':\n");
            problems.forEach(l -> msg.append(l).append('\n'));
            msg.append(diff);
            throw new AssertionError(msg.toString());
        }
    }

    /** Human-readable difference between two snapshots; empty when equal. */
    static String diff(Snapshot expected, Snapshot actual) {
        StringBuilder sb = new StringBuilder();
        if (!Objects.equals(expected.score(), actual.score())) {
            sb.append(String.format("  TOTAL   golden=%s actual=%s%n", expected.score(), actual.score()));
        }
        TreeSet<String> names = new TreeSet<>(expected.constraints().keySet());
        names.addAll(actual.constraints().keySet());
        for (String n : names) {
            ConstraintTotal e = expected.constraints().get(n);
            ConstraintTotal a = actual.constraints().get(n);
            if (e == null) {
                sb.append(String.format("  NEW     %-40s actual=%s (%d matches), not in golden%n",
                        n, a.score(), a.matches()));
            } else if (a == null) {
                sb.append(String.format("  MISSING %-40s golden=%s (%d matches), not computed any more%n",
                        n, e.score(), e.matches()));
            } else if (!e.equals(a)) {
                sb.append(String.format("  CHANGED %-40s golden=%s (%d matches) actual=%s (%d matches)%n",
                        n, e.score(), e.matches(), a.score(), a.matches()));
            }
        }
        return sb.toString();
    }
}

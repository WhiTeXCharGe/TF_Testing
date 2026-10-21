package com.yourorg.scheduler.config;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

import org.yaml.snakeyaml.LoaderOptions;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.SafeConstructor;

import com.yourorg.scheduler.solver.TimerConfig;

/**
 * Run parameters. Every parameter is resolved independently with this precedence (first source that mentions the key
 * wins): CLI {@code --key=value} (or {@code --key value}) &gt; environment variable {@code TF_<KEY>} &gt; the params
 * YAML file (top-level map) &gt; default.
 *
 * <p>A value that cannot be parsed falls back to the DEFAULT (not to a lower-precedence source) and adds a warning.
 * Unknown CLI / file keys are ignored with a warning. {@code move_thread_count} is recognised but ignored (Timefold
 * Enterprise Edition only).
 *
 * @param stage1Minutes           stage 1 time limit in minutes (&gt;= 1)
 * @param stage1UnimprovedSeconds stage 1 stops after this many seconds without a new best score (&gt;= 1)
 * @param stage1StopWhenFeasible  stage 1 also stops as soon as the hard score reaches 0
 * @param stage2Enabled           false skips the polish stage
 * @param stage2Minutes           stage 2 time limit in minutes (&gt;= 1)
 * @param stage2UnimprovedSeconds stage 2 stops after this many seconds without a new best score (&gt;= 1)
 * @param randomSeed              random seed for both stages, or null for none
 * @param tripValidation          "lenient" or "strict"
 * @param miscNeutralRegions      region ids treated as neutral for the misc rule
 * @param tripHistoryDays         trip history window in days (-1 = unlimited)
 * @param writeScoreFiles         false skips initial/stage1/final_score.txt and score_progress.txt
 * @param allowCutFixedTrip       whether a fixed trip may be cut
 * @param warnings                everything that was ignored or replaced by a default while loading
 */
public record SolverParams(int stage1Minutes,
                           int stage1UnimprovedSeconds,
                           boolean stage1StopWhenFeasible,
                           boolean stage2Enabled,
                           int stage2Minutes,
                           int stage2UnimprovedSeconds,
                           Long randomSeed,
                           String tripValidation,
                           Set<String> miscNeutralRegions,
                           int tripHistoryDays,
                           boolean writeScoreFiles,
                           boolean allowCutFixedTrip,
                           List<String> warnings) {

    public SolverParams {
        miscNeutralRegions = Collections.unmodifiableSet(new LinkedHashSet<>(miscNeutralRegions));
        warnings = List.copyOf(warnings);
    }

    // ------------------------------------------------------------------ keys

    static final String STAGE1_MINUTES = "stage1_minutes";
    static final String STAGE1_UNIMPROVED_SECONDS = "stage1_unimproved_seconds";
    static final String STAGE1_STOP_WHEN_FEASIBLE = "stage1_stop_when_feasible";
    static final String STAGE2_ENABLED = "stage2_enabled";
    static final String STAGE2_MINUTES = "stage2_minutes";
    static final String STAGE2_UNIMPROVED_SECONDS = "stage2_unimproved_seconds";
    static final String RANDOM_SEED = "random_seed";
    static final String TRIP_VALIDATION = "trip_validation";
    static final String MISC_NEUTRAL_REGIONS = "misc_neutral_regions";
    static final String TRIP_HISTORY_DAYS = "trip_history_days";
    static final String WRITE_SCORE_FILES = "write_score_files";
    static final String ALLOW_CUT_FIXED_TRIP = "allow_cut_fixed_trip";
    static final String MOVE_THREAD_COUNT = "move_thread_count";

    /** Canonical key by its normalised spelling (lower case, no '_' / '-'). */
    private static final Map<String, String> KEYS_BY_NORMALISED = new LinkedHashMap<>();
    /** Extra environment variable names per key (after {@code TF_<KEY>}), in order. */
    private static final Map<String, List<String>> EXTRA_ENV_NAMES = Map.of(
            ALLOW_CUT_FIXED_TRIP, List.of("TF_ALLOWCUTFIXEDFIFO", "ALLOW_CUT_FIXED_TRIP", "ALLOWCUTFIXEDFIFO"));

    static {
        for (String k : List.of(STAGE1_MINUTES, STAGE1_UNIMPROVED_SECONDS, STAGE1_STOP_WHEN_FEASIBLE, STAGE2_ENABLED,
                STAGE2_MINUTES, STAGE2_UNIMPROVED_SECONDS, RANDOM_SEED, TRIP_VALIDATION, MISC_NEUTRAL_REGIONS,
                TRIP_HISTORY_DAYS, WRITE_SCORE_FILES, ALLOW_CUT_FIXED_TRIP, MOVE_THREAD_COUNT)) {
            KEYS_BY_NORMALISED.put(normalise(k), k);
        }
        KEYS_BY_NORMALISED.put("allowcutfixedfifo", ALLOW_CUT_FIXED_TRIP);
    }

    static String normalise(String key) {
        return key.toLowerCase(Locale.ROOT).replace("_", "").replace("-", "");
    }

    /** Canonical key for any accepted spelling ({@code --Stage1-Minutes}, {@code stage1minutes}, ...), else null. */
    public static String canonicalKey(String spelling) {
        return spelling == null ? null : KEYS_BY_NORMALISED.get(normalise(spelling));
    }

    // ------------------------------------------------------------------ the timers

    /** The time budgets and stop rules of the two-stage solve. */
    public TimerConfig timers() {
        return new TimerConfig(stage1Minutes, stage1UnimprovedSeconds, stage1StopWhenFeasible,
                stage2Enabled, stage2Minutes, stage2UnimprovedSeconds, randomSeed);
    }

    /** One-line summary of the effective values, for the run log. */
    public String describe() {
        return "stage1=" + stage1Minutes + "min/" + stage1UnimprovedSeconds + "s"
                + " stopWhenFeasible=" + stage1StopWhenFeasible
                + " stage2=" + stage2Minutes + "min/" + stage2UnimprovedSeconds + "s"
                + " stage2Enabled=" + stage2Enabled
                + " seed=" + (randomSeed == null ? "none" : randomSeed.toString())
                + " tripValidation=" + tripValidation
                + " miscNeutralRegions=" + miscNeutralRegions
                + " tripHistoryDays=" + tripHistoryDays
                + " writeScoreFiles=" + writeScoreFiles
                + " allowCutFixedTrip=" + allowCutFixedTrip;
    }

    // ------------------------------------------------------------------ loading

    /**
     * Resolves all parameters.
     *
     * @param cliArgs    command line tokens; only {@code --key=value} / {@code --key value} flags are looked at,
     *                   anything else (positional arguments) is skipped
     * @param env        environment variables (normally {@code System.getenv()})
     * @param paramsFile the params YAML file; null or missing is not an error
     */
    public static SolverParams load(String[] cliArgs, Map<String, String> env, Path paramsFile) {
        return new Loader(cliArgs == null ? new String[0] : cliArgs, env == null ? Map.of() : env, paramsFile).run();
    }

    /** A value found in one of the sources (the value itself may be null, e.g. a YAML key without a value). */
    private record Found(Object value) {}

    private static final class Loader {
        private final List<String> warnings = new ArrayList<>();
        private final Map<String, Object> cli = new LinkedHashMap<>();
        private final Map<String, Object> envValues = new LinkedHashMap<>();
        private final Map<String, Object> file = new LinkedHashMap<>();

        Loader(String[] cliArgs, Map<String, String> env, Path paramsFile) {
            readCli(cliArgs);
            readEnv(env);
            readFile(paramsFile);
        }

        // ---- sources

        private void readCli(String[] args) {
            for (int i = 0; i < args.length; i++) {
                String token = args[i];
                if (token == null || !token.startsWith("--")) continue; // positional argument
                String body = token.substring(2);
                int eq = body.indexOf('=');
                String name = eq >= 0 ? body.substring(0, eq) : body;
                String canonical = canonicalKey(name);
                String value;
                if (eq >= 0) {
                    value = body.substring(eq + 1);
                } else if (canonical != null && i + 1 < args.length && args[i + 1] != null
                        && !args[i + 1].startsWith("--")) {
                    value = args[++i]; // "--key value"
                } else {
                    value = "";
                }
                if (canonical == null) {
                    warnings.add("unknown parameter: " + name);
                } else {
                    cli.put(canonical, value);
                }
            }
        }

        private void readEnv(Map<String, String> env) {
            for (String key : KEYS_BY_NORMALISED.values()) {
                List<String> names = new ArrayList<>();
                names.add("TF_" + key.toUpperCase(Locale.ROOT));
                names.addAll(EXTRA_ENV_NAMES.getOrDefault(key, List.of()));
                for (String n : names) {
                    String v = env.get(n);
                    if (v == null) continue;
                    // a blank variable (docker -e X=) counts as "not set", except for the list where blank = empty set
                    if (v.isBlank() && !MISC_NEUTRAL_REGIONS.equals(key)) continue;
                    envValues.put(key, v);
                    break;
                }
            }
        }

        private void readFile(Path paramsFile) {
            if (paramsFile == null || !Files.exists(paramsFile)) return;
            Object root;
            try (InputStream in = Files.newInputStream(paramsFile)) {
                LoaderOptions opts = new LoaderOptions();
                opts.setCodePointLimit(1024 * 1024);
                root = new Yaml(new SafeConstructor(opts)).load(in);
            } catch (IOException | RuntimeException e) {
                warnings.add("could not read params file " + paramsFile + " (using defaults): "
                        + firstLine(e.getMessage() == null ? e.toString() : e.getMessage()));
                return;
            }
            if (root == null) return; // empty file
            if (!(root instanceof Map<?, ?> map)) {
                warnings.add("params file " + paramsFile + " is not a YAML map (using defaults)");
                return;
            }
            for (Map.Entry<?, ?> e : map.entrySet()) {
                String name = String.valueOf(e.getKey());
                String canonical = canonicalKey(name);
                if (canonical == null) {
                    warnings.add("unknown parameter: " + name);
                } else {
                    file.put(canonical, e.getValue());
                }
            }
        }

        private static String firstLine(String s) {
            int nl = s.indexOf('\n');
            return (nl < 0 ? s : s.substring(0, nl)).trim();
        }

        // ---- lookup and parsing

        private Found find(String key) {
            if (cli.containsKey(key)) return new Found(cli.get(key));
            if (envValues.containsKey(key)) return new Found(envValues.get(key));
            if (file.containsKey(key)) return new Found(file.get(key));
            return null;
        }

        private <T> T get(String key, Function<Object, T> parser, T def) {
            Found f = find(key);
            if (f == null) return def;
            T parsed = parser.apply(f.value());
            if (parsed != null) return parsed;
            warnings.add("bad value for " + key + ": \"" + f.value() + "\" (using default " + showDefault(def) + ")");
            return def;
        }

        private static String showDefault(Object def) {
            return def == null ? "none" : String.valueOf(def);
        }

        SolverParams run() {
            int stage1Minutes = get(STAGE1_MINUTES, v -> parseInt(v, 1), 300);
            int stage1Unimproved = get(STAGE1_UNIMPROVED_SECONDS, v -> parseInt(v, 1), 600);
            boolean stopWhenFeasible = get(STAGE1_STOP_WHEN_FEASIBLE, Loader::parseBoolean, true);
            boolean stage2Enabled = get(STAGE2_ENABLED, Loader::parseBoolean, true);
            int stage2Minutes = get(STAGE2_MINUTES, v -> parseInt(v, 1), 60);
            int stage2Unimproved = get(STAGE2_UNIMPROVED_SECONDS, v -> parseInt(v, 1), 60);
            Long seed = get(RANDOM_SEED, Loader::parseLong, null);
            String validation = get(TRIP_VALIDATION, Loader::parseValidation, "lenient");
            Set<String> regions = get(MISC_NEUTRAL_REGIONS, Loader::parseRegions, Set.of("r_other"));
            int historyDays = get(TRIP_HISTORY_DAYS, v -> parseInt(v, -1), -1);
            boolean writeScoreFiles = get(WRITE_SCORE_FILES, Loader::parseBoolean, true);
            boolean allowCut = get(ALLOW_CUT_FIXED_TRIP, Loader::parseBoolean, true);
            if (find(MOVE_THREAD_COUNT) != null) {
                warnings.add(MOVE_THREAD_COUNT + ": Enterprise edition only, ignored");
            }
            return new SolverParams(stage1Minutes, stage1Unimproved, stopWhenFeasible, stage2Enabled,
                    stage2Minutes, stage2Unimproved, seed, validation, regions, historyDays,
                    writeScoreFiles, allowCut, warnings);
        }

        // ---- value parsers (null = bad value)

        static Integer parseInt(Object v, int min) {
            Long l = parseLong(v);
            if (l == null || l < min || l > Integer.MAX_VALUE) return null;
            return (int) (long) l;
        }

        static Long parseLong(Object v) {
            if (v instanceof Integer || v instanceof Long || v instanceof Short || v instanceof Byte) {
                return ((Number) v).longValue();
            }
            if (v == null || v instanceof Boolean || v instanceof Number) return null; // incl. 5.0 / big integers
            try {
                return Long.parseLong(String.valueOf(v).trim());
            } catch (NumberFormatException e) {
                return null;
            }
        }

        static Boolean parseBoolean(Object v) {
            if (v instanceof Boolean b) return b;
            if (v == null) return null;
            switch (String.valueOf(v).trim().toLowerCase(Locale.ROOT)) {
                case "true": case "yes": case "1": return Boolean.TRUE;
                case "false": case "no": case "0": return Boolean.FALSE;
                default: return null;
            }
        }

        static String parseValidation(Object v) {
            if (v == null) return null;
            String s = String.valueOf(v).trim().toLowerCase(Locale.ROOT);
            return (s.equals("lenient") || s.equals("strict")) ? s : null;
        }

        static Set<String> parseRegions(Object v) {
            Set<String> out = new LinkedHashSet<>();
            if (v == null) return out; // "misc_neutral_regions:" with no value = empty set
            if (v instanceof Collection<?> c) {
                for (Object o : c) {
                    if (o == null) continue;
                    String s = String.valueOf(o).trim();
                    if (!s.isEmpty()) out.add(s);
                }
            } else {
                for (String part : String.valueOf(v).split(",")) {
                    String s = part.trim();
                    if (!s.isEmpty()) out.add(s);
                }
            }
            return out;
        }
    }
}

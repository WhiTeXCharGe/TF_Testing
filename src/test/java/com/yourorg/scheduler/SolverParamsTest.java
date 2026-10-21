package com.yourorg.scheduler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.yourorg.scheduler.build.PlanBuilder;
import com.yourorg.scheduler.config.SolverParams;
import com.yourorg.scheduler.solver.RunResult;
import com.yourorg.scheduler.solver.ScoreLogger;
import com.yourorg.scheduler.solver.TimerConfig;
import com.yourorg.scheduler.solver.TwoStageRunner;

/** Parsing and precedence of the run parameters (CLI > env > params file > default), plus the score-file switch. */
class SolverParamsTest {

    static final Path TINY_FIXTURE = Path.of("src", "test", "resources", "fixtures", "tiny");

    @TempDir
    Path tmp;

    Path missing() { return tmp.resolve("no-such-SolverParams.yaml"); }

    Path file(String yaml) throws Exception {
        Path f = tmp.resolve("SolverParams.yaml");
        Files.writeString(f, yaml, StandardCharsets.UTF_8);
        return f;
    }

    static SolverParams load(String[] cli, Map<String, String> env, Path f) {
        return SolverParams.load(cli, env, f);
    }

    static boolean warned(SolverParams p, String fragment) {
        return p.warnings().stream().anyMatch(w -> w.contains(fragment));
    }

    // ------------------------------------------------------------------ defaults

    @Test
    void defaults_when_nothing_given() {
        SolverParams p = load(new String[0], Map.of(), missing());
        assertEquals(300, p.stage1Minutes());
        assertEquals(600, p.stage1UnimprovedSeconds());
        assertTrue(p.stage1StopWhenFeasible());
        assertTrue(p.stage2Enabled());
        assertEquals(60, p.stage2Minutes());
        assertEquals(60, p.stage2UnimprovedSeconds());
        assertNull(p.randomSeed());
        assertEquals("lenient", p.tripValidation());
        assertEquals(Set.of("r_other"), p.miscNeutralRegions());
        assertEquals(-1, p.tripHistoryDays());
        assertTrue(p.writeScoreFiles());
        assertTrue(p.allowCutFixedTrip());
        assertEquals(List.of(), p.warnings());
    }

    @Test
    void defaults_equal_TimerConfig_defaults() {
        assertEquals(TimerConfig.defaults(), load(new String[0], Map.of(), missing()).timers());
    }

    @Test
    void timers_map_every_timer_field() throws Exception {
        SolverParams p = load(new String[] {"--stage1_minutes=5", "--stage1_unimproved_seconds=6",
                "--stage1_stop_when_feasible=false", "--stage2_enabled=no", "--stage2_minutes=7",
                "--stage2_unimproved_seconds=8", "--random_seed=-9"}, Map.of(), missing());
        assertEquals(new TimerConfig(5, 6, false, false, 7, 8, -9L), p.timers());
    }

    // ------------------------------------------------------------------ precedence

    @Test
    void precedence_cli_over_env_over_file_over_default() throws Exception {
        String[] cli = {"--stage1_minutes=11"};
        Map<String, String> env = Map.of("TF_STAGE1_MINUTES", "22");
        Path f = file("stage1_minutes: 33\n");

        assertEquals(11, load(cli, env, f).stage1Minutes());                         // all four sources
        assertEquals(22, load(new String[0], env, f).stage1Minutes());               // CLI removed
        assertEquals(33, load(new String[0], Map.of(), f).stage1Minutes());          // env removed
        assertEquals(300, load(new String[0], Map.of(), missing()).stage1Minutes()); // file removed
        assertEquals(11, load(cli, Map.of(), f).stage1Minutes());                    // CLI beats file
        assertEquals(11, load(cli, env, missing()).stage1Minutes());                 // CLI beats env
    }

    @Test
    void cli_space_separated_value() {
        SolverParams p = load(new String[] {"--stage1_minutes", "7", "--stage2_enabled", "false"}, Map.of(), missing());
        assertEquals(7, p.stage1Minutes());
        assertFalse(p.stage2Enabled());
        assertEquals(List.of(), p.warnings());
    }

    @Test
    void cli_keys_ignore_case_underscore_and_dash() {
        SolverParams p = load(new String[] {"--Stage1-Minutes=8", "--STAGE2MINUTES=9"}, Map.of(), missing());
        assertEquals(8, p.stage1Minutes());
        assertEquals(9, p.stage2Minutes());
    }

    @Test
    void file_values_are_read_with_native_yaml_types() throws Exception {
        SolverParams p = load(new String[0], Map.of(), file(
                "stage1_minutes: 12\nstage2_enabled: false\nrandom_seed: 42\ntrip_validation: STRICT\n"
                + "trip_history_days: 30\nwrite_score_files: no\n"));
        assertEquals(12, p.stage1Minutes());
        assertFalse(p.stage2Enabled());
        assertEquals(42L, p.randomSeed());
        assertEquals("strict", p.tripValidation());
        assertEquals(30, p.tripHistoryDays());
        assertFalse(p.writeScoreFiles());
        assertEquals(List.of(), p.warnings());
    }

    @Test
    void file_missing_is_not_an_error() {
        SolverParams p = load(new String[0], Map.of(), missing());
        assertEquals(List.of(), p.warnings());
        assertEquals(300, p.stage1Minutes());
    }

    @Test
    void file_malformed_yaml_falls_back_with_warning() throws Exception {
        SolverParams p = load(new String[0], Map.of(), file("stage1_minutes: [1, 2\n  : :\n- nope"));
        assertEquals(1, p.warnings().size(), p.warnings().toString());
        assertTrue(p.warnings().get(0).contains("params file"), p.warnings().get(0));
        assertEquals(TimerConfig.defaults(), p.timers());
    }

    @Test
    void file_not_a_map_falls_back_with_warning() throws Exception {
        SolverParams p = load(new String[0], Map.of(), file("- a\n- b\n"));
        assertEquals(1, p.warnings().size());
        assertEquals(TimerConfig.defaults(), p.timers());
    }

    @Test
    void malformed_file_does_not_disable_cli_and_env() throws Exception {
        SolverParams p = load(new String[] {"--stage1_minutes=4"}, Map.of("TF_STAGE2_MINUTES", "5"),
                file("{ broken: [\n"));
        assertEquals(4, p.stage1Minutes());
        assertEquals(5, p.stage2Minutes());
        assertEquals(1, p.warnings().size());
    }

    @Test
    void empty_file_is_fine() throws Exception {
        SolverParams p = load(new String[0], Map.of(), file(""));
        assertEquals(List.of(), p.warnings());
        assertEquals(TimerConfig.defaults(), p.timers());
    }

    // ------------------------------------------------------------------ bad values

    @Test
    void bad_values_fall_back_with_warning() throws Exception {
        for (String bad : new String[] {"-5", "abc", "0", "1.5"}) {
            SolverParams p = load(new String[] {"--stage1_minutes=" + bad}, Map.of(), missing());
            assertEquals(300, p.stage1Minutes(), bad);
            assertEquals(List.of("bad value for stage1_minutes: \"" + bad + "\" (using default 300)"),
                    p.warnings(), bad);
        }
        SolverParams b = load(new String[] {"--stage2_enabled=maybe"}, Map.of(), missing());
        assertTrue(b.stage2Enabled());
        assertTrue(warned(b, "bad value for stage2_enabled: \"maybe\" (using default true)"), b.warnings().toString());

        SolverParams t = load(new String[] {"--trip_validation=turbo"}, Map.of(), missing());
        assertEquals("lenient", t.tripValidation());
        assertTrue(warned(t, "bad value for trip_validation: \"turbo\""), t.warnings().toString());

        SolverParams h = load(new String[] {"--trip_history_days=-2"}, Map.of(), missing());
        assertEquals(-1, h.tripHistoryDays());
        assertEquals(1, h.warnings().size());

        SolverParams s = load(new String[] {"--random_seed=xyz"}, Map.of(), missing());
        assertNull(s.randomSeed());
        assertEquals(1, s.warnings().size());
    }

    @Test
    void bad_value_falls_back_to_default_not_to_lower_source() throws Exception {
        SolverParams p = load(new String[] {"--stage1_minutes=abc"}, Map.of("TF_STAGE1_MINUTES", "22"),
                file("stage1_minutes: 33\n"));
        assertEquals(300, p.stage1Minutes());
        assertEquals(1, p.warnings().size());
    }

    @Test
    void bad_value_in_env_and_file_also_warn() throws Exception {
        SolverParams e = load(new String[0], Map.of("TF_STAGE2_MINUTES", "zero"), missing());
        assertEquals(60, e.stage2Minutes());
        assertTrue(warned(e, "bad value for stage2_minutes"));
        SolverParams f = load(new String[0], Map.of(), file("stage2_minutes: -3\n"));
        assertEquals(60, f.stage2Minutes());
        assertTrue(warned(f, "bad value for stage2_minutes"));
    }

    @Test
    void boolean_spellings() {
        for (String yes : new String[] {"true", "TRUE", "Yes", "yes", "1"}) {
            assertTrue(load(new String[] {"--stage2_enabled=" + yes}, Map.of(), missing()).stage2Enabled(), yes);
            assertEquals(List.of(), load(new String[] {"--stage2_enabled=" + yes}, Map.of(), missing()).warnings());
        }
        for (String no : new String[] {"false", "False", "NO", "no", "0"}) {
            SolverParams p = load(new String[] {"--stage2_enabled=" + no}, Map.of(), missing());
            assertFalse(p.stage2Enabled(), no);
            assertEquals(List.of(), p.warnings(), no);
        }
    }

    // ------------------------------------------------------------------ misc_neutral_regions

    @Test
    void neutral_regions_list_and_csv_and_empty() throws Exception {
        assertEquals(Set.of("a", "b"),
                load(new String[0], Map.of(), file("misc_neutral_regions: [a, b]\n")).miscNeutralRegions());
        assertEquals(Set.of("a", "b"),
                load(new String[0], Map.of(), file("misc_neutral_regions:\n  - a\n  - b\n")).miscNeutralRegions());
        assertEquals(Set.of("x", "y", "z"),
                load(new String[] {"--misc_neutral_regions=x, y,z"}, Map.of(), missing()).miscNeutralRegions());
        assertEquals(Set.of("p", "q"),
                load(new String[0], Map.of("TF_MISC_NEUTRAL_REGIONS", "p,q"), missing()).miscNeutralRegions());
        assertEquals(Set.of(), load(new String[] {"--misc_neutral_regions="}, Map.of(), missing())
                .miscNeutralRegions());
        assertEquals(Set.of(), load(new String[0], Map.of(), file("misc_neutral_regions: []\n"))
                .miscNeutralRegions());
        assertEquals(Set.of(), load(new String[0], Map.of(), file("misc_neutral_regions: ''\n"))
                .miscNeutralRegions());
        SolverParams ok = load(new String[0], Map.of(), file("misc_neutral_regions: [a]\n"));
        assertEquals(List.of(), ok.warnings());
    }

    // ------------------------------------------------------------------ aliases

    @Test
    void alias_spellings_for_allow_cut_fixed_trip() throws Exception {
        assertFalse(load(new String[0], Map.of(), file("allow_cut_fixed_trip: false\n")).allowCutFixedTrip());
        assertFalse(load(new String[0], Map.of(), file("AllowCutFixedFifo: false\n")).allowCutFixedTrip());
        assertFalse(load(new String[] {"--allow_cut_fixed_trip=false"}, Map.of(), missing()).allowCutFixedTrip());
        assertFalse(load(new String[] {"--allowCutFixedFifo", "no"}, Map.of(), missing()).allowCutFixedTrip());
        assertFalse(load(new String[0], Map.of("ALLOW_CUT_FIXED_TRIP", "false"), missing()).allowCutFixedTrip());
        assertFalse(load(new String[0], Map.of("ALLOWCUTFIXEDFIFO", "0"), missing()).allowCutFixedTrip());
        assertFalse(load(new String[0], Map.of("TF_ALLOW_CUT_FIXED_TRIP", "false"), missing()).allowCutFixedTrip());
        assertFalse(load(new String[0], Map.of("TF_ALLOWCUTFIXEDFIFO", "false"), missing()).allowCutFixedTrip());
        // none of these spellings is "unknown"
        assertEquals(List.of(), load(new String[] {"--allowCutFixedFifo=false"}, Map.of(), missing()).warnings());
        // TF_ spelling wins over the bare one; CLI wins over both
        assertTrue(load(new String[0], Map.of("TF_ALLOW_CUT_FIXED_TRIP", "true", "ALLOW_CUT_FIXED_TRIP", "false"),
                missing()).allowCutFixedTrip());
        assertTrue(load(new String[] {"--allow_cut_fixed_trip=true"}, Map.of("ALLOW_CUT_FIXED_TRIP", "false"),
                missing()).allowCutFixedTrip());
    }

    // ------------------------------------------------------------------ unknown / ignored keys

    @Test
    void unknown_key_is_ignored_with_warning() throws Exception {
        SolverParams cli = load(new String[] {"--bogus=1"}, Map.of(), missing());
        assertEquals(List.of("unknown parameter: bogus"), cli.warnings());
        assertEquals(TimerConfig.defaults(), cli.timers());
        SolverParams yaml = load(new String[0], Map.of(), file("bogus_key: 1\nstage1_minutes: 9\n"));
        assertEquals(List.of("unknown parameter: bogus_key"), yaml.warnings());
        assertEquals(9, yaml.stage1Minutes());
        // unrelated environment variables are not parameters
        assertEquals(List.of(), load(new String[0], Map.of("TF_BOGUS", "1", "PATH", "/bin"), missing()).warnings());
    }

    @Test
    void move_thread_count_warns_enterprise_only() throws Exception {
        for (SolverParams p : List.of(
                load(new String[] {"--move_thread_count=4"}, Map.of(), missing()),
                load(new String[0], Map.of("TF_MOVE_THREAD_COUNT", "4"), missing()),
                load(new String[0], Map.of(), file("move_thread_count: 4\n")))) {
            assertEquals(1, p.warnings().size(), p.warnings().toString());
            assertTrue(p.warnings().get(0).contains("move_thread_count"), p.warnings().get(0));
            assertTrue(p.warnings().get(0).contains("Enterprise edition only, ignored"), p.warnings().get(0));
            assertEquals(TimerConfig.defaults(), p.timers());
        }
    }

    // ------------------------------------------------------------------ Main argument splitting

    @Test
    void positional_args_are_not_flags() {
        Main.SplitArgs none = Main.splitArgs(new String[0]);
        assertEquals(List.of(), none.positional());
        assertEquals(0, none.flags().length);

        Main.SplitArgs a = Main.splitArgs(new String[] {"Env.yaml", "--stage1_minutes=3", "Sched.yaml"});
        assertEquals(List.of("Env.yaml", "Sched.yaml"), a.positional());
        assertEquals(List.of("--stage1_minutes=3"), List.of(a.flags()));

        // "--key value": the value belongs to the flag, not to the positional arguments
        Main.SplitArgs b = Main.splitArgs(new String[] {"Env.yaml", "Sched.yaml", "--stage1_minutes", "3",
                "--stage2_enabled", "false"});
        assertEquals(List.of("Env.yaml", "Sched.yaml"), b.positional());
        assertEquals(List.of("--stage1_minutes", "3", "--stage2_enabled", "false"), List.of(b.flags()));
        SolverParams p = load(b.flags(), Map.of(), missing());
        assertEquals(3, p.stage1Minutes());
        assertFalse(p.stage2Enabled());
    }

    @Test
    void params_file_location_prefers_TF_PARAMS_FILE_then_INPUT_DIR_then_default() {
        assertEquals(Path.of("/x/p.yaml"), Main.paramsFile(Map.of("TF_PARAMS_FILE", "/x/p.yaml", "INPUT_DIR", "/in")));
        assertEquals(Path.of("/in", "SolverParams.yaml"), Main.paramsFile(Map.of("INPUT_DIR", "/in")));
        assertEquals(Path.of("/work/input", "SolverParams.yaml"), Main.paramsFile(Map.of()));
    }

    // ------------------------------------------------------------------ describe

    @Test
    void describe_lists_the_effective_values() {
        String d = load(new String[0], Map.of(), missing()).describe();
        assertTrue(d.contains("stage1=300min/600s"), d);
        assertTrue(d.contains("stage2=60min/60s"), d);
    }

    // ------------------------------------------------------------------ write_score_files

    static final List<String> SCORE_FILES =
            List.of("initial_score.txt", "stage1_score.txt", "final_score.txt", "score_progress.txt");

    /** existence + size + mtime per file in the working directory; a changed fingerprint means "touched". */
    static String fingerprint(String name) throws Exception {
        Path f = Path.of(name);
        return Files.exists(f) ? Files.size(f) + "@" + Files.getLastModifiedTime(f).toMillis() : "absent";
    }

    @Test
    void write_score_files_false_skips_score_text_files() throws Exception {
        Path dir = tmp.resolve("run");
        Files.createDirectories(dir);
        Path env = dir.resolve("EnvConfig.yaml");
        Path sched = dir.resolve("Schedule.yaml");
        Files.copy(TINY_FIXTURE.resolve("EnvConfig.yaml"), env, StandardCopyOption.REPLACE_EXISTING);
        Files.copy(TINY_FIXTURE.resolve("Schedule.yaml"), sched, StandardCopyOption.REPLACE_EXISTING);

        java.util.Map<String, String> before = new java.util.LinkedHashMap<>();
        for (String n : SCORE_FILES) before.put(n, fingerprint(n));
        String solverLogBefore = fingerprint("solver_log.txt");

        boolean previous = ScoreLogger.writeScoreFiles();
        ScoreLogger.setWriteScoreFiles(false);
        try {
            PlanBuilder.BuiltProblem problem = PlanBuilder.buildProblem(env.toString(), sched.toString());
            // short budget: 1 minute cap, stops after 1 s without improvement, stage 2 off
            RunResult rr = TwoStageRunner.run(problem, new TimerConfig(1, 1, true, false, 1, 1, 42L));
            assertTrue(rr.plan != null);
        } finally {
            ScoreLogger.setWriteScoreFiles(previous);
        }

        for (String n : SCORE_FILES) {
            assertEquals(before.get(n), fingerprint(n), n + " must not be created or modified");
        }
        // other logs are unaffected by the switch
        assertFalse(solverLogBefore.equals(fingerprint("solver_log.txt")), "solver_log.txt is still written");
    }

    @Test
    void score_logger_default_writes_score_files() {
        assertTrue(ScoreLogger.writeScoreFiles(), "default must stay true so today's behaviour is unchanged");
    }
}

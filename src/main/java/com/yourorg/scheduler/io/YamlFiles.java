package com.yourorg.scheduler.io;

import java.io.IOException;
import java.io.InputStream;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.LoaderOptions;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.SafeConstructor;

import com.yourorg.scheduler.model.ParsedOpTaskRef;

/** YAML file access and small parsing helpers for the schedule / environment files. */
public final class YamlFiles {

    private YamlFiles() {}

    public static final DateTimeFormatter DF = DateTimeFormatter.ofPattern("yyyy/MM/dd");

    static final Pattern PHASE_NUM_PAT = Pattern.compile("p(\\d+)", Pattern.CASE_INSENSITIVE);

    @SuppressWarnings("unchecked")
    public static Map<String, Object> loadYaml(String path) throws IOException {
        try (InputStream in = Files.newInputStream(Paths.get(path))) {
            LoaderOptions opts = new LoaderOptions();
            opts.setCodePointLimit(10 * 1024 * 1024); // ≈ 5 MB

            Yaml yaml = new Yaml(new SafeConstructor(opts));
            return yaml.load(in);
        }
    }

    public static void saveYaml(String path, Map<String, Object> root) throws IOException {
        DumperOptions opt = new DumperOptions();
        opt.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK);
        opt.setPrettyFlow(true);
        Yaml yaml = new Yaml(opt);
        try (Writer w = Files.newBufferedWriter(Paths.get(path))) {
            yaml.dump(root, w);
        }
    }

    public static String safeStr(Object o) { return o == null ? "" : String.valueOf(o); }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> asMap(Object o) {
        return (o instanceof Map) ? (Map<String, Object>) o : null;
    }

    public static int parseInt(Object o, int def) {
        if (o == null) return def;
        try { return Integer.parseInt(String.valueOf(o)); } catch (Exception e) { return def; }
    }

    public static int phaseNumFromId(String pid) {
        if (pid == null) return 0;
        String t = pid.trim();
        Matcher m = PHASE_NUM_PAT.matcher(t);

        int last = 0;
        while (m.find()) {
            try {
                last = Integer.parseInt(m.group(1));
            } catch (Exception ignore) {}
        }
        return last; // 0 if not found
    }

    public static ParsedOpTaskRef parseOpTaskRef(String raw) {
        ParsedOpTaskRef out = new ParsedOpTaskRef();
        String t = safeStr(raw).trim();
        if (t.isBlank()) {
            out.module = "";
            out.phaseToken = "";
            out.phaseNum = 0;
            out.opToken = "";
            return out;
        }

        // Supports both:
        //   e1_p1, e1_p1o2
        //   e1p1,  e1p1o2
        Matcher m = Pattern.compile("^(.*?)(?:_)?(p\\d+(?:o\\d+)?)$", Pattern.CASE_INSENSITIVE).matcher(t);
        if (m.matches()) {
            out.module = safeStr(m.group(1));
            while (out.module.endsWith("_") || out.module.endsWith("-")) {
                out.module = out.module.substring(0, out.module.length() - 1);
            }
            out.opToken = safeStr(m.group(2));
            String phaseOnly = out.opToken.replaceFirst("(?i)(p\\d+).*$", "$1");
            out.phaseToken = phaseOnly;
            out.phaseNum = phaseNumFromId(phaseOnly);
            return out;
        }

        // Fallback: keep old behavior, but still try to extract a phase number safely.
        out.module = t;
        out.opToken = t;
        out.phaseNum = phaseNumFromId(t);
        out.phaseToken = (out.phaseNum > 0) ? ("p" + out.phaseNum) : "";
        return out;
    }

    public static Integer dayIdFromDate(LocalDate planStart, String ymd) {
        LocalDate d = LocalDate.parse(ymd.replace("-", "/"), DF);
        return (int) (d.toEpochDay() - planStart.toEpochDay());
    }
}

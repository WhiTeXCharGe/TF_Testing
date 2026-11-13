import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))      # weekly_planning/ (the folder of common)
from common import tf_config as tfc, timefold_api as api   # noqa: E402


class Parse(unittest.TestCase):
    def test_comments_spaces_quotes_and_inline_comments(self):
        values, unknown = tfc.parse_text(
            "\ufeff# a comment\n; another\n\nmode = Local\nAzure-URL = \"https://x.example.com/\"\nstage1_minutes = 45   # minutes\n"
            "schedule = 'C:/a b/03_Schedule.yaml'\nrun_id =\n", tfc.SUBMIT_KEYS)
        self.assertEqual("Local", values["mode"])
        self.assertEqual("https://x.example.com/", values["azure_url"])
        self.assertEqual("45", values["stage1_minutes"])
        self.assertEqual("C:/a b/03_Schedule.yaml", values["schedule"])
        self.assertEqual("", values["run_id"])
        self.assertEqual([], unknown)

    def test_aliases_and_unknown_keys(self):
        values, unknown = tfc.parse_text("url = https://x\njob-id = run7\nstage_1_minutes = 3\n", tfc.SUBMIT_KEYS)
        self.assertEqual(("https://x", "run7"), (values["azure_url"], values["run_id"]))
        self.assertEqual(["stage_1_minutes"], unknown)

    def test_a_hash_inside_a_path_without_a_space_before_it_is_kept(self):
        values, _ = tfc.parse_text("schedule = D:/week#45/03_Schedule.yaml\n")
        self.assertEqual("D:/week#45/03_Schedule.yaml", values["schedule"])


class Load(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def write(self, text):
        p = self.dir / "config.txt"
        p.write_text(text, encoding="utf-8")
        return str(p)

    def load(self, path):
        return tfc.load(path, None, tfc.SUBMIT_KEYS + tfc.RECEIVE_KEYS)

    def test_without_the_option_the_default_file_is_used_and_may_be_missing(self):
        self.write("mode = online\n")
        self.assertEqual("online", tfc.load(None, self.dir / "config.txt", tfc.SUBMIT_KEYS).text("mode"))
        empty = tfc.load(None, self.dir / "none.txt", tfc.SUBMIT_KEYS)
        self.assertIsNone(empty.path)
        self.assertIsNone(empty.text("mode"))

    def test_keys_of_the_other_tool_are_reported_as_unknown(self):
        cfg = tfc.load(self.write("mode = local\nwatch = 5\nschedule = a.yaml\n"), None, tfc.RECEIVE_KEYS)
        self.assertEqual(["schedule"], cfg.unknown)

    def test_a_config_that_was_asked_for_must_exist(self):
        with self.assertRaises(tfc.ConfigError):
            tfc.load(str(self.dir / "missing.txt"), self.dir / "config.txt", tfc.SUBMIT_KEYS)

    def test_typed_getters(self):
        cfg = self.load(self.write("watch = 30\nstage2_enabled = No\nallow_cut_fixed_trip = TRUE\nschedule = a/b.yaml\nrun_id =\n"))
        self.assertEqual(30, cfg.int("watch"))
        self.assertIs(False, cfg.bool("stage2_enabled"))
        self.assertIs(True, cfg.bool("allow_cut_fixed_trip"))
        self.assertIsNone(cfg.int("random_seed"))
        self.assertIsNone(cfg.text("run_id"))
        self.assertEqual(str(self.dir / "a" / "b.yaml"), cfg.file("schedule"))

    def test_wrong_types_name_the_setting(self):
        cfg = self.load(self.write("watch = often\nstage2_enabled = maybe\n"))
        with self.assertRaisesRegex(tfc.ConfigError, "watch"):
            cfg.int("watch")
        with self.assertRaisesRegex(tfc.ConfigError, "stage2_enabled"):
            cfg.bool("stage2_enabled")


class Connection(unittest.TestCase):
    def cfg(self, text):
        return tfc.Config(tfc.parse_text(text)[0], path=Path("config.txt"))

    def test_local_defaults_to_the_local_server(self):
        conn = tfc.connection(self.cfg("mode = local\n"))
        self.assertEqual(("local", "http://localhost:3001"), tuple(conn))

    def test_local_url_and_trailing_slash(self):
        self.assertEqual("http://localhost:4000", tfc.connection(self.cfg("mode=local\nlocal_url=http://localhost:4000/\n")).url)

    def test_online_uses_azure_url_and_ignores_local_url(self):
        conn = tfc.connection(self.cfg("mode=Online\nazure_url=https://app.example.com/\nlocal_url=http://localhost:4000\n"))
        self.assertEqual(("online", "https://app.example.com"), tuple(conn))

    def test_online_needs_an_address(self):
        with self.assertRaisesRegex(tfc.ConfigError, "azure_url is not set"):
            tfc.connection(self.cfg("mode=online\n"))

    def test_bad_addresses_and_modes(self):
        for text, word in (("mode=online\nazure_url=app.example.com\n", "usable address"),
                           ("mode=online\nazure_url=ftp://x\n", "usable address"),
                           ("mode=online\nazure_url=https://<your-solver-api>.x.io\n", "placeholder"),
                           ("mode=cloud\n", "unknown mode"), ("", "mode is not set")):
            with self.assertRaisesRegex(tfc.ConfigError, word):
                tfc.connection(self.cfg(text))

    def test_command_line_wins_over_config(self):
        cfg = self.cfg("mode=online\nazure_url=https://app.example.com\n")
        self.assertEqual("local", tfc.connection(cfg, mode="local").mode)
        self.assertEqual("https://other.example.com", tfc.connection(cfg, api_url="https://other.example.com/").url)

    def test_the_run_record_is_only_the_last_resort(self):
        record = {"mode": "local", "apiUrl": "http://localhost:3999"}
        self.assertEqual("http://localhost:3999", tfc.connection(self.cfg(""), record=record).url)           # nothing else says
        self.assertEqual("http://localhost:3001", tfc.connection(self.cfg("mode=local\n"), record=record).url)  # config wins
        self.assertEqual("online", tfc.connection(self.cfg("mode=online\nazure_url=https://a.b\n"), record=record).mode)


class RunId(unittest.TestCase):
    def test_run_id_check(self):
        for good in ("20261111_103000", "wk45-final", "a" * 64):
            self.assertEqual(good, api.check_run_id(good))
        for bad in ("", "a b", "a/b", "a" * 65, "週45"):
            with self.assertRaises(api.ApiError):
                api.check_run_id(bad)


if __name__ == "__main__":
    unittest.main()

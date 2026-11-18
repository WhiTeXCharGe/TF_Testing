"""tool_runner: runs the tools of weekly_planning from their files as main(argv)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # weekly_planning/ (the folder of common)
from common.tool_runner import TOOLS, ToolRunner   # noqa: E402


class TempCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)


class Runner(TempCase):
    def runner_with(self, source):
        tools = self.tools(["2-5"])
        (tools / TOOLS["2-5"][0] / TOOLS["2-5"][1]).write_text(source, encoding="utf-8")
        return ToolRunner(tools)

    def tools(self, steps):
        """A weekly_planning folder with `common` and a stub script for each step."""
        tools = self.root / "tools"
        (tools / "common").mkdir(parents=True, exist_ok=True)
        for step in steps:
            script = tools / TOOLS[step][0] / TOOLS[step][1]
            script.parent.mkdir(parents=True, exist_ok=True)
            script.write_text("def main(argv):\n    return 0\n", encoding="utf-8")
        return tools

    def test_the_output_arrives_line_by_line_and_the_exit_code_is_returned(self):
        runner = self.runner_with("import sys\ndef main(argv):\n    print('a')\n    print('b', 'c')\n"
                                  "    print('oops', file=sys.stderr)\n    sys.stdout.write('no newline')\n    return 3\n")
        seen = []
        result = runner.run("2-5", ["--x"], seen.append)
        self.assertEqual(3, result.rc)
        self.assertEqual(["a", "b c", "oops", "no newline"], seen)
        self.assertEqual(["oops"], result.errors)
        self.assertEqual("oops", result.last_error())

    def test_none_is_success_and_exit_and_exceptions_are_caught(self):
        self.assertEqual(0, self.runner_with("def main(argv):\n    pass\n").run("2-5", []).rc)
        self.assertEqual(2, self.runner_with("import sys\ndef main(argv):\n    sys.exit(2)\n").run("2-5", []).rc)
        self.assertEqual(0, self.runner_with("import sys\ndef main(argv):\n    sys.exit()\n").run("2-5", []).rc)
        self.assertEqual(1, self.runner_with("import sys\ndef main(argv):\n    sys.exit('bad')\n").run("2-5", []).rc)
        result = self.runner_with("def main(argv):\n    raise ValueError('wrong value')\n").run("2-5", [])
        self.assertEqual(1, result.rc)
        self.assertEqual("ValueError: wrong value", result.last_error())

    def test_argparse_errors_do_not_stop_the_program(self):
        runner = self.runner_with("import argparse\ndef main(argv):\n    ap = argparse.ArgumentParser()\n"
                                  "    ap.add_argument('--needed', required=True)\n    ap.parse_args(argv)\n")
        result = runner.run("2-5", [])
        self.assertEqual(2, result.rc)
        self.assertIn("--needed", result.last_error())

    def test_arguments_are_given_as_text_and_warnings_are_not_printed(self):
        runner = self.runner_with("import warnings\ndef main(argv):\n    warnings.warn('noise')\n    print(argv)\n")
        seen = []
        runner.run("2-5", [Path("a"), 3], seen.append)
        self.assertEqual(["['a', '3']"], seen)

    def test_missing_scripts_are_listed(self):
        runner = ToolRunner(self.root / "tools")
        self.assertEqual(2, len(runner.missing(["2-2-3"])))               # common and the script
        self.tools(["2-2-3", "2-4", "2-5", "2-6"])
        self.assertEqual([], runner.missing(["2-2-3", "2-4", "2-5", "2-6"]))
        self.assertEqual(1, len(runner.missing(["3-1"])))


if __name__ == "__main__":
    unittest.main()

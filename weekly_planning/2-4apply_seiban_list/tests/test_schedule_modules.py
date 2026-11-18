"""Tests of schedule_modules.py (check a row, add a module, delete a module).  Run from the tool folder:  python -m unittest discover -s tests -v"""
import copy
import unittest

from envs import TEMPLATE_LIKE_WF_TOOL, TEMPLATE_WITH_P1, make_env, make_schedule, plan_row, template
from common import env_workflows as wfs   # noqa: E402
import schedule_modules as sm


def sched():
    return make_schedule()["schedule"]


def ref_and_template(ops=TEMPLATE_WITH_P1):
    """A new workflow (wf1) for the template, as step B makes it."""
    env = make_env()
    t = template(2, ops)
    return wfs.ensure_workflow(env, t, []), t


class Adding(unittest.TestCase):
    def test_a_module_for_a_new_workflow(self):
        ref, t = ref_and_template()
        s = sched()
        task = sm.add_module(s, "530N00005A", ref, "f3", t, plan_row())
        self.assertIs(task, s["workflow_task_list"][-1])
        self.assertEqual(("e3", "530N00005A", "wf1", "f3"), (task["id"], task["name"], task["workflow"], task["fab"]))
        self.assertEqual(["e3p1", "e3p2", "e3p4"], [p["id"] for p in task["phase_task_list"]])       # only the processes of the template
        self.assertEqual([("wf1p1", "p1"), ("wf1p2", "p2"), ("wf1p4", "p4")], [(p["phase"], p["name"]) for p in task["phase_task_list"]])
        self.assertEqual([("2026/12/01", "2026/12/10"), ("2026/12/05", "2026/12/20"), ("2026/12/22", "2027/01/10")],
                         [(p["start_date"], p["end_date"]) for p in task["phase_task_list"]])
        ops = [o for p in task["phase_task_list"] for o in p["operation_task_list"]]
        self.assertEqual(["e3p1o1", "e3p1o2", "e3p2o1", "e3p4o1"], [o["id"] for o in ops])
        self.assertEqual(["wf1p1o1", "wf1p1o2", "wf1p2o1", "wf1p4o1"], [o["operation"] for o in ops])
        self.assertEqual(["Mech", "Elec", "Mech", "QC"], [o["name"] for o in ops])
        self.assertEqual([160, 40, 576, 288], [o["workload_hours"] for o in ops])

    def test_the_recommended_workers_are_not_written(self):
        ref, t = ref_and_template()
        task = sm.add_module(sched(), "530N00005A", ref, "f3", t, plan_row())
        for p in task["phase_task_list"]:
            for o in p["operation_task_list"]:
                self.assertNotIn("recommends_worker_min", o)
                self.assertNotIn("recommends_worker_max", o)
                self.assertEqual({"id", "name", "description", "operation", "workload_hours", "color_code"}, set(o))

    def test_a_module_for_an_existing_workflow_uses_its_names(self):
        env = make_env()
        t = template(1, TEMPLATE_LIKE_WF_TOOL)
        ref = wfs.ensure_workflow(env, t, [])
        task = sm.add_module(sched(), "530N00005A", ref, "f1", t, plan_row())
        self.assertEqual("wf_tool", task["workflow"])
        self.assertEqual([("p2", "Hardware Setup"), ("p3", "Function Setup"), ("p4", "Acceptance Inspection")],
                         [(p["phase"], p["name"]) for p in task["phase_task_list"]])
        self.assertEqual(["p2o1", "p2o2", "p3o1", "p4o1"], [o["operation"] for p in task["phase_task_list"] for o in p["operation_task_list"]])

    def test_hours_that_are_whole_numbers_are_written_as_integers(self):
        ref, t = ref_and_template([(2, 1, "Mech", 576.0, 1, 2), (2, 2, "Elec", 7.5, 1, 1)])
        task = sm.add_module(sched(), "530N00005A", ref, "f3", t, plan_row())
        hours = [o["workload_hours"] for o in task["phase_task_list"][0]["operation_task_list"]]
        self.assertEqual([576, 7.5], hours)
        self.assertIsInstance(hours[0], int)

    def test_the_new_id_is_above_every_module_and_not_used_by_any_task(self):
        s = sched()
        self.assertEqual("e3", sm.next_module_id(s))
        s["misc_task_list"].append({"id": "e3p1", "name": "x"})                  # an id that would clash with the phase of e3
        s["workflow_task_list"][0]["id"] = "e9"
        self.assertEqual("e10", sm.next_module_id(s))
        s["workflow_task_list"][0]["id"] = "e1"
        self.assertEqual("e4", sm.next_module_id(s))                              # e3 is free as an id but e3p1 is taken

    def test_two_modules_get_two_ids(self):
        ref, t = ref_and_template()
        s = sched()
        a = sm.add_module(s, "530N00005A", ref, "f3", t, plan_row())
        b = sm.add_module(s, "530N00006A", ref, "f3", t, plan_row())
        self.assertEqual(("e3", "e4"), (a["id"], b["id"]))

    def test_the_other_parts_of_the_schedule_are_not_touched(self):
        ref, t = ref_and_template()
        s = sched()
        before = copy.deepcopy(s)
        sm.add_module(s, "530N00005A", ref, "f3", t, plan_row())
        del s["workflow_task_list"][-1]
        self.assertEqual(before, s)


class Deleting(unittest.TestCase):
    def test_the_module_and_all_assignment_rows_of_its_operations_go(self):
        s = sched()
        gone = sm.delete_module(s, s["workflow_task_list"][0])
        self.assertEqual({"phases": 3, "operations": 4, "rows": 3, "fixed_rows": 1}, gone)
        self.assertEqual(["e2"], [t["id"] for t in s["workflow_task_list"]])
        self.assertEqual([("w001", "e2p2o1"), ("w002", "misc_other_1")], [(a["worker"], a["operation_task"]) for a in s["assignment_list"]])

    def test_a_module_without_assignments(self):
        s = sched()
        s["assignment_list"] = []
        self.assertEqual({"phases": 3, "operations": 4, "rows": 0, "fixed_rows": 0}, sm.delete_module(s, s["workflow_task_list"][1]))
        self.assertEqual(["e1"], [t["id"] for t in s["workflow_task_list"]])

    def test_the_other_lists_stay_as_they_were(self):
        s = sched()
        misc, other = copy.deepcopy(s["misc_task_list"]), copy.deepcopy(s["workflow_task_list"][1])
        sm.delete_module(s, s["workflow_task_list"][0])
        self.assertEqual((misc, other), (s["misc_task_list"], s["workflow_task_list"][0]))
        self.assertEqual({"start_date": "2026/12/01", "end_date": "2027/03/31"}, s["plan_range"])


class Finding(unittest.TestCase):
    def test_by_the_seiban_in_the_name(self):
        s = sched()
        self.assertEqual(["e2"], [t["id"] for t in sm.module_tasks(s, "530N00002A")])
        self.assertEqual([], sm.module_tasks(s, "530N09999A"))

    def test_a_full_width_name_is_the_same_seiban(self):
        s = sched()
        s["workflow_task_list"][0]["name"] = "５３０Ｎ００００１Ａ"
        self.assertEqual(["e1"], [t["id"] for t in sm.module_tasks(s, "530N00001A")])


class AddRowChecks(unittest.TestCase):
    def setUp(self):
        self.t = template(2, TEMPLATE_WITH_P1)                     # processes 1, 2 and 4

    def test_a_complete_row_is_valid(self):
        self.assertEqual([], sm.add_row_problems(plan_row(template=2), self.t))

    def test_a_missing_date_of_a_process_of_the_template(self):
        row = plan_row(template=2, phases={1: ("2026/12/01", None), 2: (None, "2026/12/20"), 4: ("2026/12/22", "2027/01/10")})
        self.assertEqual(["process 1 has no end date", "process 2 has no start date"], sm.add_row_problems(row, self.t))

    def test_a_process_the_sheet_has_no_column_for(self):
        row = plan_row(template=2, phases={1: ("2026/12/01", "2026/12/10"), 2: ("2026/12/05", "2026/12/20")})
        self.assertEqual(["process 4 has no start date", "process 4 has no end date"], sm.add_row_problems(row, self.t))

    def test_the_end_before_the_start(self):
        row = plan_row(template=2, phases={1: ("2026/12/11", "2026/12/10"), 2: ("2026/12/05", "2026/12/20"), 4: ("2026/12/22", "2027/01/10")})
        self.assertEqual(["process 1: end date 2026/12/10 is before start date 2026/12/11"], sm.add_row_problems(row, self.t))

    def test_the_same_day_is_fine(self):
        row = plan_row(template=2, phases={1: ("2026/12/10", "2026/12/10"), 2: ("2026/12/05", "2026/12/20"), 4: ("2026/12/22", "2027/01/10")})
        self.assertEqual([], sm.add_row_problems(row, self.t))

    def test_dates_of_a_process_the_template_does_not_have_are_ignored(self):
        t = template(1, TEMPLATE_LIKE_WF_TOOL)                     # processes 2, 3, 4: no process 1
        row = plan_row(phases={1: (None, None), 2: ("2026/12/05", "2026/12/20"), 3: ("2026/12/15", "2026/12/25"), 4: ("2026/12/22", "2027/01/10")},
                       problems=["process 1 start: not a date ('abc')"])
        self.assertEqual([], sm.add_row_problems(row, t))

    def test_an_unreadable_date_of_a_process_the_template_has_stays_a_problem(self):
        row = plan_row(template=2, problems=["process 2 end: not a date ('abc')"])
        self.assertEqual(["process 2 end: not a date ('abc')"], sm.add_row_problems(row, self.t))

    def test_problems_of_the_row_itself_stay(self):
        row = plan_row(code=None, problems=["seiban 'x' is not a module code"])
        self.assertEqual(["seiban 'x' is not a module code"], sm.add_row_problems(row, template(1, TEMPLATE_LIKE_WF_TOOL)))

    def test_no_template_sheet(self):
        problems = sm.add_row_problems(plan_row(template=7), None)
        self.assertEqual(1, len(problems))
        self.assertIn("no sheet for template 7", problems[0])

    def test_a_template_with_problems(self):
        bad = template(2, TEMPLATE_WITH_P1)
        bad.problems.append("row 4: minimum workers is not a number")
        problems = sm.add_row_problems(plan_row(template=2), bad)
        self.assertEqual(["template 2 cannot be used: row 4: minimum workers is not a number"], problems)

    def test_a_template_without_operations(self):
        self.assertIn("it has no operations", sm.add_row_problems(plan_row(template=2), template(2, []))[0])

    def test_no_template_chosen_is_the_problem_of_the_row_only(self):
        row = plan_row(template=None, problems=["no process template chosen (工程選択 is empty)"])
        self.assertEqual(["no process template chosen (工程選択 is empty)"], sm.add_row_problems(row, None))


class DeleteRowChecks(unittest.TestCase):
    def test_only_the_seiban_matters(self):
        row = plan_row(kind="delete", problems=["no process template chosen (工程選択 is empty)", "process 1 start: x"])
        self.assertEqual([], sm.delete_row_problems(row))

    def test_a_row_without_a_module_code(self):
        row = plan_row(kind="delete", code=None, problems=["seiban 'abc' is not a module code"])
        self.assertEqual(["seiban 'abc' is not a module code"], sm.delete_row_problems(row))
        self.assertEqual(["seiban is not a module code"], sm.delete_row_problems(plan_row(kind="delete", code=None)))


if __name__ == "__main__":
    unittest.main()

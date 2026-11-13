"""Tests of env_workflows.py (match a template to a workflow, or create one).  Run from the tool folder:  python -m unittest discover -s tests -v"""
import unittest

from envs import TEMPLATE_LIKE_WF_TOOL, TEMPLATE_WITH_P1, make_env, template
import env_workflows as wfs


class Matching(unittest.TestCase):
    def test_a_template_with_the_same_operations_uses_the_existing_workflow(self):
        env = make_env()
        before = repr(env)
        ref = wfs.ensure_workflow(env, template(2, TEMPLATE_LIKE_WF_TOOL), [])
        self.assertEqual(("wf_tool", False, []), (ref.workflow_id, ref.created, ref.new_ops))
        self.assertEqual({"id": "p2", "name": "Hardware Setup"}, ref.phases[2])
        self.assertEqual({"id": "p3o1", "name": "QC"}, ref.ops[(3, 1)])
        self.assertEqual({"id": "p2o2", "name": "Elec"}, ref.ops[(2, 2)])
        self.assertEqual(before, repr(env))                                # nothing changed

    def test_other_minimum_or_maximum_workers_do_not_match(self):
        env = make_env()
        ops = [(2, 1, "Mech", 576, 2, 4), (2, 2, "Elec", 144, 1, 1), (3, 1, "QC", 64, 1, 2), (4, 1, "QC", 288, 1, 2)]
        report = []
        ref = wfs.ensure_workflow(env, template(2, ops), report)
        self.assertEqual(("wf1", True), (ref.workflow_id, ref.created))
        self.assertEqual(["wf_tool", "wf1"], [w["id"] for w in env["workflow_list"]])     # the old one stays as it was
        self.assertIn("workflow wf_tool has the same works but other minimum / maximum workers", report[0])

    def test_a_different_work_does_not_match(self):
        env = make_env()
        ops = [(2, 1, "Mech", 576, 1, 8), (2, 2, "Safety", 144, 1, 8), (3, 1, "QC", 64, 1, 8), (4, 1, "QC", 288, 1, 8)]
        self.assertTrue(wfs.ensure_workflow(env, template(2, ops), []).created)

    def test_work_names_are_compared_by_key(self):
        env = make_env()
        ops = [(2, 1, " mech ", 576, 1, 8), (2, 2, "ELEC", 144, 1, 8), (3, 1, "qc", 64, 1, 8), (4, 1, "QC", 288, 1, 8)]
        self.assertFalse(wfs.ensure_workflow(env, template(2, ops), []).created)

    def test_hours_and_names_are_not_compared(self):
        env = make_env()
        ops = [(2, 1, "Mech", 1, 1, 8), (2, 2, "Elec", 2, 1, 8), (3, 1, "QC", 3, 1, 8), (4, 1, "QC", 4, 1, 8)]
        self.assertEqual("wf_tool", wfs.ensure_workflow(env, template(2, ops), []).workflow_id)


class Creating(unittest.TestCase):
    def test_ids_names_and_fields_of_a_new_workflow(self):
        env = make_env()
        ref = wfs.ensure_workflow(env, template(1, TEMPLATE_WITH_P1), [])
        wf = env["workflow_list"][-1]
        self.assertEqual(("wf1", "wf1"), (wf["id"], wf["name"]))
        self.assertEqual([("wf1p1", "p1"), ("wf1p2", "p2"), ("wf1p4", "p4")], [(p["id"], p["name"]) for p in wf["phase_list"]])
        ops = [o for p in wf["phase_list"] for o in p["operation_list"]]
        self.assertEqual(["wf1p1o1", "wf1p1o2", "wf1p2o1", "wf1p4o1"], [o["id"] for o in ops])
        self.assertEqual({"id": "wf1p1o1", "name": "Mech", "work_hours": [10], "min_worker_num": 2, "max_worker_num": 4}, ops[0])
        self.assertEqual({"id": "wf1p4o1", "name": "QC", "work_hours": [10], "min_worker_num": 1, "max_worker_num": 2}, ops[3])
        self.assertEqual([1, 2, 4], sorted(ref.phases))                      # a template without process 3 has no process 3
        self.assertEqual({"id": "wf1p1o2", "name": "Elec"}, ref.ops[(1, 2)])
        self.assertEqual(["wf1p1o1", "wf1p1o2", "wf1p2o1", "wf1p4o1"], [o["id"] for o in ref.new_ops])

    def test_the_work_hours_list_is_not_shared_between_operations(self):
        env = make_env()
        wfs.ensure_workflow(env, template(1, TEMPLATE_WITH_P1), [])
        ops = [o for p in env["workflow_list"][-1]["phase_list"] for o in p["operation_list"]]
        ops[0]["work_hours"].append(5)
        self.assertEqual([10], ops[1]["work_hours"])

    def test_the_phase_number_of_the_new_ids_reads_back_correctly(self):
        env = make_env()
        ref = wfs.ensure_workflow(env, template(1, TEMPLATE_WITH_P1), [])
        self.assertEqual({1, 2, 4}, set(ref.phases))
        self.assertEqual({(1, 1), (1, 2), (2, 1), (4, 1)}, set(ref.ops))

    def test_numbers_continue_and_legacy_ids_are_ignored(self):
        env = make_env()
        self.assertEqual(1, wfs.next_workflow_number(env))                    # only 'wf_tool'
        env["workflow_list"].append({"id": "wf5", "name": "x", "phase_list": []})
        self.assertEqual(6, wfs.next_workflow_number(env))
        ref = wfs.ensure_workflow(env, template(1, TEMPLATE_WITH_P1), [])
        self.assertEqual("wf6", ref.workflow_id)

    def test_a_second_template_gets_the_next_number_and_the_same_template_again_finds_its_workflow(self):
        env = make_env()
        first = wfs.ensure_workflow(env, template(1, TEMPLATE_WITH_P1), [])
        other = [(2, 1, "Mech", 1, 1, 1)]
        second = wfs.ensure_workflow(env, template(2, other), [])
        again = wfs.ensure_workflow(env, template(1, TEMPLATE_WITH_P1), [])
        self.assertEqual(("wf1", "wf2", "wf1"), (first.workflow_id, second.workflow_id, again.workflow_id))
        self.assertEqual((True, True, False), (first.created, second.created, again.created))
        self.assertEqual(3, len(env["workflow_list"]))

    def test_report_lines(self):
        env = make_env()
        report = []
        wfs.ensure_workflow(env, template(2, TEMPLATE_LIKE_WF_TOOL), report)
        wfs.ensure_workflow(env, template(1, TEMPLATE_WITH_P1), report)
        self.assertEqual("INFO: template 2 uses the existing workflow wf_tool", report[0])
        self.assertTrue(report[1].startswith("INFO: template 1: new workflow wf1 created"))


if __name__ == "__main__":
    unittest.main()

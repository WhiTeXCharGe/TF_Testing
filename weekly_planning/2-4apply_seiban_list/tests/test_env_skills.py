"""Tests of env_skills.py (skills of the workers for new operations).  Run from the tool folder:  python -m unittest discover -s tests -v"""
import unittest

from envs import TEMPLATE_WITH_P1, make_env, template
import env_skills as sk
import env_workflows as wfs


def worker(env, wid):
    return next(w for w in env["worker_list"] if w["id"] == wid)


class Copy(unittest.TestCase):
    def new_workflow(self, ops):
        env = make_env()
        ref = wfs.ensure_workflow(env, template(1, ops), [])
        return env, ref

    def test_a_new_operation_gets_the_highest_level_of_the_same_work_name(self):
        env, ref = self.new_workflow([(1, 1, "Mech", 1, 1, 2), (2, 1, "QC", 1, 1, 2)])
        report = []
        sk.add_skills(env, ref.new_ops, report)
        w1, w2, w3 = (worker(env, w) for w in ("w001", "w002", "w003"))
        self.assertEqual(2, w1["skill_map"]["wf1p1o1"])                      # Mech: p2o1 = 2
        self.assertEqual(3, w1["skill_map"]["wf1p2o1"])                      # QC: max(p3o1 = 3, p4o1 = 1)
        self.assertEqual((1, 0), (w2["skill_map"]["wf1p1o1"], w2["skill_map"]["wf1p2o1"]))
        self.assertNotIn("wf1p1o1", w3["skill_map"])                          # a worker without that work gets nothing
        self.assertEqual({"other_op": 1, "personal_business_op": 1}, w3["skill_map"])

    def test_the_worker_type_comes_from_the_operation_with_the_highest_level(self):
        env, ref = self.new_workflow([(2, 1, "QC", 1, 1, 2)])
        sk.add_skills(env, ref.new_ops, [])
        self.assertEqual("spot", worker(env, "w001")["worker_type_by_operation"]["wf1p2o1"])      # from p3o1 (level 3), not p4o1 (level 1)
        self.assertNotIn("wf1p2o1", worker(env, "w002")["worker_type_by_operation"])             # the source has no type entry

    def test_a_level_of_zero_is_copied_and_does_not_count_as_able(self):
        env, ref = self.new_workflow([(1, 1, "Elec", 1, 1, 1)])
        report = []
        sk.add_skills(env, ref.new_ops, report)
        self.assertEqual(0, worker(env, "w001")["skill_map"]["wf1p1o1"])
        self.assertEqual(1, worker(env, "w002")["skill_map"]["wf1p1o1"])
        self.assertEqual(["INFO: operation wf1p1o1 (Elec): 1 workers can do it (highest level of the same work name)"], report)

    def test_a_work_name_that_no_existing_operation_has_is_a_warning(self):
        env, ref = self.new_workflow([(1, 1, "Safety", 1, 1, 1)])
        report = []
        sk.add_skills(env, ref.new_ops, report)
        self.assertEqual(1, len(report))
        self.assertTrue(report[0].startswith("WARN") and "no existing operation has this work name" in report[0])
        self.assertTrue(all("wf1p1o1" not in w["skill_map"] for w in env["worker_list"]))

    def test_nobody_able_is_a_warning(self):
        env, ref = self.new_workflow([(1, 1, "Mech", 1, 1, 1)])
        for w in env["worker_list"]:
            w["skill_map"]["p2o1"] = 0
        report = []
        sk.add_skills(env, ref.new_ops, report)
        self.assertTrue(report[0].startswith("WARN") and "no worker has a skill" in report[0])

    def test_operations_created_together_do_not_feed_each_other(self):
        env, ref = self.new_workflow([(1, 1, "Mech", 1, 1, 1), (2, 1, "Mech", 1, 1, 1)])
        sk.add_skills(env, ref.new_ops, [])
        self.assertEqual((2, 2), (worker(env, "w001")["skill_map"]["wf1p1o1"], worker(env, "w001")["skill_map"]["wf1p2o1"]))

    def test_a_second_workflow_copies_from_everything_that_exists_then(self):
        env, ref = self.new_workflow([(1, 1, "Mech", 1, 1, 1)])
        sk.add_skills(env, ref.new_ops, [])
        worker(env, "w002")["skill_map"]["wf1p1o1"] = 5                       # a later edit: the new highest Mech level of w002
        ref2 = wfs.ensure_workflow(env, template(2, [(3, 1, "Mech", 1, 1, 1)]), [])
        sk.add_skills(env, ref2.new_ops, [])
        self.assertEqual(5, worker(env, "w002")["skill_map"]["wf2p3o1"])

    def test_workers_without_a_skill_map_are_left_alone(self):
        env, ref = self.new_workflow([(1, 1, "Mech", 1, 1, 1)])
        env["worker_list"].append({"id": "w004", "name": "Worker 4"})
        sk.add_skills(env, ref.new_ops, [])
        self.assertEqual({"id": "w004", "name": "Worker 4"}, worker(env, "w004"))

    def test_nothing_to_do_without_new_operations(self):
        env = make_env()
        before = repr(env)
        report = []
        sk.add_skills(env, [], report)
        self.assertEqual((before, []), (repr(env), report))


class WholeStep(unittest.TestCase):
    def test_template_to_workflow_to_skills(self):
        env = make_env()
        report = []
        ref = wfs.ensure_workflow(env, template(1, TEMPLATE_WITH_P1), report)
        sk.add_skills(env, ref.new_ops, report)
        w1 = worker(env, "w001")["skill_map"]
        self.assertEqual((2, 0, 2, 3), (w1["wf1p1o1"], w1["wf1p1o2"], w1["wf1p2o1"], w1["wf1p4o1"]))     # Mech 2, Elec 0, Mech 2, QC max(3, 1)
        self.assertEqual("regular", worker(env, "w001")["worker_type_by_operation"]["wf1p1o1"])
        self.assertFalse(any(line.startswith("WARN") for line in report))
        self.assertEqual(4, sum(1 for line in report if line.startswith("INFO: operation")))               # one line per new operation


if __name__ == "__main__":
    unittest.main()

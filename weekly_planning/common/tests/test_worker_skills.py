"""Tests of common/worker_skills.py: role + level text -> skill per work name, status -> worker type.

Run from weekly_planning/:   python -m unittest discover -s common/tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # weekly_planning/ (the folder of common)
from common.worker_skills import STATUS_TO_TYPE, parse_worker_skills   # noqa: E402


class ParseWorkerSkills(unittest.TestCase):
    def test_roles_become_work_names(self):
        self.assertEqual(({"Mech": 2}, None), parse_worker_skills("M", "2"))
        self.assertEqual(({"Elec": 3}, None), parse_worker_skills("E", "3"))
        self.assertEqual(({"QC": 1}, None), parse_worker_skills("QC", "1"))

    def test_one_level_for_all_roles_or_one_level_per_role(self):
        self.assertEqual(({"Mech": 3, "QC": 3}, None), parse_worker_skills("M/QC", "3"))
        self.assertEqual(({"Mech": 2, "QC": 3}, None), parse_worker_skills("M/QC", "2/3"))

    def test_the_result_is_in_the_order_mech_elec_qc(self):
        skills, _ = parse_worker_skills("QC/E/M", "1/2/3")
        self.assertEqual(["Mech", "Elec", "QC"], list(skills))

    def test_a_lone_q_is_qc_and_other_text_is_case_and_width_insensitive(self):
        self.assertEqual(({"QC": 2}, None), parse_worker_skills("Q", "2"))
        self.assertEqual(({"Mech": 2, "Elec": 2}, None), parse_worker_skills("ｍ／ｅ", "２"))

    def test_a_zero_level_gives_no_skill_for_that_role(self):
        self.assertEqual(({"QC": 3}, None), parse_worker_skills("M/QC", "0/3"))

    def test_workers_without_role_or_level_are_excluded_with_a_reason(self):
        self.assertEqual(({}, "担当職種 blank"), parse_worker_skills("", "2"))
        self.assertEqual(({}, "スキルレベル blank"), parse_worker_skills("M", ""))
        self.assertEqual(({}, "スキルレベル 0"), parse_worker_skills("M", "0"))
        self.assertEqual("スキルレベル unreadable ('x')", parse_worker_skills("M", "x")[1])

    def test_an_unknown_role_gives_no_skill(self):
        self.assertEqual(({}, None), parse_worker_skills("X", "2"))


class StatusToType(unittest.TestCase):
    def test_only_primary_is_listed_as_regular(self):
        self.assertEqual({"PRIMARY": "regular"}, STATUS_TO_TYPE)     # every other status is spot (the caller's default)


if __name__ == "__main__":
    unittest.main()

"""Tests of env_masters.py (customers, fabs, regions).  Run from the tool folder:  python -m unittest discover -s tests -v"""
import unittest

from envs import make_env
import env_masters as m


def site(env, customer, fab, region, site_region=None):
    report = []
    fab_id = m.resolve_site(env, customer, fab, region, site_region, report)
    return fab_id, report


class Existing(unittest.TestCase):
    def test_known_customer_fab_and_region_change_nothing(self):
        env = make_env()
        before = repr(env)
        fab_id, report = site(env, "Cust A", "Fab A", "Region X")
        self.assertEqual(("f1", []), (fab_id, report))
        self.assertEqual(before, repr(env))

    def test_names_match_by_key_not_by_exact_text(self):
        env = make_env()
        fab_id, report = site(env, " cust a ", "FAB　A", "region x")
        self.assertEqual(("f1", []), (fab_id, report))
        self.assertEqual(3, len(env["fab_list"]))

    def test_the_same_call_twice_creates_nothing_the_second_time(self):
        env = make_env()
        first, _ = site(env, "Cust C", "Fab C", "Region Z")
        second, report = site(env, "Cust C", "Fab C", "Region Z")
        self.assertEqual(first, second)
        self.assertEqual([], report)


class NewEntries(unittest.TestCase):
    def test_new_customer_and_fab_in_a_known_region(self):
        env = make_env()
        fab_id, report = site(env, "Cust C", "Fab C", "Region Y")
        self.assertEqual("f3", fab_id)
        self.assertEqual({"id": "c3", "name": "Cust C", "unavailable_dates": []}, env["customer_company_list"][-1])
        self.assertEqual({"id": "f3", "name": "Fab C", "region": "r2", "customer_company": "c3", "unavailable_dates": []}, env["fab_list"][-1])
        self.assertEqual(3, len(env["region_list"]))
        self.assertEqual(["INFO: customer added: c3 'Cust C'", "INFO: fab added: f3 'Fab C' (customer c3, region r2)"], report)

    def test_the_same_fab_name_under_another_customer_is_another_fab(self):
        env = make_env()
        fab_id, _ = site(env, "Cust B", "Fab A", "Region X")
        self.assertEqual("f3", fab_id)
        self.assertEqual("c2", env["fab_list"][-1]["customer_company"])
        self.assertEqual("f1", site(env, "Cust A", "Fab A", "Region X")[0])        # the first one is still its own fab

    def test_new_region_gets_the_decoder_defaults_and_transit_rows_both_ways(self):
        env = make_env()
        _, report = site(env, "Cust A", "Fab A2", "Region Z")
        self.assertEqual({"id": "r3", "name": "Region Z", "max_stay_on": 10000, "max_annual_stay": 10000, "stay_off_interval": 3,
                          "unavailable_dates": []}, env["region_list"][-1])
        transit = env["transite_day_map"]
        self.assertEqual(6 + 6, len(transit))                                    # 3 regions * 2 directions added
        for other in ("r_other", "r1", "r2"):
            self.assertIn({"from": "r3", "to": other, "days": 1}, transit)
            self.assertIn({"from": other, "to": "r3", "days": 1}, transit)
        self.assertTrue(any("region added: r3" in line for line in report))

    def test_ids_continue_after_the_highest_number_and_ignore_other_ids(self):
        self.assertEqual("r1", m.next_id([{"id": "r_other"}], "r"))
        self.assertEqual("c8", m.next_id([{"id": "c_other"}, {"id": "c2"}, {"id": "c7"}, {"id": "wc9"}], "c"))
        self.assertEqual("f1", m.next_id([], "f"))


class WithoutNames(unittest.TestCase):
    def test_customer_without_fab_uses_a_fab_named_after_the_customer(self):
        env = make_env()
        fab_id, _ = site(env, "Cust C", "", "Region X")
        self.assertEqual({"id": fab_id, "name": "Cust C", "region": "r1", "customer_company": "c3", "unavailable_dates": []}, env["fab_list"][-1])
        self.assertEqual(fab_id, site(env, "Cust C", "", "Region X")[0])          # found again by the same rule

    def test_no_customer_and_no_fab_is_f_other(self):
        env = make_env()
        self.assertEqual(("f_other", []), site(env, "", "", "Region X"))
        self.assertEqual(3, len(env["fab_list"]))

    def test_no_customer_but_a_fab_goes_under_c_other(self):
        env = make_env()
        fab_id, _ = site(env, "", "Fab Z", "Region X")
        self.assertEqual("c_other", env["fab_list"][-1]["customer_company"])
        self.assertEqual("Fab Z", env["fab_list"][-1]["name"])

    def test_the_other_entries_are_added_when_the_env_has_none(self):
        env = {"region_list": [], "customer_company_list": [], "fab_list": [], "transite_day_map": []}
        fab_id, _ = site(env, "", "", "")
        self.assertEqual("f_other", fab_id)
        self.assertEqual(["r_other"], [r["id"] for r in env["region_list"]])
        self.assertEqual(["c_other"], [c["id"] for c in env["customer_company_list"]])


class RegionRules(unittest.TestCase):
    def test_empty_region_is_filled_from_the_site_table(self):
        env = make_env()
        fab_id, report = site(env, "Cust C", "Fab C", "", site_region="Region Y")
        self.assertEqual("r2", env["fab_list"][-1]["region"])
        self.assertFalse(any(line.startswith("WARN") for line in report))

    def test_the_row_wins_over_the_site_table_and_the_difference_is_a_warning(self):
        env = make_env()
        _, report = site(env, "Cust C", "Fab C", "Region X", site_region="Region Y")
        self.assertEqual("r1", env["fab_list"][-1]["region"])
        self.assertTrue(any(line.startswith("WARN") and "site table" in line for line in report))

    def test_unknown_region_of_a_new_fab_is_r_other_with_a_warning(self):
        env = make_env()
        _, report = site(env, "Cust C", "Fab C", "")
        self.assertEqual("r_other", env["fab_list"][-1]["region"])
        self.assertTrue(any(line.startswith("WARN") and "r_other" in line for line in report))

    def test_unknown_region_of_an_existing_fab_leaves_it_alone(self):
        env = make_env()
        fab_id, report = site(env, "Cust A", "Fab A", "")
        self.assertEqual(("f1", []), (fab_id, report))
        self.assertEqual("r1", env["fab_list"][1]["region"])

    def test_existing_fab_with_another_region_is_updated_and_reported(self):
        env = make_env()
        fab_id, report = site(env, "Cust A", "Fab A", "Region Y")
        self.assertEqual(("f1", "r2"), (fab_id, env["fab_list"][1]["region"]))
        self.assertEqual(["INFO: fab f1 'Fab A' (Cust A): region changed 'Region X' -> 'Region Y'"], report)


if __name__ == "__main__":
    unittest.main()

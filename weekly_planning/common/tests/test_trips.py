"""Tests of trips.py: reading FI / FO marks, pairing them and repairing marks that have no partner (no Excel files needed).

Run from the common folder:   python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # weekly_planning/ (the folder of common)
from common.trips import flatten_trips, pair_fifo, pair_fifo_repaired, read_mark   # noqa: E402


def days(first, last):
    from datetime import date, timedelta
    a, b = date(*map(int, first.split("/"))), date(*map(int, last.split("/")))
    return [(a + timedelta(n)).strftime("%Y/%m/%d") for n in range((b - a).days + 1)]


class Pairing(unittest.TestCase):
    def test_pair_fifo_and_orphans(self):
        pairs, warns = pair_fifo({"2026/10/01": "FI", "2026/10/05": "FO", "2026/10/09": "FI", "2026/10/12": "FI",
                                     "2026/10/15": "FO", "2026/10/20": "FO", "2026/10/25": "FI"})
        self.assertEqual([("2026/10/01", "2026/10/05"), ("2026/10/12", "2026/10/15")], pairs)
        self.assertEqual(3, len(warns))
        self.assertTrue(any("10/09" in w and "next FI" in w for w in warns))
        self.assertTrue(any("FO 2026/10/20" in w for w in warns))
        self.assertTrue(any("FI 2026/10/25" in w for w in warns))



class FiFoPatterns(unittest.TestCase):
    WIN = ("2026/01/01", "2027/03/31")

    def test_read_mark_accepts_typos_case_and_notes_but_not_other_words(self):
        for text, mark in [("FI", "FI"), ("Fi", "FI"), ("ＦＩ", "FI"), (" fi ", "FI"), ("F1", "FI"), ("Fl", "FI"), ("FI(仮)", "FI"),
                           ("FO", "FO"), ("Fo", "FO"), ("F0", "FO"), ("FO（仮）", "FO")]:
            self.assertEqual(mark, read_mark(text), text)
        for text in ["移動", "FabC", "FC", "", None, 5, "FI 530N00009A"]:
            self.assertIsNone(read_mark(text), text)

    def test_complete_pair_is_unchanged(self):
        trips, notes = pair_fifo_repaired({"2026/03/01": "FI", "2026/03/20": "FO"}, [], self.WIN)
        self.assertEqual([("2026/03/01", "2026/03/20")], trips)
        self.assertEqual([], notes)

    def test_fi_without_fo_ends_on_the_last_busy_day_of_the_run(self):
        busy = days("2026/10/20", "2026/12/18") + days("2027/02/01", "2027/02/03")     # a long gap after 12/18
        trips, notes = pair_fifo_repaired({"2026/10/19": "FI"}, busy, self.WIN)
        self.assertEqual([("2026/10/19", "2026/12/18")], trips)
        self.assertTrue(notes[0].startswith("INFERRED end 2026/12/18"))

    def test_fi_without_fo_and_nothing_busy_is_a_one_day_trip(self):
        trips, _ = pair_fifo_repaired({"2026/10/19": "FI"}, [], self.WIN)
        self.assertEqual([("2026/10/19", "2026/10/19")], trips)

    def test_fi_followed_by_another_fi_is_closed_before_the_next_fi(self):
        busy = days("2026/10/21", "2026/11/18") + days("2026/11/26", "2026/12/23")
        marks = {"2026/10/20": "FI", "2026/11/25": "FI", "2026/12/30": "FI"}
        trips, _ = pair_fifo_repaired(marks, busy, self.WIN, max_gap=14)
        self.assertEqual([("2026/10/20", "2026/11/18"), ("2026/11/25", "2026/12/23"), ("2026/12/30", "2026/12/30")], trips)

    def test_fo_without_fi_starts_on_the_first_busy_day_of_the_run(self):
        busy = days("2026/05/02", "2026/06/27")
        trips, notes = pair_fifo_repaired({"2026/06/28": "FO"}, busy, self.WIN)
        self.assertEqual([("2026/05/02", "2026/06/28")], trips)
        self.assertTrue(notes[0].startswith("INFERRED start 2026/05/02"))

    def test_first_fo_of_the_sheet_means_the_trip_began_before_the_sheet(self):
        trips, _ = pair_fifo_repaired({"2026/01/07": "FO"}, days("2026/01/03", "2026/01/06"), self.WIN)
        self.assertEqual([("2026/01/01", "2026/01/07")], trips)

    def test_fo_after_fo_starts_on_the_date_of_the_previous_fo(self):
        busy = days("2026/10/04", "2027/01/27")
        marks = {"2026/10/03": "FO", "2027/01/28": "FO"}
        trips, _ = pair_fifo_repaired(marks, busy, self.WIN)
        # the first FO has nothing busy before it: a one-day trip; the second trip starts on the same date as the first FO
        self.assertEqual([("2026/10/03", "2026/10/03"), ("2026/10/03", "2027/01/28")], trips)

    def test_fi_taska_fo_taskb_fo_chain_shares_the_fo_date(self):
        """FI | TaskA ... | FO | TaskB ... | FO  ->  two trips, the second starts on the date of the first FO (not on the first TaskB day)."""
        busy = days("2026/10/21", "2026/10/29") + days("2026/10/31", "2026/11/09")
        marks = {"2026/10/20": "FI", "2026/10/30": "FO", "2026/11/10": "FO"}
        trips, notes = pair_fifo_repaired(marks, busy, self.WIN)
        self.assertEqual([("2026/10/20", "2026/10/30"), ("2026/10/30", "2026/11/10")], trips)
        self.assertTrue(any("same date as the previous FO" in n for n in notes))

    def test_fo_chain_covers_a_weekend_between_the_fo_and_the_next_work(self):
        busy = days("2026/11/02", "2026/11/09")                      # FO on a Friday, work starts on the Monday (3 days later)
        trips, _ = pair_fifo_repaired({"2026/10/30": "FO", "2026/11/10": "FO"}, busy, self.WIN)
        self.assertEqual(("2026/10/30", "2026/11/10"), trips[-1])

    def test_fo_chain_does_not_apply_when_the_work_starts_long_after_the_previous_fo(self):
        busy = days("2026/11/06", "2026/11/09")                      # 6 days after the first FO
        trips, _ = pair_fifo_repaired({"2026/10/30": "FO", "2026/11/10": "FO"}, busy, self.WIN)
        self.assertEqual(("2026/11/06", "2026/11/10"), trips[-1])

    def test_fi_taska_fi_taskb_fo_chain_shares_the_fi_date(self):
        busy = days("2026/10/21", "2026/10/29") + days("2026/11/01", "2026/11/09")
        marks = {"2026/10/20": "FI", "2026/10/30": "FI", "2026/11/10": "FO"}
        trips, _ = pair_fifo_repaired(marks, busy, self.WIN)
        self.assertEqual([("2026/10/20", "2026/10/30"), ("2026/10/30", "2026/11/10")], trips)

    def test_a_one_day_trip_between_two_fo_is_not_taken_for_a_duplicate_mark(self):
        trips, _ = pair_fifo_repaired({"2026/10/30": "FO", "2026/11/01": "FO"}, ["2026/10/31"], self.WIN)
        self.assertEqual(2, len(trips))

    def test_double_marks_on_neighbouring_days_are_one_mark(self):
        trips, notes = pair_fifo_repaired(
            {"2026/05/01": "FI", "2026/05/02": "FI", "2026/05/22": "FO", "2026/05/23": "FO"}, [], self.WIN)
        self.assertEqual([("2026/05/01", "2026/05/23")], trips)
        self.assertEqual(2, sum(1 for n in notes if n.startswith("duplicate")))

    def test_a_gap_larger_than_max_gap_stops_the_run(self):
        busy = days("2026/10/20", "2026/10/25") + days("2026/12/01", "2026/12/05")
        trips, _ = pair_fifo_repaired({"2026/10/19": "FI"}, busy, self.WIN, max_gap=14)
        self.assertEqual([("2026/10/19", "2026/10/25")], trips)



class FlattenTrips(unittest.TestCase):
    def test_nested_and_flat_forms_give_the_same_entries(self):
        nested = [{"worker": "w1", "trips": [{"start_date": "2026/10/01", "end_date": "2026/10/05"}]}]
        flat = [{"worker": "w1", "start_date": "2026/10/01", "end_date": "2026/10/05"}]
        self.assertEqual([{"worker": "w1", "trip": nested[0]["trips"][0]}], flatten_trips(nested))
        self.assertEqual([{"worker": "w1", "trip": flat[0]}], flatten_trips(flat))
        self.assertEqual([], flatten_trips(None))
        self.assertEqual([], flatten_trips([{"trips": []}, "x", {"worker": "w2"}]))


if __name__ == "__main__":
    unittest.main()

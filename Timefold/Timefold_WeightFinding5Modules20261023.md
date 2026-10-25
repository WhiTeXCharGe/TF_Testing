# 5-module sample: why Timefold stays infeasible, and what the weights do

Date: 2026-10-23. Input: `web/Timefold/src/main/resource/{EnvConfig,Schedule}.yaml` (5 modules, 40 workers, plan 2025/09/01 to 2025/11/03, `assignment_list: []`).
Runs: Docker, 2 to 5 minutes, stage 2 off, `stage1_unimproved_seconds` large, fixed seeds. Code: branch `feature/timefold-refactor` (8ae01b5).

## 1. Is the input wrong?

No error found. 6,400 hours of work against 20 to 26 skilled workers per operation (for example `p2o2`: 1,000 h needed, about 7,360 h
available at 8 h/day), every phase window is long enough for the critical path with a full crew (3 workers, 12 h/day: about 21 working days
per module, windows are 25 to 40 days). A first run with `assignment_list: []` is valid.

## 2. What the solver does (2.7.1 version, seed 42, 2 minutes)

* Hard score `-98` after about 4 seconds, then **no hard improvement for the remaining 110 seconds** (only soft changes).
* Remaining hard violations: `phase-order` -84 (17 matches) and `block-end-within-window` -14 (7 matches). Everything about workers
  (`block-no-underfill`, `emp-one-task-per-day`, `seat-daily-cap-12h`, transit gap, overtime) is 0.
* Many blocks end up with **one worker for 20+ days** (for example `e1p2o1`: 1 worker, 10 h, 24 days) although `min_worker_num` is 2 (only the soft
  `block-recommend-headcount-range` asks for 2). Those long blocks push the next phase past its window.

## 3. Why it cannot get out (cause)

A block has three linked decisions: how many workers sit on it, hours per day, and number of days. Production must match the required hours:

* `block-no-underfill` (hard): **10 per missing hour**.
* `block-overfill-at-most-one-day` (hard): 1 per extra hour beyond one day of production.
* `phase-order` and `block-end-within-window` (hard): **1 per day** of violation.

To fix a late block the solver must shorten it **and** add a worker. Each half is refused alone:
shortening a 1-worker, 10 h block by one day costs 100 hard (10 hours x 10) and saves 1; adding a worker without shortening creates a large
overfill. So the plan is locked in a local optimum. This is your underfill guess, made precise: the problem is the ratio between the underfill
weight and the time-violation weights.

## 4. Experiment (branch `exp/weights`, local only, weights read from environment variables)

| Run | Underfill per hour | Phase-order and window per day | Result |
|---|---|---|---|
| now (seed 42) | 10 | 1 | -98 hard after 2 min, stuck |
| A | 10 | 100 | 0 hard after 26 s |
| B | 1 | 1 | -112 hard after 2 min, stuck (lowering underfill alone does not help) |
| C | 1 | 10 | 0 hard after 11 s |
| C, seeds 1, 2, 3 | 1 | 10 | 0 hard after 30 s, 74 s, 36 s |
| now, seed 7, 5 min | 10 | 1 | -170 hard, stuck |
| old TFDocker 1.27, 5 min | 10 | 1 | -8 hard, stuck (never reached 0 either) |

Reading: the ratio underfill : time weight is what matters. With 1 day of lateness weighing about one working day of hours (10) per hour of underfill,
the solver finds hard 0 within about a minute on every seed tried. The old 1.27 version has the same weights and also does not reach 0 (it just
gets stuck closer, -8); it is not "easy" for the old version either.

## 5. Requirement: even an impossible plan must be filled

Requirement from the user: if a plan cannot be made feasible, Timefold must still put people on every block (breaking other hard rules if it has
to) instead of leaving blocks empty; but it must not lock itself in because underfill is too heavy. This rules out setting C
(underfill 1): on the data_generator sample (241 blocks, over-constrained, 3 minutes, seed 42) it left **113 blocks unstaffed**.

data_generator sample, 3 minutes, seed 42 (hard scores are not comparable between settings, the counts are):

| | now (10 / 1) | C (1 / 10) | **A (10 / 100)** |
|---|---|---|---|
| blocks left (partly) unstaffed | 0 | 113 (-11,032 h) | **3** (68 h) |
| phase-order and window matches | 94 + 92 | 11 + 1 | **2 + 0** |
| worker clashes and cap (one-task-per-day, 12 h cap) matches | 550 + 550 | 180 + 180 | 819 + 819 |

Setting A keeps underfill 10 times heavier than the worker rules, so it staffs everything and, when workers are short, double-books them
(the plan is impossible, so something has to give, and it is the worker rules, as requested). Setting C gave up on filling instead.

5-module sample with A (stage 1 stops at hard 0): seeds 2, 3, 4, 5 reach 0 hard after 30 s, 21 s, 25 s, 13 s; seed 1 ends at -2 hard after 3 minutes
(the old weights: -98 and -170 after 2 and 5 minutes).

## 6. Decision (taken 2026-10-23)

Option 1 with setting A: **`UNDERFILL_MULT` stays 10 per missing hour, new `TIME_VIOLATION_MULT = 100` per day** for `phase-order` and
`block-end-within-window` (constants in `constraints/BlockConstraints`). This changes the score values, so the characterisation goldens were regenerated
on purpose (the opt-in real-sample golden was not, it needs a run of the heavy file). Not done: a compound move (shorten the block and add a worker in one step),
which would keep the old weights; worth trying if seed 1 type stalls matter.

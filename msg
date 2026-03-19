Absolutely. Here is the **English version** in the same structure, kept **short and clear** so you can save it in your notes.

---

# Target Requirements

Overview of the requirements that the schedule should satisfy.

## Hard Constraints

| Constraint Name                | Description                                                                     | Notes                                        |
| ------------------------------ | ------------------------------------------------------------------------------- | -------------------------------------------- |
| Phase Order                    | A successor phase cannot start before the predecessor phase has finished        | Active                                       |
| Working Hours                  | Daily working hours for each task must be selected only from the allowed values | Active                                       |
| Start / End Window             | Each task must be completed within its allowed scheduling window                | Active                                       |
| Required Workload              | Each task block must satisfy at least the required workload                     | Active                                       |
| Overfill Limit                 | Excess workload is allowed only within a limited range                          | Active                                       |
| No Multiple Tasks per Day      | One worker cannot be assigned to multiple tasks on the same day                 | Active                                       |
| Daily Working Hour Limit       | A worker’s total working hours per day must not exceed the upper limit          | Active                                       |
| Transit-Day Constraint         | Required transit days must be secured when moving between different regions     | Active                                       |
| Responsible Manager Constraint | Seats requiring a manager can only be assigned to manager workers               | Controlled by ValueRangeProvider             |
| Skill Map                      | Only workers with the required skill are included in the candidate set          | Handled by candidate filtering               |
| Unavailable Date Constraint    | Unavailable dates are excluded from valid working-day calculation               | No direct hard score currently               |
| Min / Max Headcount            | Each task keeps its headcount conditions in the planning model                  | Min reflected indirectly, max via seat count |
| No Multiple Factories per Day  | One worker cannot work in multiple factories on the same day                    | Implemented, currently inactive              |
| Stay Duration Constraint       | Consecutive or annual stay duration in a region must not exceed limits          | Implemented, currently inactive              |
| Overtime Limit                 | Monthly and annual overtime must not exceed limits                              | Implemented, currently inactive              |

## Soft Constraints

| Constraint Name                    | Description                                                          | Notes                            |
| ---------------------------------- | -------------------------------------------------------------------- | -------------------------------- |
| Earlier Start                      | Prefer starting tasks as early as possible                           | Active                           |
| Smaller Daily Hours                | Prefer smaller daily work-hour values when possible                  | Active                           |
| Recommended Headcount Range        | Prefer staffing levels close to the recommended range                | Active                           |
| Suppress Over-Recommended Staffing | Lower priority for assignments beyond the recommended range          | Active                           |
| Workload Balancing                 | Reduce imbalance in total assigned hours across workers              | Active                           |
| Same Company Pairing               | Prefer teams composed of workers from the same company               | Active                           |
| Average Skill Adjustment           | Keep the average block skill close to the target skill level         | Active                           |
| Skill Diversity                    | Avoid overly uniform skill composition inside the same block         | Active                           |
| Regional Suitability               | Consider worker suitability for specific regions                     | Reflected in `softFabPreference` |
| Customer / Company Suitability     | Consider worker suitability for specific customers / companies       | Reflected in `softFabPreference` |
| Regional Continuity                | Prefer assignments that keep workers in the same region continuously | Implemented, currently inactive  |

---

# Modeling

## Parameters

Representative values obtained from `EnvConfig.yaml` / `Schedule.yaml` and used in the model and constraint calculations.

| Parameter                             | Description                                      | Main Source                       |
| ------------------------------------- | ------------------------------------------------ | --------------------------------- |
| plan range                            | Planning start date and end date                 | Schedule / `plan_range`           |
| day id                                | Integer ID for each date in the planning horizon | Generated internally              |
| worker id / name                      | Worker identification information                | EnvConfig `worker_list`           |
| skill map                             | Skill values of each worker for each operation   | EnvConfig `worker_list`           |
| manager flag                          | Whether the worker is a manager                  | EnvConfig `worker_list`           |
| worker company                        | Worker’s company                                 | EnvConfig `worker_list`           |
| region preference                     | Worker suitability / preference for regions      | EnvConfig `worker_list`           |
| customer preference                   | Worker suitability / preference for customers    | EnvConfig `worker_list`           |
| module / workflow / factory           | Task target information                          | Schedule                          |
| phase id / phase num / operation id   | Phase and operation identifiers                  | Schedule                          |
| task window start / end               | Allowed scheduling window                        | Schedule                          |
| allowed hours                         | Allowed daily work-hour options                  | Schedule                          |
| workload days / required hours        | Required amount of work                          | Schedule, internally converted    |
| min heads / max heads                 | Headcount conditions                             | Schedule                          |
| recommend min / max heads             | Recommended headcount range                      | Schedule                          |
| fab → region                          | Region of each factory                           | EnvConfig `fab_list`              |
| fab → customer                        | Customer associated with each factory            | EnvConfig `fab_list`              |
| fab unavailable dates                 | Factory unavailable dates                        | EnvConfig `fab_list`              |
| region unavailable dates              | Region unavailable dates                         | EnvConfig `region_list`           |
| customer unavailable dates            | Customer unavailable dates                       | EnvConfig `customer_company_list` |
| worker company unavailable dates      | Worker-company unavailable dates                 | EnvConfig `worker_company_list`   |
| worker unavailable dates              | Personal unavailable dates                       | EnvConfig `worker_list`           |
| transit days                          | Required transit days between regions            | EnvConfig `transite_day_map`      |
| region max stay / annual max stay     | Maximum stay duration in a region                | EnvConfig `region_list`           |
| monthly / annual overtime limit       | Monthly / annual overtime upper bounds           | EnvConfig `worker_company_list`   |
| fixed hours by emp-day                | Historical fixed working hours                   | Schedule assignment history       |
| fixed task count by emp-day           | Historical fixed task count                      | Schedule assignment history       |
| fixed factories by emp-day            | Historical fixed factory assignments             | Schedule assignment history       |
| historical last region / last factory | Last fixed historical region / factory           | Schedule history                  |

## Optimization Variables

| Element                  | Type              | Description                                                                  |
| ------------------------ | ----------------- | ---------------------------------------------------------------------------- |
| DaySlot                  | Problem Fact      | Basic unit representing one day in the planning horizon                      |
| EmployeeFact             | Problem Fact      | Fixed worker information and historical assignment data                      |
| TaskWindow               | Derived Input     | Original task frame holding window, headcount, allowed hours, workload, etc. |
| BlockDecision            | Planning Entity   | Planning decision unit for each task block                                   |
| `BlockDecision.startDay` | Planning Variable | Start day of the block                                                       |
| `BlockDecision.days`     | Planning Variable | Duration of the block                                                        |
| `BlockDecision.hours`    | Planning Variable | Daily working hours of the block                                             |
| CrewSeat                 | Planning Entity   | Worker-assignment seat under a block                                         |
| `CrewSeat.employee`      | Planning Variable | Worker assigned to the seat                                                  |
| SinglePassPlan           | Planning Solution | Full solution holding facts, entities, and score                             |

## Model Structure

| Relation      | Meaning               |
| ------------- | --------------------- |
| `\mathcal{D}` | Set of DaySlots       |
| `\mathcal{E}` | Set of EmployeeFacts  |
| `\mathcal{T}` | Set of TaskWindows    |
| `\mathcal{B}` | Set of BlockDecisions |
| `\mathcal{S}` | Set of CrewSeats      |

[
\forall t \in \mathcal{T},\ \exists, b \in \mathcal{B} \text{ such that } b \text{ is derived from } t
]

[
\forall b \in \mathcal{B},\ \exists, S_b \subseteq \mathcal{S}
]

[
|S_b| = \text{seat count of } b
]

[
\forall s \in S_b,\ \mathrm{employee}(s) \in \mathcal{E} \cup {\text{UNASSIGNED}}
]

[
\text{TaskWindow} \rightarrow \text{BlockDecision} \rightarrow \text{CrewSeat}
]

In other words:

* **TaskWindow** defines the original task frame,
* **BlockDecision** defines the actual scheduling decision for that task,
* **CrewSeat** defines who is assigned under that block.

---

# Constraint Formulation

## ValueRange / Candidate Filtering

### Skill Map

* Applied technique: `ValueRangeProvider` / candidate filtering
* Only workers satisfying the required skill are included in the candidate set of a `CrewSeat`
* Detail:

[
\mathcal{E}_s = {, e \in \mathcal{E} \mid \mathrm{skill}(e,\mathrm{op}(s)) > 0 ,}
]

[
\mathrm{employee}(s) \in \mathcal{E}_s
]

In the current code, this is handled by prebuilding the eligible candidate set rather than assigning a direct hard penalty.

### Responsible Manager Constraint

* Applied technique: `ValueRangeProvider`
* A manager-required seat can only be assigned to manager workers
* Detail:

[
\mathcal{E}^{mgr}_s = {, e \in \mathcal{E}_s \mid \mathrm{isManager}(e)=1 ,}
]

[
\mathrm{needManager}(s)=1 \Rightarrow \mathrm{employee}(s)\in \mathcal{E}^{mgr}_s
]

For non-manager seats, `UNASSIGNED` may be included as a candidate, but not for manager-required seats.

### Fixed Seat Constraint

* Applied technique: `ValueRangeProvider` / PlanningPin
* A fixed seat uses only the fixed worker, or only the fallback fixed value if needed
* Detail:

[
\mathrm{pinnedFixed}(s)=1 \Rightarrow \mathcal{E}_s = {\mathrm{pinnedWorker}(s)}
]

A fixed seat is excluded from solver changes by `isPinned()`.

### Start-Day Candidate Range

* Applied technique: `ValueRangeFactory`
* `BlockDecision.startDay` is selected only from the allowed window range
* Detail:

[
\mathrm{startDay}(b) \in [\mathrm{windowStart}(b),\ \mathrm{windowEnd}(b)]
]

For phase 1, the current code effectively restricts the candidate to `windowStart`.

### Duration Candidate Range

* Applied technique: `ValueRangeFactory`
* `BlockDecision.days` is selected from 1 day up to the window length
* Detail:

[
\mathrm{days}(b) \in {1,2,\dots,\mathrm{windowEnd}(b)-\mathrm{windowStart}(b)+1}
]

### Daily-Hour Candidate Range

* Applied technique: `ValueRangeProvider`
* `BlockDecision.hours` is selected from the discrete allowed-hour set
* Detail:

[
\mathrm{hours}(b) \in H_b
]

[
H_b = \mathrm{allowedHours}(b)
]

---

## Hard Constraints

### Phase Order

* Applied technique: HardScore
* In the same module, phase (p+1) cannot start before phase (p) ends
* Detail:

[
\mathrm{end}(b)=\mathrm{startDay}(b)+\mathrm{days}(b)-1
]

[
\forall (b_p,b_{p+1}),\
\mathrm{phaseNum}(b_{p+1})=\mathrm{phaseNum}(b_p)+1
]

[
\mathrm{HardScore}
==================

-\max\left(0,\ \mathrm{end}(b_p)-\mathrm{startDay}(b_{p+1})+1\right)
]

### Start / End Window

* Applied technique: HardScore
* The block end date must not exceed the allowed scheduling window
* Detail:

[
\mathrm{HardScore}
==================

-\max\left(0,\ \mathrm{startDay}(b)+\mathrm{days}(b)-1-\mathrm{windowEnd}(b)\right)
]

### Working Hours

* Applied technique: HardScore
* The selected daily working hours must belong to the allowed set
* Detail:

[
\mathrm{hours}(b)\notin H_b \Rightarrow \mathrm{HardScore}=-1
]

### Required Workload

* Applied technique: HardScore
* The effective production of the block must not fall below the required workload
* Detail:

[
P(b)=\sum_{s\in S_b,\ \mathrm{employee}(s)\neq \varnothing}\ \sum_{d\in W(b,s)} \mathrm{hours}(b)
]

Here, (W(b,s)) is the set of effective working days inside the block range after filtering by factory, region, customer, and worker availability calendars.

[
\mathrm{HardScore}
==================

-10\cdot \max(0,\ \mathrm{requiredHours}(b)-P(b))
]

### Overfill Limit

* Applied technique: HardScore
* Overfill is allowed, but only up to roughly one extra workday per staffed flexible seat
* Detail:

[
P_f(b)=\text{flexible-seat production only}
]

[
n_f(b)=|{s\in S_b \mid s \text{ is flexible and staffed}}|
]

[
\mathrm{HardScore}
==================

-\max\left(0,\ P_f(b)-\mathrm{requiredHours}(b)-n_f(b)\cdot \mathrm{hours}(b)\right)
]

### No Multiple Tasks per Day

* Applied technique: HardScore
* For a given worker and day, the task count must not exceed 1
* Detail:

[
N(e,d)=N^{dyn}(e,d)+N^{fixed}(e,d)
]

[
\mathrm{HardScore}
==================

-\max(0,\ N(e,d)-1)
]

Historical fixed task counts are included.

### Daily Working Hour Limit

* Applied technique: HardScore
* For a given worker and day, total working hours must not exceed `DAILY_CAP = 12`
* Detail:

[
H(e,d)=H^{dyn}(e,d)+H^{fixed}(e,d)
]

[
\mathrm{HardScore}
==================

-\max(0,\ H(e,d)-12)
]

Historical fixed working hours are included.

### Transit-Day Constraint

* Applied technique: HardScore
* When a worker moves between different regions, the required number of transit days must be respected
* Detail:

[
r(d)=\mathrm{regionOfFab}(\mathrm{fab}(d))
]

[
g(r_i,r_j)=\mathrm{transitDays}(r_i,r_j)
]

For adjacent working dates in different regions, a violation occurs if the date gap is smaller than the required transit days.

### Unavailable Date Constraint

* Applied technique: Calendar filtering / helper logic
* Unavailable dates are not counted directly as hard-score violations; instead, they are excluded from valid working-day calculations
* Detail:

[
W(b,s)={d \mid d \text{ is inside block range},\ d \text{ is a working day},\ d \notin \mathrm{Unavailable}(s)}
]

Therefore, unavailable days are excluded from workload and daily-count calculations.

### No Multiple Factories per Day

* Applied technique: HardScore
* A worker should not be assigned to multiple factories on the same day
* Detail:

[
|\mathrm{Factories}(e,d)| \le 1
]

This logic exists but is currently inactive.

### Stay Duration Constraint

* Applied technique: HardScore
* Consecutive and annual stay duration in the same region must not exceed the defined limits
* Detail:

[
\mathrm{staySpan}(e,r) \le \mathrm{maxStayOn}(r)
]

[
\mathrm{annualStay}(e,r) \le \mathrm{maxAnnualStay}(r)
]

Currently implemented but inactive.

### Overtime Limit

* Applied technique: HardScore
* Monthly and annual cumulative overtime must not exceed the upper bound
* Detail:

[
OT^{day}(e,d)=\max(0,\ H(e,d)-8)
]

[
OT^{month}(e,y,m)\le L^{month}(\mathrm{company}(e))
]

[
OT^{year}(e,y)\le L^{year}(\mathrm{company}(e))
]

Currently implemented but inactive.

---

## Soft Constraints

### Earlier Start

* Applied technique: SoftScore
* Prefer earlier start dates when possible
* Detail:

[
\mathrm{SoftScore} = -\left(\mathrm{startDay}(b)-\mathrm{windowStart}(b)\right)
]

### Smaller Daily Hours

* Applied technique: SoftScore + strength comparator
* Prefer smaller daily-hour values when possible
* Detail:

[
\mathrm{SoftScore} \propto -,\mathrm{hours}(b)
]

### Recommended Headcount Range

* Applied technique: MediumScore
* Prefer the actual staffing level to stay inside the recommended min-max range
* Detail:

[
n(b)=|{s\in S_b\mid \mathrm{employee}(s)\neq \varnothing}|
]

[
\delta(b)=
\begin{cases}
\mathrm{recMin}(b)-n(b) & n(b)<\mathrm{recMin}(b)\
n(b)-\mathrm{recMax}(b) & n(b)>\mathrm{recMax}(b)\
0 & \text{otherwise}
\end{cases}
]

[
\mathrm{MediumScore} = - w \cdot 3^{\delta(b)}
]

### Suppress Over-Recommended Staffing

* Applied technique: MediumScore
* Seats beyond the recommended range are given lower priority
* Detail:

[
\mathrm{MediumScore}(s)=
\begin{cases}
0 & \mathrm{seatIndex}(s) < \mathrm{recMin}(b)\
-1 & \mathrm{recMin}(b)\le \mathrm{seatIndex}(s)<\mathrm{recMax}(b)\
-5(\mathrm{seatIndex}(s)-\mathrm{recMax}(b)+1)^2 & \mathrm{seatIndex}(s)\ge \mathrm{recMax}(b)
\end{cases}
]

### Workload Balancing

* Applied technique: SoftScore
* Reduce imbalance in total assigned hours among workers
* Detail:

[
H^{tot}(e)=H^{dyn}(e)+H^{fixed}(e)
]

[
\mathrm{SoftScore}\propto -|H^{tot}(e)-H^{target}|
]

### Same Company Pairing

* Applied technique: SoftScore
* Prefer worker combinations from the same company within the same block
* Detail:

[
\mathrm{SoftScore}\propto +#{(s_i,s_j)\in S_b^2 \mid \mathrm{company}(s_i)=\mathrm{company}(s_j)}
]

### Average Skill Adjustment

* Applied technique: SoftScore
* Keep the average block skill close to the target average skill
* Detail:

[
\bar{\sigma}(b)=\frac{1}{n(b)}\sum_{s\in S_b}\mathrm{skill}(\mathrm{employee}(s),\mathrm{op}(b))
]

[
\mathrm{SoftScore}\propto -|\bar{\sigma}(b)-\mathrm{avgSkill}(\mathrm{op}(b))|
]

### Skill Diversity

* Applied technique: SoftScore
* Avoid overly uniform skill composition within the same block
* Detail:

[
\mathrm{SoftScore}\propto -\sum_{\ell} n_{\ell}(b)\left(n_{\ell}(b)-1\right)
]

### Regional / Customer Suitability

* Applied technique: SoftScore
* Prefer workers whose region and customer preferences better match the assignment
* Detail:

[
p_r(e,r)=\mathrm{regionPreference}(e,r),\quad
p_c(e,c)=\mathrm{customerPreference}(e,c)
]

[
\mathrm{SoftScore}\propto f(p_r(e,r),p_c(e,c))
]

### Regional Continuity

* Applied technique: SoftScore
* Prefer keeping workers continuously assigned within the same region
* Detail:

[
\mathrm{SoftScore}\propto +\mathrm{continuousStay}(e,r)
]

Implemented but currently inactive.

---

If you want, I can next turn this into a cleaner **English company-document style version** with slightly more formal wording.

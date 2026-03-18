## 1. Overall Model Structure


## 2. Planning Facts and Core Data

### 2.1 DaySlot

`DaySlot` represents one calendar day inside the planning horizon.  
It stores the day ID and the actual date value.  
This is the basic time unit used throughout the model.

### 2.2 EmployeeFact

`EmployeeFact` represents the fixed information of each worker.  
It stores worker ID, worker code, name, skill map, manager flag, worker company, regional preference, customer preference, and historical fixed assignment information.  
The historical information includes total fixed hours, fixed hours by day, fixed factories by day, last fixed day, last fixed region, and last fixed factory.

### 2.3 TaskWindow

`TaskWindow` represents the original scheduling frame of each task before solver decisions are created.  
It stores the module, workflow, factory, phase, operation, available start and end day range, allowed working-hour values, minimum headcount, maximum headcount, workload size, and recommended headcount range.  
This is the larger task definition that describes what may be scheduled.

### 2.4 Calendar and Environment Data

The model also uses calendar-related background data such as factory off-days, region off-days, customer off-days, worker-company off-days, worker unavailable days, region mapping, customer mapping, transit-day mapping, region stay limits, and overtime limits.  
These data are loaded into the internal `Calendars` structure and are used during feasibility checks.

## 3. Planning Entities

### 3.1 BlockDecision

`BlockDecision` is the main planning entity for task execution decisions.  
It is derived from the task window and stores the selected scheduling values for one task block.  
Its main fact-side fields include module, workflow, factory, phase, operation, window start, window end, required hours, allowed hours, minimum headcount, maximum headcount, and recommended headcount range.  
Its planning variables are `startDay`, `days`, and `hours`.  
In simple terms, `TaskWindow` defines the schedulable frame, while `BlockDecision` defines how that task is actually scheduled by the solver.

### 3.2 CrewSeat

`CrewSeat` is the worker-assignment planning entity under each block.  
Each seat belongs to one `BlockDecision` and represents one possible worker assignment slot.  
It stores block ID, module, workflow, factory, phase, operation, seat index, and whether the seat requires a manager.  
It also stores assignment-state fields for fixed or pinned rows, such as pinned worker, pinned start, pinned days, pinned hours, and pinned work days.  
The planning variable of this entity is the assigned `EmployeeFact`.

### 3.3 Candidate Range per Seat

Each `CrewSeat` has its own candidate worker range.  
For fixed seats, the candidate range is limited to the fixed worker if available.  
For manager-required seats, the candidate range is restricted to managers only.  
For normal seats, real workers plus an `UNASSIGNED` fallback value may be included.  
This allows the model to distinguish between hard-fixed assignment, manager-only assignment, and general assignment flexibility.

## 4. Planning Solution

### 4.1 SinglePassPlan

`SinglePassPlan` is the full planning solution passed into the solver.  
It contains `days`, `employees`, `blocks`, and `seats`, together with the `HardMediumSoftScore`.  
This structure allows the solver to optimize timing decisions and worker assignment decisions in one unified model.

## 5. Data Relationship Summary

The current model can be summarized as follows:

- `DaySlot` defines the planning timeline.
    
- `EmployeeFact` defines worker-side fixed data and historical data.
    
- `TaskWindow` defines the original schedulable task frame.
    
- `BlockDecision` defines the actual solver-side scheduling decision for that task.
    
- `CrewSeat` defines who is assigned to each block.
    
- `SinglePassPlan` groups all facts and entities into one optimization model.
    

---

# Draft: Constraint Logic

## 6. Score Structure

The current solver uses `HardMediumSoftScore`.  
The active constraints are organized as hard constraints and soft constraints, with hard feasibility rules prioritized over optimization preferences.

## 7. Currently Applied Hard Constraints

### 7.1 endWithinWindow

This constraint prevents a block from ending after its allowed scheduling window.  
If `startDay + days - 1` exceeds `windowEnd`, the block is penalized.  
This ensures the scheduled block remains inside its allowed end boundary.

### 7.2 hoursValueAllowed

This constraint ensures that the selected daily work hours of a block are contained in the allowed hour list.  
If the selected value is outside the allowed set, the block is penalized as infeasible.

### 7.3 phaseOrder

This constraint keeps the phase sequence of the same module in the correct order.  
A later phase must not start before the previous phase has finished.  
This is checked by joining consecutive phases within the same module and comparing their scheduled timing.

### 7.4 noUnderfillByBlock

This constraint ensures that each block produces at least the required amount of work.  
The model calculates block production from assigned seats and compares it with `requiredHours`.  
If the produced amount is below the required amount, a strong hard penalty is applied.

### 7.5 overfillAtMostOneDayByBlock

This constraint limits excessive overproduction.  
The model allows some overfill, but only up to roughly one extra working day per staffed flexible seat.  
If the block exceeds that level, it receives a hard penalty.

### 7.6 oneTaskPerEmpPerDay

This constraint prevents one worker from being assigned to multiple tasks on the same day.  
The calculation includes dynamic assignments from the current solution together with fixed task-count history from the background data.  
If the combined count exceeds one, the worker-day pair is penalized.

### 7.7 dailyCap12h

This constraint limits each worker’s total daily working hours to the daily cap.  
The current cap is controlled by `DAILY_CAP`, which is set to 12.  
The calculation combines dynamic scheduled hours with fixed historical hours for the same worker and day.

### 7.8 regionTransitGap

This constraint enforces a required gap when a worker moves between different regions.  
The region is derived from the assigned factory, and the required transition gap comes from the transit-day mapping in the calendar data.  
The logic also considers the worker’s last fixed historical region as a starting point for continuity checks.

---

## 8. Currently Applied Soft Constraints

### 8.1 preferSmallerHours

This constraint prefers smaller daily work-hour values.  
It encourages the solver to choose lighter daily hour settings when possible.

### 8.2 preferEarlierStart

This constraint prefers earlier start dates within the allowed task window.  
The penalty is based on delay relative to the earliest valid day in the block window.

### 8.3 softSameCompanyPairs

This constraint rewards worker combinations from the same company inside the same block.  
It encourages team composition with company consistency.

### 8.4 softEncourageSkillVariety

This constraint discourages too many workers with the same skill level from being assigned together to the same block and operation.  
Its purpose is to make the block-level team composition less uniform.

### 8.5 softBalanceBlockAvgSkill

This constraint compares the average assigned skill of a block with the expected average skill for the operation.  
A larger difference leads to a larger penalty.  
It aims to keep the average team skill close to the intended level.

### 8.6 softBalanceTotalHours

This constraint balances total assigned hours across workers.  
It sums dynamic scheduled hours and historical fixed hours, then compares the result with the target workload per employee.  
This helps avoid concentration of work on a small number of workers.

### 8.7 softFabPreference

This constraint uses regional and customer preference data stored in `EmployeeFact`.  
It encourages assignments that better match worker-side preference information and factory-related attributes.

### 8.8 recommendHeadcount

This constraint prefers staffing levels that stay close to the recommended headcount range of the task.  
The relevant fields are `recommendMinHeads` and `recommendMaxHeads`, which exist both in `TaskWindow` and `BlockDecision`.

### 8.9 preferSeatPriorityByRecommendRange

This constraint lowers the priority of extra assignment beyond the recommended range.  
It helps distinguish between truly recommended staffing and merely allowed staffing up to maximum capacity.

---

## 9. Constraints Present in Code but Currently Not Activated

###  employeeAvailableAllDays

This constraint checks whether a worker is assigned on unavailable days.  
The logic exists in the code, but it is currently commented out in `defineConstraints()`.

###  oneFactoryPerEmpPerDay

This constraint is intended to prevent one worker from being assigned to multiple factories on the same day.  
Its logic also considers fixed factory history, but it is currently commented out.

###  regionStayMaxOn

This is an extension constraint for limiting consecutive stay duration within a region.  
It is referenced as a candidate but not currently active.

###  regionAnnualStayMax

This is an extension constraint for limiting annual stay duration in a region.  
The related calendar data structure exists, but the constraint is not currently active.

###  annualOvertimeLimit / monthlyOvertimeLimit

These are extension constraints for controlling worker overtime by year and by month.  
The model already stores fixed overtime background data and company-level overtime limits, but these constraints are currently not applied in `defineConstraints()`.

###  softContinuousRegionStay

This is a soft extension constraint related to regional continuity.  
The logic exists in the code but is currently not active in the applied constraint list.
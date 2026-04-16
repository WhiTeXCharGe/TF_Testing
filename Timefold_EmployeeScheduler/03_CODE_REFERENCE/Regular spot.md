# Draft ideas: Regular / Spot worker type  
  
Just gathering ideas for now, not final design.  
  
Based on current solver flow, this feature probably does **not** need a big redesign.    
It looks like the main change would be around:  
  
- worker data in `EnvConfig`  
- employee fact fields  
- candidate ordering for each seat  
- maybe soft score later if needed  
  
The main meaning is:  
  
- **regular** should be tried first  
- **spot** should be used after that  
- both are still valid workers unless business says otherwise  
  
So this feels more like **priority** than a hard rule.  
  
---  
  
## Common implementation direction  
  
For all cases, the most natural idea is:  
  
1. build feasible candidates as usual    
2. among those candidates, sort:  
   - regular first  
   - spot second  
   - then manager / skill / preference / balance as today  
  
So in Timefold terms, this feels closer to:  
  
- comparator  
- ordered candidate list  
- maybe soft penalty  
  
and **not** really a ValueRangeFactory problem.  
  
Because the legal range is still the same.    
The issue is not “who is allowed”, but “who should be picked first”.  
  
---  
  
# 1. Global regular / spot per employee  
  
## Image  
  
Each worker has only one type for everything.  
  
Example:  
  
- A = regular  
- B = spot  
  
No fab split, no phase split.  
  
## EnvConfig draft  
  
```yaml  
worker_list:  
  - id: w1  
    name: A  
    worker_type: regular  
  
  - id: w2  
    name: B  
    worker_type: spot
```
Maybe missing field = `spot` by default.

## Code draft

In `EmployeeFact`:

public String workerType;

## Solver idea

Very simple.

When candidates are prepared, sort them so:

- regular first
- then spot

Sample idea:

int roleRank(EmployeeFact e) {  
    return "regular".equalsIgnoreCase(e.workerType) ? 0 : 1;  
}

Then use that in comparator.

## Comment

This is the easiest version.  
Good for testing the concept first.  
But probably too rough for real use.

---

# 2. Regular / spot depends on phase or operation

## Image

Worker can be regular for some tasks and spot for others.

Example:

- A = regular for p1, p2
- A = spot for p3, p4

This feels much more realistic.

## EnvConfig draft

worker_list:  
  - id: w1  
    name: A  
    worker_type_by_operation:  
      p1: regular  
      p2: regular  
      p3: spot  
      p4: spot

Or phase style:

worker_type_by_phase:  
  tool_p1: regular  
  tool_p2: regular  
  tool_p3: spot  
  tool_p4: spot

I think operation-based is a bit easier because your current model already uses `p1/p2/p3/p4` a lot.

## Code draft

public Map<String, String> workerTypeByOperation = new HashMap<>();

Helper idea:

String getTypeForOp(String opId) {  
    return workerTypeByOperation.getOrDefault(opId, "spot");  
}

## Solver idea

For each seat/block, check worker type for that operation.

Then order candidates:

- regular for this op first
- spot for this op second

## Comment

This is probably the best balance.

Not too hard, not too weak.

---

# 3. Regular depends on fab + task

## Image

Worker is regular only for some fab/task combinations.

Example:

- regular for `f3 + p1`
- regular for `f3 + p2`
- spot for most others

This is the detailed version.

Because fab count is large, storing both regular and spot everywhere would be too heavy.  
So your idea makes sense:

- store **regular only**
- everything else = spot by default

## EnvConfig draft

### Style A

worker_list:  
  - id: w1  
    name: A  
    regular_assignment_map:  
      f3: [p1, p2]  
      f10: [p4]

### Style B

worker_list:  
  - id: w1  
    name: A  
    regular_assignment_keys:  
      - "f3|p1"  
      - "f3|p2"  
      - "f10|p4"

I think Style B is easier in code.

## Code draft

public Set<String> regularAssignmentKeys = new HashSet<>();

Helper:

boolean isRegularFor(String fabId, String opId) {  
    return regularAssignmentKeys.contains(fabId + "|" + opId);  
}

## Solver idea

When building seat candidates:

- if worker matches regular key for this fab/op → treat as regular
- else → treat as spot

Then sort regular first.

## Comment

This is probably the best long-term version if the business really works this way.  
But data prep is heavier.

---

# Technique choice draft

## Comparator / ordered candidates

This feels like the main tool.

Use when:

- all workers are still allowed
- but regular should be tried before spot

This sounds like your case.

## ValueRangeProvider / ValueRangeFactory

Probably not the main tool.

Reason:  
the value range is still basically the same employees.  
What changes is priority, not legality.

## Soft score

Can be added later if comparator is not enough.

Example idea:

- penalize assigning spot when some regular candidate was possible

Not needed for first version, but can help.

## Hard rule

Only if business really says:

- do not assign spot while regular exists

I would not start with this.

---

# Small code style idea

Maybe candidate handling can be like this:

List<EmployeeFact> regulars = new ArrayList<>();  
List<EmployeeFact> spots = new ArrayList<>();  
  
for (EmployeeFact e : feasibleEmployees) {  
    if (isRegularForThisSeat(e, seat, block)) regulars.add(e);  
    else spots.add(e);  
}  
  
regulars.sort(existingComparator);  
spots.sort(existingComparator);  
  
List<EmployeeFact> ordered = new ArrayList<>();  
ordered.addAll(regulars);  
ordered.addAll(spots);

This is simple and easy to read.

---

# Other possible case

## Region + task instead of fab + task

Maybe real business is not fab-based but region-based.

Then data becomes smaller.

Example:

worker_list:  
  - id: w1  
    name: A  
    regular_region_op_keys:  
      - "r3|p1"  
      - "r3|p2"

This could be a middle option between case 2 and case 3.

---

## Customer + task

If same customer has many fabs, maybe this is more useful than fab-specific.

regular_customer_op_keys:  
  - "c3|p1"  
  - "c3|p2"

---

## Mandatory regular seat

Another separate case:

- at least one regular worker must be inside the task
- or seat 0 must be regular

This is no longer just priority.  
This becomes a constraint case.

---

# Rough recommendation

## If want fastest test

Use case 1 first.

## If want the most practical version

Case 2 looks best.

## If real business is strongly fab-based

Use case 3, but only store regular mapping.

---

# Simple conclusion draft

This feature looks more like a **candidate priority** feature than a domain/range feature.

So first idea would be:

- add regular/spot data to worker
- detect regular/spot relative to seat
- sort feasible candidates with regular first
- keep spot as fallback
- maybe add soft penalty later if needed

So for Timefold, I would first think about:

- comparator
- candidate ordering
- optional soft score later

not ValueRangeFactory as the main solution.



# Next step draft: task-based regular / spot

This note assumes the direction is:

- decide **regular / spot by task**
- use task meaning as current `opId` such as `p1`, `p2`, `p3`, `p4`
- keep current skill, manager, availability, region/customer gate logic as-is
- add regular / spot as one more preference layer on top

In the current model, employee assignment already goes through:

- `EmployeeFact` as worker data
- block/seat data on `CrewSeat` and `BlockDecision`
- seat candidate preparation in `fillSeatCandidatesSinglePass(...)`
- seat value range in `eligibleEmployeesForSeat()`
- employee planning variable with `strengthComparatorClass = EmployeeStrength.class`
- soft constraints such as fab preference and block/team balance

That means task-based regular / spot fits the current structure quite naturally. The current code already stores per-employee skill and per-employee region/customer preference, and seat candidates are built per task (`s.opId`) before solving. fileciteturn5file0turn5file2

---

## 1. Goal of this version

The business meaning is:

- a worker may be **regular** for some tasks
- the same worker may be **spot** for other tasks
- for example, worker A is regular for `p1` and `p2`, but spot for `p3` and `p4`

This is different from skill.

- **skill** means the worker can do the task at all
- **regular / spot** means among valid workers, who is the normal/core member and who is backup

So the rough priority becomes:

$$
\text{feasible worker} = \text{skill OK} \land \text{availability OK} \land \text{manager rule OK} \land \text{region/customer gate OK}
$$

Then inside the feasible pool, regular / spot can guide the search:

$$
\text{priority score} = \text{regular-first term} + \text{other preference terms}
$$

Timefold’s model supports exactly this kind of split: planning variables get values from value ranges, and the final solution is guided by hard constraints and soft constraints. Constraint streams use `penalize()` and `reward()` to shape that score. citeturn502251search2turn633415search3turn502251search1

---

## 2. EnvConfig storage idea

Since the plan is task-based, the simplest storage is on each worker in `worker_list`.

### Option A: explicit map by operation

```yaml
worker_list:
  - id: w101
    name: Worker A
    worker_company: wc1
    is_manager: false
    skill_map:
      p1: 1
      p2: 1
      p3: 1
      p4: 0
    worker_type_by_operation:
      p1: regular
      p2: regular
      p3: spot
      p4: spot
```

This is the clearest format.

### Option B: store only regular tasks

```yaml
worker_list:
  - id: w101
    name: Worker A
    worker_company: wc1
    is_manager: false
    skill_map:
      p1: 1
      p2: 1
      p3: 1
      p4: 0
    regular_operations:
      - p1
      - p2
```

Meaning:

- if task is listed in `regular_operations` → regular
- otherwise → spot

This version is smaller and probably easier to maintain.

### Option C: task + certificate meaning together

```yaml
worker_list:
  - id: w205
    name: Worker B
    worker_company: wc2
    is_manager: true
    skill_map:
      p1: 1
      p2: 1
      p3: 1
      p4: 1
    regular_operations:
      - p1
      - p2
    certificate_tags:
      - install_lead
      - setup_core
```

This is useful if later you want to explain *why* the person is regular.

### Suggested choice

For the first implementation, Option B is probably the easiest:

- shorter YAML
- fallback is simple
- less duplication

---

## 3. Recommended Java shape

A small addition to `EmployeeFact` is enough.

### Option A: map version

```java
public Map<String, String> workerTypeByOperation = new HashMap<>();
```

### Option B: regular-only version

```java
public Set<String> regularOperations = new HashSet<>();
```

With helper:

```java
public boolean isRegularFor(String opId) {
    return regularOperations != null && regularOperations.contains(opId);
}
```

Since the current model already uses `s.opId` and `skill(e, s.opId)`, task-based regular / spot lines up cleanly with the existing task token model. fileciteturn5file0turn5file2

---

## 4. How it should be used in Timefold

There are two main places to use it.

### 4.1 Candidate priority during employee assignment

Right now, seat candidates are prepared in `fillSeatCandidatesSinglePass(...)`.
The current gating already checks:

- skill
- region/customer preference gate (`0` blocks candidate)
- personal availability
- manager requirement

Then the seat returns those candidates through `eligibleEmployeesForSeat()`. fileciteturn5file0turn5file2

So the first use of regular / spot is:

$$
\text{candidate order for seat} = [\text{regular workers for } opId] + [\text{spot workers for } opId]
$$

This does **not** replace current gates.
It only changes the order inside the already-feasible pool.

A simple draft comparator is:

```java
static int regularRank(EmployeeFact e, String opId) {
    if (e == null || e.id == 0) return 999;
    return e.isRegularFor(opId) ? 0 : 1;
}
```

And then in candidate sort:

```java
Comparator<EmployeeFact> cmp =
    Comparator.comparingInt((EmployeeFact e) -> regularRank(e, s.opId))
              .thenComparing((EmployeeFact e) -> !e.isManager)
              .thenComparingInt((EmployeeFact e) -> -skill(e, s.opId))
              .thenComparingInt(e -> e.id);

cand.sort(cmp);
```

This is a practical next step because the current code already uses a seat-specific candidate list and already has an `EmployeeStrength` comparator on the employee planning variable. Timefold also documents that planning values come from `@ValueRangeProvider` and planning variables are then assigned from those values during solving. fileciteturn5file0turn5file2 citeturn633415search3turn502251search2

### 4.2 Soft score to prefer regular over spot

Comparator/order helps early search, but score should also express the business meaning.

The clean soft idea is:

- no penalty if assigned employee is regular for the task
- soft penalty if assigned employee is spot for the task

A simple math view is:

$$
\text{RegularSpotPenalty} = \sum_{s \in \text{assigned seats}} I(\text{assigned}(s) \text{ is spot for } op(s))
$$

where

$$
I(x)=
\begin{cases}
1 & \text{if } x \text{ is true} \\
0 & \text{otherwise}
\end{cases}
$$

Then the score contribution can be:

$$
\text{Score} = \text{existing score} - w_{spot} \cdot \text{RegularSpotPenalty}
$$

This matches Timefold’s constraint-stream scoring model, where you can penalize or reward solution patterns with soft or medium weights. citeturn502251search1

A small draft constraint could look like this:

```java
Constraint preferRegularByTask(ConstraintFactory f) {
    return f.forEach(CrewSeat.class)
        .filter(s -> !isUnassigned(s.employee))
        .filter(s -> s.employee != null && !s.employee.isRegularFor(s.opId))
        .penalize(HardMediumSoftScore.ONE_SOFT)
        .asConstraint("prefer-regular-by-task");
}
```

Or weighted:

```java
static final int REGULAR_SPOT_W = 20;

Constraint preferRegularByTask(ConstraintFactory f) {
    return f.forEach(CrewSeat.class)
        .filter(s -> !isUnassigned(s.employee))
        .filter(s -> s.employee != null && !s.employee.isRegularFor(s.opId))
        .penalize(HardMediumSoftScore.ONE_SOFT, s -> REGULAR_SPOT_W)
        .asConstraint("prefer-regular-by-task");
}
```

Because your current solver already uses `HardMediumSoftScore` and already has multiple soft constraints, this fits the current scoring structure well. fileciteturn5file0turn5file2

---

## 5. Relationship with current region/customer preference

Current model already has:

```java
public Map<String,Integer> regionPreference   = new HashMap<>();
public Map<String,Integer> customerPreference = new HashMap<>();
```

and current candidate building blocks workers when either region or customer preference is `0`. It also keeps positive preference values available for later soft scoring. fileciteturn5file0turn5file2

So the clean separation is:

- **skill** = can do task or not
- **region/customer preference** = can go there or not, and how suitable the destination/customer is
- **regular / spot** = core member vs backup member for the task

That can be written as:

$$
\text{assignable}(e,s) =
\big(\text{skill}(e,op_s) \ge 1\big)
\land \big(\text{regionPref}(e,r_s) > 0\big)
\land \big(\text{customerPref}(e,c_s) > 0\big)
\land \big(\text{availability OK}\big)
$$

and then among assignable workers:

$$
\text{prefer regular}(e,s) =
\begin{cases}
1 & \text{if } e \text{ is regular for } op_s \\
0 & \text{if } e \text{ is spot for } op_s
\end{cases}
$$

So regular / spot should be treated as **another preference layer**, not as a replacement for current region/customer gating.

---

## 6. Practical next steps

### Step 1: decide data format

Recommended first choice:

```yaml
regular_operations:
  - p1
  - p2
```

with default = spot.

### Step 2: add field on `EmployeeFact`

Use:

```java
public Set<String> regularOperations = new HashSet<>();
```

### Step 3: parse from EnvConfig

Read `regular_operations` when building each worker.
If field is missing, keep empty set.

### Step 4: use it in seat candidate order

Inside `fillSeatCandidatesSinglePass(...)`, after all current gates pass, sort `cand` so that:

- regular for `s.opId` first
- spot later

### Step 5: add one soft constraint

Add a soft penalty for spot assignment on a seat.
That gives the business meaning directly in score.

### Step 6: tune weight

Start small, then compare output.
For example:

- too small → effect is weak
- too big → solver may over-focus on regular and ignore balance too much

---

## 7. Minimal example block for the note

```java
public static class EmployeeFact {
    @PlanningId public int id;
    public String wid;
    public String name;
    public Map<String,Integer> skills = new HashMap<>();
    public boolean isManager;
    public String workerCompany;

    public Map<String,Integer> regionPreference   = new HashMap<>();
    public Map<String,Integer> customerPreference = new HashMap<>();
    public Set<String> regularOperations = new HashSet<>();

    public boolean isRegularFor(String opId) {
        return regularOperations != null && regularOperations.contains(opId);
    }
}
```

```java
Comparator<EmployeeFact> cmp =
    Comparator.comparingInt((EmployeeFact e) -> e.isRegularFor(s.opId) ? 0 : 1)
              .thenComparing((EmployeeFact e) -> !e.isManager)
              .thenComparingInt((EmployeeFact e) -> -skill(e, s.opId))
              .thenComparingInt(e -> e.id);

cand.sort(cmp);
```

```java
Constraint preferRegularByTask(ConstraintFactory f) {
    return f.forEach(CrewSeat.class)
        .filter(s -> !isUnassigned(s.employee))
        .filter(s -> s.employee != null && !s.employee.isRegularFor(s.opId))
        .penalize(HardMediumSoftScore.ONE_SOFT)
        .asConstraint("prefer-regular-by-task");
}
```

---

## 8. Final draft conclusion

If the chosen direction is **regular / spot by task**, then the clean next step is:

- store regular membership per task in `EnvConfig`
- keep current skill, manager, availability, region/customer gates unchanged
- sort feasible candidates so regular comes first for that task
- add a soft penalty so spot assignment is still allowed, but less preferred

That approach matches the current single-pass model well because the solver already:

- builds seat candidates per task
- exposes employee candidates through a value range on `CrewSeat`
- uses a planning variable for employee assignment
- scores the solution with `HardMediumSoftScore` and constraint streams. fileciteturn5file0turn5file2 citeturn502251search2turn502251search1turn633415search3

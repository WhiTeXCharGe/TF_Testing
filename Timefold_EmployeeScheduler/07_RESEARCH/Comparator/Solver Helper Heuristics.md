# Solver Helper Heuristics 
  
These are **helper heuristics** to guide the solver toward better search paths.    
They **do not replace constraints** and are only intended to improve search efficiency.  
  
---  
  
# Task / Block Priority Heuristics  
  
These heuristics help the solver focus on **difficult tasks first**.  
  
---  
  
## Smaller Date Window  
  
Tasks with fewer scheduling days should be prioritized first.  
  
```java  
static int windowSize(BlockDecision b) {  
    if (b.windowStart == null || b.windowEnd == null)  
        return Integer.MAX_VALUE;  
    return b.windowEnd - b.windowStart;  
}

Comparator example:

if (windowSize(a) != windowSize(b))  
    return Integer.compare(windowSize(a), windowSize(b));

---

## Rarer Skill Requirement

Tasks that require a skill with **few qualified workers** should be prioritized.

static int eligibleWorkerCount(BlockDecision b, List<EmployeeFact> employees) {  
    int count = 0;  
    for (EmployeeFact e : employees) {  
        if (e.skills != null && e.skills.getOrDefault(b.operation, 0) > 0)  
            count++;  
    }  
    return count;  
}

Comparator example:

int ea = eligibleWorkerCount(a, employees);  
int eb = eligibleWorkerCount(b, employees);  
  
if (ea != eb)  
    return Integer.compare(ea, eb); // fewer workers first

---

## Longer Workload Tasks

Tasks consuming more worker-days should be solved earlier.

static int workload(BlockDecision b) {  
    return b.requiredHours;  
}

Comparator example:

if (workload(a) != workload(b))  
    return Integer.compare(b.requiredHours, a.requiredHours);

---

## Manager Requirement

Tasks requiring a manager seat should be prioritized.

static boolean requiresManager(BlockDecision b) {  
    return b.managerSeatCount > 0;  
}

Comparator example:

if (requiresManager(a) != requiresManager(b))  
    return requiresManager(a) ? -1 : 1;

---

## Fewer Eligible Employees

Blocks with fewer candidate employees are harder.

static int candidateCount(BlockDecision b) {  
    return b.eligibleEmployeeCount;  
}

Comparator example:

if (candidateCount(a) != candidateCount(b))  
    return Integer.compare(candidateCount(a), candidateCount(b));

---

## Earlier Phase

Earlier phases should generally be completed first.

static int phaseIndex(BlockDecision b) {  
    return b.phaseIndex; // p1=1 p2=2 p3=3 p4=4  
}

Comparator example:

if (phaseIndex(a) != phaseIndex(b))  
    return Integer.compare(phaseIndex(a), phaseIndex(b));

---

## Tighter Recommended Staffing

Blocks where `recommendMax - recommendMin` is small are stricter.

static int recommendSlack(BlockDecision b) {  
    return b.recommendMaxHeads - b.recommendMinHeads;  
}

Comparator example:

if (recommendSlack(a) != recommendSlack(b))  
    return Integer.compare(recommendSlack(a), recommendSlack(b));

---

# Combined Block Difficulty Comparator

Example combining the previous heuristics.

public static class BlockDifficultyComparator implements Comparator<BlockDecision> {  
  
    List<EmployeeFact> employees;  
  
    public BlockDifficultyComparator(List<EmployeeFact> employees) {  
        this.employees = employees;  
    }  
  
    @Override  
    public int compare(BlockDecision a, BlockDecision b) {  
  
        int winA = windowSize(a);  
        int winB = windowSize(b);  
        if (winA != winB)  
            return Integer.compare(winA, winB);  
  
        int eligA = eligibleWorkerCount(a, employees);  
        int eligB = eligibleWorkerCount(b, employees);  
        if (eligA != eligB)  
            return Integer.compare(eligA, eligB);  
  
        if (a.requiredHours != b.requiredHours)  
            return Integer.compare(b.requiredHours, a.requiredHours);  
  
        if (requiresManager(a) != requiresManager(b))  
            return requiresManager(a) ? -1 : 1;  
  
        if (phaseIndex(a) != phaseIndex(b))  
            return Integer.compare(phaseIndex(a), phaseIndex(b));  
  
        return Integer.compare(a.id, b.id);  
    }  
}

---

# Employee Priority Heuristics

These heuristics determine **which employee should be tried first for a seat**.

---

## Few Skill Employees First

Workers with fewer skills should be assigned first to preserve flexible workers.

static int skillCount(EmployeeFact e) {  
    if (e.skills == null)  
        return 0;  
    return e.skills.size();  
}

Comparator usage:

if (skillCount(a) != skillCount(b))  
    return Integer.compare(skillCount(a), skillCount(b));

---

## Prefer Same Region

Workers already suitable for the region should be preferred.

static boolean sameRegion(EmployeeFact e, BlockDecision b) {  
    return e.regions != null && e.regions.contains(b.region);  
}

Comparator example:

boolean ar = sameRegion(a, block);  
boolean br = sameRegion(b, block);  
  
if (ar != br)  
    return ar ? -1 : 1;

---

## Prefer Same Fab

Workers who already work in the same fab should be preferred.

static boolean sameFab(EmployeeFact e, BlockDecision b) {  
    return e.fabs != null && e.fabs.contains(b.factory);  
}

Comparator example:

boolean af = sameFab(a, block);  
boolean bf = sameFab(b, block);  
  
if (af != bf)  
    return af ? -1 : 1;

---

## Lower Current Workload

Prefer workers with fewer assigned hours.

static int workload(EmployeeFact e) {  
    return e.currentAssignedHours;  
}

Comparator example:

if (workload(a) != workload(b))  
    return Integer.compare(workload(a), workload(b));

---

## Exact Skill Match

Prefer workers with higher skill level for the required operation.

static int opSkill(EmployeeFact e, String op) {  
    if (e.skills == null)  
        return 0;  
    return e.skills.getOrDefault(op, 0);  
}

Comparator example:

int sa = opSkill(a, block.operation);  
int sb = opSkill(b, block.operation);  
  
if (sa != sb)  
    return Integer.compare(sb, sa);

---

## Shorter Travel Gap

Prefer workers already near the required region.

static int travelGap(EmployeeFact e, BlockDecision b) {  
    if (e.lastRegion == null)  
        return 999;  
    return regionDistance(e.lastRegion, b.region);  
}

Comparator example:

if (travelGap(a, block) != travelGap(b, block))  
    return Integer.compare(travelGap(a, block), travelGap(b, block));

---

# Combined Employee Comparator Example

public static class EmployeeStrength implements Comparator<EmployeeFact> {  
  
    @Override  
    public int compare(EmployeeFact a, EmployeeFact b) {  
  
        if (a == b) return 0;  
        if (a == null) return 1;  
        if (b == null) return -1;  
  
        boolean aUnassigned = "__UNASSIGNED__".equals(a.wid);  
        boolean bUnassigned = "__UNASSIGNED__".equals(b.wid);  
  
        if (aUnassigned != bUnassigned)  
            return aUnassigned ? 1 : -1;  
  
        if (a.isManager != b.isManager)  
            return a.isManager ? -1 : 1;  
  
        int aSkills = skillCount(a);  
        int bSkills = skillCount(b);  
        if (aSkills != bSkills)  
            return Integer.compare(aSkills, bSkills);  
  
        int wa = workload(a);  
        int wb = workload(b);  
        if (wa != wb)  
            return Integer.compare(wa, wb);  
  
        return Integer.compare(a.id, b.id);  
    }  
}
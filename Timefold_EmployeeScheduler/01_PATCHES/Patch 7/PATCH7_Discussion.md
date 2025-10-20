# Patch 7 — Overall Plan: Two-Pass Scheduling (Block → Assignment)

## Why this patch

The current “single-pass” approach mixes two very different decisions:

1. sizing and timing of work blocks (when/how many people/how many days), and
    
2. assigning specific people to those blocks.
    

Coupling them makes feasibility harder, encourages cycling, and obscures where constraints fail. Patch 7 separates concerns into two passes to improve feasibility rate, clarity, and tuning.

## What changes at a high level

- **Pass 1: Block Construction**
    
    - Produce a timeline of “blocks” per module+operation: start_day, heads, days, daily hours.
        
    - Satisfy phase windows and ordering first; respect per-op staffing min/max.
        
    - Output: a feasible block plan (no people yet) plus diagnostics on coverage gaps.
        
- **Pass 2: People Assignment**
    
    - Fill each block with named workers per day.
        
    - Respect skills, company mix preferences, manager coverage, OT caps, and fairness/variety soft rules.
        
    - Output: per-person daily schedule and team-quality metrics.
        

## Expected benefits

- **Higher feasibility**: hard constraints tackled in the right layer (windows/order/min-max in Pass 1; skills/OT in Pass 2).
    
- **Faster iterations**: can adjust block counts/lengths without redoing all assignments.
    
- **Better analytics**: clear KPIs per pass (coverage vs. staffing quality).
    
- **Cleaner tuning**: different move sets/parameters per pass (small date/head tweaks vs. swaps/reassigns).
    

## Core flow

1. Seed blocks from plan windows and workload_days.
    
2. Optimize blocks to hit coverage with minimal breaches.
    
3. Freeze blocks and derive per-day seat demand.
    
4. Assign people to seats, optimizing skill fit, cohesion/variety, OT and fairness.
    
5. Export Excel (Tasks×Dates, Employees×Dates, Dashboard, Breaches) for review.
    

## Data/model impacts

- **New “BlockDecision”** entity (module, op_id, start_day, heads, days, hours).
    
- **Derived “Seat”** demand per block/day for Pass 2.
    
- **Constraints split**:
    
    - Pass 1 hard: phase window/order, staffing min/max; soft: smooth starts, minimal shifts.
        
    - Pass 2 hard: skill match, daily caps, manager per block; soft: company cohesion, skill variety, fairness.
        

## Outputs & KPIs

- **Pass 1**: coverage (% required hours met), breaches by type, block stability.
    
- **Pass 2**: feasibility rate (hard=0), time to first feasible, OT hours, fairness (stdev/CoV), team quality (balance vs variety).
    
- **Overall**: completion %, delays vs plan, utilization%.
    

## Rollout plan

- 7.0: Introduce two-pass skeleton, keep current defaults for search.
    
- 7.1–7.3: Tune block heuristics and assignment neighborhoods; add dashboards.
    
- 7.4.x: Stabilize KPIs, refine soft scores, address edge-case breaches.
    

## Risks & mitigations

- Risk: infeasible blocks starve Pass 2.
    
    - Mitigation: conservative min/max, early detection and backoff in Pass 1.
        
- Risk: handover mismatch (seats vs. available skills).
    
    - Mitigation: feedback loop to nudge heads/days where skill pools are thin.
        
- Risk: performance on large horizons.
    
    - Mitigation: limit move scope per pass; incremental recalculation; batched exports.
        

## Out of scope for Patch 7

- Multi-site cross-fab leveling beyond current inputs.
    
- New algorithm families; focus is on restructuring and metrics.
    

**Bottom line:** Patch 7 reframes scheduling as two simpler problems in sequence—build the right blocks, then staff them well—unlocking better feasibility, clearer analytics, and faster tuning.
package com.yourorg.scheduler.constraints;

import java.util.*;

import ai.timefold.solver.core.api.score.HardMediumSoftScore;
import ai.timefold.solver.core.api.score.stream.Constraint;
import ai.timefold.solver.core.api.score.stream.ConstraintCollectors;
import ai.timefold.solver.core.api.score.stream.ConstraintFactory;
import ai.timefold.solver.core.api.score.stream.Joiners;

import com.yourorg.scheduler.calendar.SolveContext;
import com.yourorg.scheduler.model.*;
import static com.yourorg.scheduler.constraints.ConstraintSupport.clampToInt;
import static com.yourorg.scheduler.constraints.ConstraintSupport.seatCoveredDays;

/** Soft (and medium) preferences (11 constraints). */
final class SoftConstraints {
    // soft weights
    static final int PREF_HOURS_WEIGHT = 3000;
    static final int SMALLER_HOURS_W   = 40;
    static final int EARLIER_START_W   = 1;
    static final int AFFINITY_W        = 5;
    static final int SKILL_DIVERSITY_W = 3;
    static final int SKILL_AVG_W       = 10;
    static final int REGION_TRANSITION_W = 5;
    static final int FAB_PREFERENCE_W  = 5;
    static final int RECOMMEND_HEADS_W = 50; // tune
    static final int REGULAR_SPOT_W    = 40;

    private static int staffedCountForBlock(List<CrewSeat> seats) {
        int c = 0;
        for (CrewSeat s : seats) if (!WorkerRules.isUnassigned(s.employee)) c++;
        return c;
    }

    // ---------- Softs ----------

    Constraint preferSmallerHours(ConstraintFactory f) {
        return f.forEach(BlockDecision.class)
            .penalize(HardMediumSoftScore.ONE_SOFT, b -> SMALLER_HOURS_W * b.chosenHours())
            .asConstraint("soft-smaller-hours");
    }

    // Constraint preferEarlierStart(ConstraintFactory f) {
    //     return f.forEach(BlockDecision.class)
    //         .penalize(HardMediumSoftScore.ONE_SOFT, b -> b.startDay == null ? 0 : EARLIER_START_W * b.startDay)
    //         .asConstraint("soft-earlier-start");
    // }

    Constraint preferEarlierStart(ConstraintFactory f) {
        return f.forEach(BlockDecision.class)
                // only consider blocks that actually have a startDay
                .filter(b -> b.startDay != null)
                .penalize(HardMediumSoftScore.ONE_SOFT, b -> {
                    // delay relative to earliest allowed day in the window
                    int delay = b.startDay - b.windowStart;
                    if (delay <= 0) {
                        return 0; // already as early as window allows
                    }
                    return EARLIER_START_W * delay;
                })
                .asConstraint("soft-earlier-start");
    }



    Constraint softAffinityPenalty(ConstraintFactory f) {
        return f.forEachIncludingUnassigned(CrewSeat.class)
            .filter(s -> !WorkerRules.isUnassigned(s.employee))
            .join(f.forEachIncludingUnassigned(CrewSeat.class).filter(s -> !WorkerRules.isUnassigned(s.employee)),
                Joiners.equal(s -> s.blockId),
                Joiners.lessThan(s -> s.employee.id))
            .penalize(HardMediumSoftScore.ONE_SOFT,
                (s1, s2) -> {
                    int edge = WorkerRules.computeAffinityEdge(s1.employee, s2.employee);
                    return AFFINITY_W * Math.max(0, -edge);
                })
            .asConstraint("soft-affinity-penalty");
    }

    Constraint softAffinityReward(ConstraintFactory f) {
        return f.forEachIncludingUnassigned(CrewSeat.class)
            .filter(s -> !WorkerRules.isUnassigned(s.employee))
            .join(f.forEachIncludingUnassigned(CrewSeat.class).filter(s -> !WorkerRules.isUnassigned(s.employee)),
                Joiners.equal(s -> s.blockId),
                Joiners.lessThan(s -> s.employee.id))
            .reward(HardMediumSoftScore.ONE_SOFT,
                (s1, s2) -> {
                    int edge = WorkerRules.computeAffinityEdge(s1.employee, s2.employee);
                    return AFFINITY_W * Math.max(0, edge);
                })
            .asConstraint("soft-affinity-reward");
    }

    Constraint softEncourageSkillVariety(ConstraintFactory f) {
        return f.forEachIncludingUnassigned(CrewSeat.class)
            .filter(s -> !WorkerRules.isUnassigned(s.employee))
            .groupBy(
                // key: [blockId, opId, skillLevel]
                s -> Arrays.asList(s.blockId, s.opId, WorkerRules.skill(s.employee, s.opId)),
                ConstraintCollectors.count()
            )
            .filter((key, count) -> count > 1)
            .penalize(HardMediumSoftScore.ONE_SOFT,
                (key, count) -> SKILL_DIVERSITY_W * (count * (count - 1) / 2)
            )
            .asConstraint("soft-encourage-skill-variety");
    }

    Constraint softBalanceBlockAvgSkill(ConstraintFactory f) {
        return f.forEachIncludingUnassigned(CrewSeat.class)
            .filter(s -> !WorkerRules.isUnassigned(s.employee))
            .groupBy(s -> Arrays.asList(s.blockId, s.opId),
                     ConstraintCollectors.sum(s -> WorkerRules.skill(s.employee, s.opId)),
                     ConstraintCollectors.count())
            .filter((key, sumLv, cnt) -> cnt > 0)
            .penalize(HardMediumSoftScore.ONE_SOFT,
                (key, sumLv, cnt) -> (int)(SKILL_AVG_W *
                    Math.abs((sumLv / Math.max(1.0, cnt)) - WorkerRules.avgSkill((String) key.get(1))) * 100))
            .asConstraint("soft-balance-block-avg-skill");
    }

    Constraint softBalanceTotalHours(ConstraintFactory f) {
        // Performance redesign: a seat's hours over its whole span equal
        // coveredDays * chosenHours, so the per-day DaySlot join and the
        // day flattening are unnecessary - one seat-to-block tuple suffices.
        return f.forEachIncludingUnassigned(CrewSeat.class)
            .filter(s -> !WorkerRules.isUnassigned(s.employee))
            .join(BlockDecision.class,
                Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
            .groupBy((CrewSeat s, BlockDecision b) -> s.employee,
                    ConstraintCollectors.sum((CrewSeat s, BlockDecision b) ->
                            seatCoveredDays(s, b).size() * b.chosenHours()))
            .penalize(HardMediumSoftScore.ONE_SOFT, (emp, dynTot) -> {
                long total = dynTot + emp.histFixedTotalHours;
                return (int) Math.abs(total - SolveContext.current().targetHoursPerEmp);
            })
            .asConstraint("soft-balance-total-hours");
    }
    Constraint softContinuousRegionStay(ConstraintFactory f) {
        return f.forEachIncludingUnassigned(CrewSeat.class)
            .filter(s -> !WorkerRules.isUnassigned(s.employee))
            .join(f.forEachIncludingUnassigned(CrewSeat.class)
                .filter(s -> !WorkerRules.isUnassigned(s.employee)),
                Joiners.equal(s -> s.employee.id),
                Joiners.lessThan(s -> s.blockId))
            .filter((s1, s2) -> {
                int r1 = s1.regionIdx;
                int r2 = s2.regionIdx;
                return r1 >= 0 && r2 >= 0 && r1 != r2;
            })
            .penalize(HardMediumSoftScore.ONE_SOFT,
                (s1, s2) -> REGION_TRANSITION_W)
            .asConstraint("soft-minimize-region-transitions");
    }

    Constraint softFabPreference(ConstraintFactory f) {
        return f.forEachIncludingUnassigned(CrewSeat.class)
            .join(f.forEach(BlockDecision.class),
                Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
            .filter((s, b) -> !WorkerRules.isUnassigned(s.employee))
            .reward(
                HardMediumSoftScore.ONE_SOFT,
                (s, b) -> {
                    String fabId      = s.factory;
                    String regionId   = s.regionId();
                    String customerId = SolveContext.current().calendars.customerOfFab(fabId);

                    int rLevel = WorkerRules.regionPref(s.employee, regionId);
                    int cLevel = WorkerRules.customerPref(s.employee, customerId);

                    // level 0/1 → 0 reward; >=2 → n(n-1)
                    int rScore = (rLevel <= 1) ? 0 : (rLevel * (rLevel - 1));
                    int cScore = (cLevel <= 1) ? 0 : (cLevel * (cLevel - 1));

                    return FAB_PREFERENCE_W * (rScore + cScore);
                }
            )
            .asConstraint("soft-fab-preference");
    }

    Constraint preferRegularOverSpot(ConstraintFactory f) {
        return f.forEachIncludingUnassigned(CrewSeat.class)
            .join(f.forEach(BlockDecision.class),
                Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
            .filter((s, b) -> !WorkerRules.isUnassigned(s.employee))
            .filter((s, b) -> !s.employee.isRegularForOperation(s.opId))
            .penalize(
                HardMediumSoftScore.ONE_SOFT,
                (s, b) -> REGULAR_SPOT_W
            )
            .asConstraint("prefer-regular-over-spot");
    }
    static int expPenalty(int diff) {
        // exponential growth but clamped to int
        // diff=0 => 0
        if (diff <= 0) return 0;
        long v = 1;
        for (int i = 0; i < diff; i++) {
            v *= 3;          // base 3 exponential (3,9,27,...)
            if (v > 1_000_000_000L) break;
        }
        return clampToInt(v);
    }
    Constraint recommendHeadcount(ConstraintFactory f) {
        var perBlock = f.forEach(BlockDecision.class)
            .filter(b -> b.recommendMinHeads > 0 || b.recommendMaxHeads > 0)
            .join(f.forEachIncludingUnassigned(CrewSeat.class),
                Joiners.equal((BlockDecision b) -> b.id, (CrewSeat s) -> s.blockId))
            .groupBy((b, s) -> b, ConstraintCollectors.toList((b, s) -> s));

        return perBlock
            .penalize(HardMediumSoftScore.ONE_MEDIUM, (b, seats) -> {
                int staffed = staffedCountForBlock(seats);

                int min = b.recommendMinHeads;
                int max = b.recommendMaxHeads;

                if (min <= 0 && max > 0) min = max;
                if (max <= 0 && min > 0) max = min;
                if (min <= 0 && max <= 0) return 0;
                if (max < min) max = min;

                int diff = 0;
                if (staffed < min) {
                    diff = min - staffed;
                } else if (staffed > max) {
                    diff = staffed - max;
                } else {
                    diff = 0;
                }

                if (diff == 0) return 0;

                int expo = expPenalty(diff);
                return clampToInt((long) RECOMMEND_HEADS_W * (long) expo);
            })
            .asConstraint("block-recommend-headcount-range");
    }

    Constraint[] all(ConstraintFactory f) {
        return new Constraint[] {
            preferSmallerHours(f),
            preferEarlierStart(f),
            softAffinityPenalty(f),
            softAffinityReward(f),
            softEncourageSkillVariety(f),
            softBalanceBlockAvgSkill(f),
            softBalanceTotalHours(f),
            softContinuousRegionStay(f),
            softFabPreference(f),
            preferRegularOverSpot(f),
            recommendHeadcount(f),
        };
    }
}

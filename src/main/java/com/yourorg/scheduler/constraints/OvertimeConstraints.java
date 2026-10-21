package com.yourorg.scheduler.constraints;

import java.util.*;

import ai.timefold.solver.core.api.score.HardMediumSoftScore;
import ai.timefold.solver.core.api.score.stream.Constraint;
import ai.timefold.solver.core.api.score.stream.ConstraintCollectors;
import ai.timefold.solver.core.api.score.stream.ConstraintFactory;
import ai.timefold.solver.core.api.score.stream.Joiners;

import com.yourorg.scheduler.calendar.SolveContext;
import com.yourorg.scheduler.model.*;
import static com.yourorg.scheduler.config.SolverConstants.BASE_HOURS_PER_DAY;
import static com.yourorg.scheduler.constraints.ConstraintSupport.seatCoveredDays;

/** Overtime limits per employee: annual and monthly (2 constraints). */
final class OvertimeConstraints {

    // One overtime contribution for a (seat, calendar-bucket) pair. The
    // overtime constraints pre-aggregate a block span by year (annual) or
    // year-month (monthly) before grouping, so each bucket is emitted once.
    static final class EmpBucketOt {
        final int empId; final String company; final int bucket; final int ot;
        EmpBucketOt(int empId, String company, int bucket, int ot) {
            this.empId = empId; this.company = company; this.bucket = bucket; this.ot = ot;
        }
    }
    Constraint annualOvertimeLimit(ConstraintFactory f) {
        // Performance redesign: dropped the DaySlot cartesian product. Each
        // seat's block span is expanded via the coveredWorkingDays shadow
        // variable and pre-aggregated by year inside map(); the year of a
        // dayId is resolved through DAY_TO_YEAR instead of a DaySlot join.
        var dynEmpYearOt = f.forEachIncludingUnassigned(CrewSeat.class)
            .filter(s -> !WorkerRules.isUnassigned(s.employee))
            .join(BlockDecision.class,
                Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
            .map((CrewSeat s, BlockDecision b) -> {
                // Pinned/warm-pinned overtime is loaded from
                // FIXED_ANNUAL_OT_BY_EMP_YEAR, so those seats contribute 0
                // here but still create their (emp, company, year) group key.
                int otPerDay = (s.pinnedFixed || s.warmPinned)
                        ? 0
                        : Math.max(0, b.chosenHours() - BASE_HOURS_PER_DAY);
                int empId = s.employee.id;
                String comp = WorkerRules.company(s.employee);
                Map<Integer,Integer> byYear = new HashMap<>();
                for (int dayId : seatCoveredDays(s, b)) {
                    Integer year = SolveContext.current().dayToYear.get(dayId);
                    if (year == null) continue;
                    byYear.merge(year, otPerDay, Integer::sum);
                }
                List<EmpBucketOt> rows = new ArrayList<>(byYear.size());
                for (Map.Entry<Integer,Integer> e : byYear.entrySet()) {
                    rows.add(new EmpBucketOt(empId, comp, e.getKey(), e.getValue()));
                }
                return rows;
            })
            .flattenLast(rows -> rows)
            .groupBy(x -> Arrays.asList(x.empId, x.company, x.bucket),
                     ConstraintCollectors.sum(x -> x.ot));

        return dynEmpYearOt
            .filter((key, dynOtHours) -> {
                int empId      = (int) key.get(0);
                String companyId = (String) key.get(1);
                int year       = (int) key.get(2);

                if (companyId == null || companyId.isBlank()) return false;
                int limit = SolveContext.current().calendars.annualOtLimit(companyId);
                if (limit == Integer.MAX_VALUE) return false;

                int fixedOt = SolveContext.current().fixedAnnualOtByEmpYear
                        .getOrDefault(empId, Collections.emptyMap())
                        .getOrDefault(year, 0);

                long totalOt = dynOtHours + fixedOt;
                return totalOt > limit;
            })
            .penalize(
                HardMediumSoftScore.ONE_HARD,
                (key, dynOtHours) -> {
                    int empId      = (int) key.get(0);
                    String companyId = (String) key.get(1);
                    int year       = (int) key.get(2);

                    int limit = SolveContext.current().calendars.annualOtLimit(companyId);
                    int fixedOt = SolveContext.current().fixedAnnualOtByEmpYear
                            .getOrDefault(empId, Collections.emptyMap())
                            .getOrDefault(year, 0);

                    long totalOt = dynOtHours + fixedOt;
                    return Math.max(1, totalOt - limit);
                }
            )
            .asConstraint("emp-annual-overtime-limit");
    }

    Constraint monthlyOvertimeLimit(ConstraintFactory f) {
        // Performance redesign: dropped the DaySlot cartesian product. Each
        // seat's block span is expanded via the coveredWorkingDays shadow
        // variable and pre-aggregated by year-month (ym = year*100+month)
        // through DAY_TO_YM instead of a DaySlot join. The group key now
        // carries ym directly, so the filter/penalize read key.get(2)=ym.
        var dynEmpMonthOt = f.forEachIncludingUnassigned(CrewSeat.class)
            .filter(s -> !WorkerRules.isUnassigned(s.employee))
            .join(BlockDecision.class,
                Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
            .map((CrewSeat s, BlockDecision b) -> {
                // Pinned overtime is loaded from FIXED_MONTHLY_OT_BY_EMP_YM,
                // so pinned seats contribute 0 here but still create the key.
                int otPerDay = s.pinnedFixed
                        ? 0
                        : Math.max(0, b.chosenHours() - BASE_HOURS_PER_DAY);
                int empId = s.employee.id;
                String comp = WorkerRules.company(s.employee);
                Map<Integer,Integer> byYm = new HashMap<>();
                for (int dayId : seatCoveredDays(s, b)) {
                    Integer ym = SolveContext.current().dayToYm.get(dayId);
                    if (ym == null) continue;
                    byYm.merge(ym, otPerDay, Integer::sum);
                }
                List<EmpBucketOt> rows = new ArrayList<>(byYm.size());
                for (Map.Entry<Integer,Integer> e : byYm.entrySet()) {
                    rows.add(new EmpBucketOt(empId, comp, e.getKey(), e.getValue()));
                }
                return rows;
            })
            .flattenLast(rows -> rows)
            .groupBy(x -> Arrays.asList(x.empId, x.company, x.bucket),
                     ConstraintCollectors.sum(x -> x.ot));

        return dynEmpMonthOt
            .filter((key, dynOtHours) -> {
                int empId      = (int) key.get(0);
                String companyId = (String) key.get(1);
                int ym         = (int) key.get(2);

                if (companyId == null || companyId.isBlank()) return false;
                int limit = SolveContext.current().calendars.monthlyOtLimit(companyId);
                if (limit == Integer.MAX_VALUE) return false;

                int fixedOt = SolveContext.current().fixedMonthlyOtByEmpYm
                        .getOrDefault(empId, Collections.emptyMap())
                        .getOrDefault(ym, 0);

                long totalOt = dynOtHours + fixedOt;
                return totalOt > limit;
            })
            .penalize(
                HardMediumSoftScore.ONE_HARD,
                (key, dynOtHours) -> {
                    int empId      = (int) key.get(0);
                    String companyId = (String) key.get(1);
                    int ym         = (int) key.get(2);

                    int limit = SolveContext.current().calendars.monthlyOtLimit(companyId);
                    int fixedOt = SolveContext.current().fixedMonthlyOtByEmpYm
                            .getOrDefault(empId, Collections.emptyMap())
                            .getOrDefault(ym, 0);

                    long totalOt = dynOtHours + fixedOt;
                    return Math.max(1, totalOt - limit);
                }
            )
            .asConstraint("emp-monthly-overtime-limit");
    }

    Constraint[] all(ConstraintFactory f) {
        return new Constraint[] {
            annualOvertimeLimit(f),
            monthlyOvertimeLimit(f),
        };
    }
}

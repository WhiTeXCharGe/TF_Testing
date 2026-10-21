package com.yourorg.scheduler.model;

import ai.timefold.solver.core.api.domain.solution.PlanningEntityCollectionProperty;
import ai.timefold.solver.core.api.domain.solution.PlanningScore;
import ai.timefold.solver.core.api.domain.solution.PlanningSolution;
import ai.timefold.solver.core.api.domain.solution.ProblemFactCollectionProperty;
import ai.timefold.solver.core.api.score.HardMediumSoftScore;

import java.util.List;

@PlanningSolution
public class SinglePassPlan {
    @ProblemFactCollectionProperty
    public List<DaySlot> days;

    @ProblemFactCollectionProperty
    public List<EmployeeFact> employees;

    @PlanningEntityCollectionProperty
    public List<BlockDecision> blocks;

    @PlanningEntityCollectionProperty
    public List<CrewSeat> seats;

    @PlanningScore
    private HardMediumSoftScore score;
    public HardMediumSoftScore getScore() { return score; }
    public void setScore(HardMediumSoftScore s) { this.score = s; }
    public SinglePassPlan() {}
}

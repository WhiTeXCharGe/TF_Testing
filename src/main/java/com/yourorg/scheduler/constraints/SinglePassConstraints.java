package com.yourorg.scheduler.constraints;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import ai.timefold.solver.core.api.score.stream.Constraint;
import ai.timefold.solver.core.api.score.stream.ConstraintFactory;
import ai.timefold.solver.core.api.score.stream.ConstraintProvider;

/**
 * Registers the 23 scoring rules of the single-pass schedule. The rules live in the group classes
 * (block, seat, region, overtime, soft); this class only fixes the order in which they are registered.
 */
public class SinglePassConstraints implements ConstraintProvider {

    private final BlockConstraints block = new BlockConstraints();
    private final SeatConstraints seat = new SeatConstraints();
    private final RegionConstraints region = new RegionConstraints();
    private final OvertimeConstraints overtime = new OvertimeConstraints();
    private final SoftConstraints soft = new SoftConstraints();

    @Override public Constraint[] defineConstraints(ConstraintFactory f) {
        List<Constraint> all = new ArrayList<>();
        Collections.addAll(all, block.all(f));     // block feasibility, production & capacity
        Collections.addAll(all, seat.all(f));      // seat-level hard rules: one task a day, daily cap
        Collections.addAll(all, region.all(f));    // region transit gap, stay maximum, annual stay
        Collections.addAll(all, overtime.all(f));  // annual and monthly overtime limits
        Collections.addAll(all, soft.all(f));      // softs
        return all.toArray(new Constraint[0]);
    }
}

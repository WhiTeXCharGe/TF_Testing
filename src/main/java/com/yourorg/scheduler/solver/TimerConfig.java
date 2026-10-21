package com.yourorg.scheduler.solver;

/**
 * Time budgets and stop rules of the two-stage solve.
 *
 * @param stage1Minutes            stage 1 time limit in minutes
 * @param stage1UnimprovedSeconds  stage 1 stops after this many seconds without a new best score
 * @param stage1StopWhenFeasible   stage 1 also stops as soon as the hard score reaches 0 ("0hard/*medium/*soft")
 * @param stage2Enabled            false skips the polish stage entirely (the stage 1 result is the result)
 * @param stage2Minutes            stage 2 time limit in minutes
 * @param stage2UnimprovedSeconds  stage 2 stops after this many seconds without a new best score
 * @param seed                     random seed for both stages, or null for none
 */
public record TimerConfig(int stage1Minutes,
                          int stage1UnimprovedSeconds,
                          boolean stage1StopWhenFeasible,
                          boolean stage2Enabled,
                          int stage2Minutes,
                          int stage2UnimprovedSeconds,
                          Long seed) {

    /** The production defaults: stage 1 up to 300 min / 600 s unimproved, stage 2 up to 60 min / 60 s unimproved. */
    public static TimerConfig defaults() {
        return new TimerConfig(300, 600, true, true, 60, 60, null);
    }
}

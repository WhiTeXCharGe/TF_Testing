package com.yourorg.scheduler.model;

import java.util.Iterator;
import java.util.Random;

import ai.timefold.solver.core.api.domain.valuerange.ValueRange;
import ai.timefold.solver.core.impl.domain.valuerange.AbstractCountableValueRange;
import ai.timefold.solver.core.impl.domain.valuerange.sort.SortableValueRange;
import ai.timefold.solver.core.impl.domain.valuerange.sort.ValueRangeSorter;

/**
 * A value range that behaves exactly like {@code base} (size, order, iterators, sorted copies) except for ONE thing:
 * {@link #contains(Object)} also accepts the single instance {@code tolerated}, compared by identity.
 *
 * <p>Why: Timefold 1.35 validates once, when a working solution is set, that every initial planning value is
 * {@code contains}-ed by its entity's range (Timefold 1.27 did not). In this model the initial value can legitimately be
 * outside the range (UNASSIGNED on a manager seat, the real employee of a pinned seat, a warm-start employee
 * that fails the candidate gates, a warm-start phase-1 startDay after windowStart). Adding that value to the range would make the
 * search select it (e.g. empty manager seats). Only {@code contains} is used for validation, while move selection
 * iterates/samples the range, so the tolerated value stays invisible to the search. {@code tolerated} is
 * the entity's value at the time the range is built, so it is the initial value.
 */
public final class ValidationTolerantRange<T> extends AbstractCountableValueRange<T> {

    private final ValueRange<T> base;
    private final T tolerated;

    public ValidationTolerantRange(ValueRange<T> base, T tolerated) {
        this.base = base;
        this.tolerated = tolerated;
    }

    @Override
    public long getSize() {
        return base.getSize();
    }

    @Override
    public T get(long index) {
        return base.get(index);
    }

    @Override
    public Iterator<T> createOriginalIterator() {
        return base.createOriginalIterator();
    }

    @Override
    public Iterator<T> createRandomIterator(Random workingRandom) {
        return base.createRandomIterator(workingRandom);
    }

    @Override
    public boolean contains(T value) {
        return base.contains(value) || (tolerated != null && value == tolerated);
    }

    /** Sorted copies are the plain base range: the tolerated value is not carried over. */
    @Override
    @SuppressWarnings("unchecked")
    public ValueRange<T> sort(ValueRangeSorter<T> sorter) {
        return ((SortableValueRange<T>) base).sort(sorter);
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof ValidationTolerantRange<?> that && base.equals(that.base) && tolerated == that.tolerated;
    }

    @Override
    public int hashCode() {
        return 31 * base.hashCode() + System.identityHashCode(tolerated);
    }

    @Override
    public String toString() {
        return base.toString();
    }
}

# Decoder5 Major Changes

  

This note summarizes the main changes from **Decoder4_1** to **Decoder5**.  

The purpose is to explain the logic change at a design level, not at source-code level.

  

---

  

## 1. Main concept change

  

Decoder4_1 mainly used the planned phase timeline from **新規製番リスト** and then shifted/rescaled it to match the actual span found in **SU_Others**.  

Decoder5 changes this idea for tool work.

  

In Decoder5, the workflow no longer uses **phase 1**.  

The tool workflow now starts from **p2**, and the actual phase border is no longer decided only by ratio-based shifting.  

Instead, the decoder now uses the real worker assignment pattern in **SU_Others**, especially the appearance of workers with **QC role**, to decide when phase 3 should begin.

  

So the design changed from:

- "planned phases are shifted onto actual timeline"

  

to:

- "actual work timeline starts from p2, then phase 3 begins based on QC-related evidence and remaining time rules"

  

---

  

## 2. Removal of phase 1

  

A major structural change in Decoder5 is that **phase 1 is removed** from normal `wf_tool` task generation.

  

That means:

- output task phases start from **p2**

- normal workflow output is now only:

  - `p2`

  - `p3`

  - `p4`

- phase-related planned data from 新規製番リスト is now interpreted mainly for the later phases, not for generating a full p1-p4 flow

  

This is not just a display change. It changes how the decoder builds:

- shifted phase windows

- task list in `Schedule.yaml`

- worker skill inference connected to phase participation

  

---

  

## 3. New phase split logic

  

### Decoder4_1 style

Decoder4_1 used a ratio-based shift method:

- read planned phase dates from 新規製番リスト

- calculate planned duration ratio of each phase

- find the actual working span in SU_Others

- split that actual span according to planned ratios

  

### Decoder5 style

Decoder5 changes this to a more rule-driven phase split.

  

The new idea is:

- the first real worked day of the module becomes the start of **p2**

- **p3** does not start immediately from planned ratio

- instead, the decoder looks for the first participation of a worker who has **QC role**

- that QC appearance becomes the signal used to determine the transition toward phase 3

  

However, even if a QC worker appears early, Decoder5 still keeps a special rule:

- the remaining total period for **phase 3 + phase 4** is limited to **4 weeks / 28 days** by default

- this value is treated as a parameter, so it can be changed later

- phase 2 must still keep at least **1 day minimum**

  

So if QC appears too early and there are still too many days left, Decoder5 does not immediately start p3.  

Instead, it delays the p3 start until the remaining timeline is reduced to the allowed `p3+p4` tail length.

  

This creates a prototype rule such as:

- start from p2

- detect QC-related participation

- keep p2 long enough

- when remaining tail becomes small enough, start p3

- then split p3 and p4 inside the remaining tail

  

---

  

## 4. New rule for phase 3 and phase 4 tail length

  

Decoder5 introduces a new constraint-like decoding rule:

  

- **phase 3 + phase 4 combined length** should be at most **28 days** by default

- this value is meant to be configurable

- if no limit is used in the future, the cap can be disabled

  

This rule is important because it changes the meaning of the timeline:

- Decoder4_1 mainly followed shifted ratio output

- Decoder5 gives priority to the idea that the later part of the job should stay inside a fixed maximum tail window

  

Then inside that remaining tail:

- `p3` and `p4` are split based on the planned relationship from 新規製番リスト

- but the total tail itself is bounded first

  

So the order of decision becomes:

1. determine actual working timeline

2. choose p3 start based on QC evidence and remaining days rule

3. keep at least one day for p2

4. split the remaining tail into p3 and p4

  

---

  

## 5. Skill-map logic is changed

  

Another major change is the worker skill assignment logic.

  

### Old idea

Decoder4_1 already had the old skill map source from the skill summary file and also used assignment evidence in SU_Others.

  

### New idea in Decoder5

Decoder5 now uses a **new role column in SU_Others** for skill inference.  

This is the same column that was already checked for the text **責** to assign manager role.

  

Now that same source is also used to detect **QC** workers.

  

### New QC-based skill rules

If a worker has QC role:

- set minimum skill for **p3 = 1**

- set minimum skill for **p4 = 1**

  

In addition:

- if that QC worker has at least one real assignment before the module enters phase 3

- and that assignment is not a dummy category such as other-tool or personal business

- then set minimum skill for **p2 = 1** as well

  

This means QC role now affects not only the later phases, but also earlier participation if the worker was already involved before the p3 boundary.

  

---

  

## 6. Skill merge rule is now explicit

  

Decoder5 uses the following merge logic for worker skills:

  

- take the old skill level from the original skill map source

- take the new skill level inferred from Decoder5 assignment/QC logic

- use the **maximum** of both values

  

So the final skill is not a replacement-only method.  

It is a merge rule:

  

**final skill = max(old skill, newly inferred skill)**

  

This is important because it preserves existing skill information while allowing Decoder5 to upgrade the skill map using actual SU_Others evidence.

  

---

  

## 7. SU_Others is still the main actual evidence source

  

Decoder5 still keeps the idea that **SU_Others** is the main source for actual work evidence.

  

It still uses SU_Others to:

- detect actual worked days

- detect actual worker participation

- infer skills from actual behavior

- decide real assignment timing

- separate dummy/non-real tasks from normal tool work

  

So even though phase shifting logic changed, the decoder still depends strongly on SU_Others as the real execution-side data source.

  

---

  

## 8. Old cut / dummy handling is still kept

  

Decoder5 does **not** throw away the earlier protection logic.

  

The older cut behavior is still intended to remain, including cases such as:

- abnormal cells

- dummy-like cells

- too few real assignment days

- labels that should stay as non-tool or non-usable task evidence

  

In other words:

- the new phase logic changed

- the new skill logic changed

- but the previous filtering/cut philosophy is still kept

  

This is important because Decoder5 is not a full rewrite of the cleaning layer.  

It is mainly a redesign of **phase interpretation** and **skill inference** on top of the existing decoder structure.

  

---

  

## 9. Task generation philosophy changed

  

Decoder4_1 generated task phases by shifting the planned structure.  

Decoder5 generates tasks based on what remains meaningful after the new cut logic and the new actual-phase interpretation.

  

So the task output should now be understood as:

- based on the remaining valid information from 新規製番リスト

- filtered by the previous cut rules

- then rebuilt with the new p2/p3/p4 interpretation

- using actual SU_Others participation to decide the later phase border

  

This means the task structure is now more behavior-driven than pure plan-driven.

  

---

  

## 10. Why Decoder5 is a prototype-style version

  

Decoder5 is still a prototype-style decoder because the timeline rule is intentionally unusual.

  

The design currently follows the requested experimental logic first:

- no phase 1

- start from p2

- use QC participation as phase 3 signal

- keep p3+p4 tail within a capped duration

- give p2 at least one day

  

This is not a standard phase-planning model yet.  

It is a rule-based prototype created to surface mistakes and confirm whether the real data behavior matches the expected production flow.

  

So Decoder5 should be seen as:

- a major logic update from Decoder4_1

- focused on new experimental phase interpretation

- designed to make later validation and correction easier

  

---

  

## 11. Practical summary of the difference

  

In simple terms, Decoder4_1 and Decoder5 differ like this:

  

### Decoder4_1

- uses p1-p4

- shifts planned phase windows onto actual timeline

- uses earlier skill logic plus assignment inference

- phase borders mainly follow shifted ratios

  

### Decoder5

- removes p1

- starts workflow from p2

- uses QC-related participation to decide when p3 begins

- keeps p3+p4 inside a limited tail window

- keeps at least one day for p2

- uses SU_Others role column for new QC-based skill inference

- merges new inferred skill with old skill map by max rule

- keeps previous dummy/cut handling behavior

  

---

  

## 12. Short conclusion

  

Decoder5 is a major behavioral change from Decoder4_1.  

The biggest update is that the decoder no longer treats phase generation as a simple shifted version of the original planned schedule.  

Instead, it now interprets the actual work timeline more directly, removes phase 1, starts from phase 2, and uses QC-related participation plus remaining-time rules to decide the beginning of later phases.  

At the same time, worker skill generation is strengthened by reading QC information from SU_Others and merging that inferred result with the existing skill map.
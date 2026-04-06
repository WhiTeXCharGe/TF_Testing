# Draft summary: Regular / Spot concept in current scheduler

## 1. What “regular / spot” means in the current model

In the current model, assignment already depends on a few layers:

- task/phase skill match
- manager requirement
- personal unavailable dates
- region / customer preference
- region transit / stay rules
- workload and headcount balance

So **regular / spot** would not replace the whole assignment logic.  
It would act more like an extra business meaning on top of the current worker model.

A simple meaning could be:

- **regular** = worker is a normal / preferred member for that type of work
- **spot** = worker is still possible, but more like backup / overflow / temporary support

So in schedule behavior, regular / spot can help answer:

- who should be assigned first when several workers are all possible
- who is considered the “core team” for a block
- when the solver should use backup people
- how to express “preferred pool” without making everyone else impossible

That matches your current model well, because your current solver already separates:

- **cannot assign** cases  
- **can assign but less preferred** cases

For example, right now skill and manager are direct feasibility checks, while region/customer preference is already partly used as preference logic. Also, preference level `0` for region or customer is already treated as “not a candidate” in the candidate build, while positive levels are used later as a soft preference reward. :contentReference[oaicite:0]{index=0}

Timefold also separates this general idea into:
- **hard constraints** for things that must not happen,
- **soft constraints** for things that should be preferred,
and hard constraints always outweigh soft ones. :contentReference[oaicite:1]{index=1}

---

## 2. How regular / spot is different from current preference

### Current preference meaning
In the current model:

- `regionPreference` and `customerPreference` already exist on employee facts
- region/customer are derived from fab
- level `0` means worker is blocked from candidacy for that fab’s region/customer
- level `2+` gives more soft reward than level `1`

So current preference is closer to:

- **0** = cannot go
- **1** = can go, neutral
- **2,3,...** = can go and preferred

This means your current preference is already doing two jobs:

1. eligibility gate at level 0  
2. preference strength at level 1+ 

### Regular / spot meaning
Regular / spot is a bit different:

- it is more about **worker role in the staffing pool**
- not only about liking a region/customer
- not only about legality

Example:
- someone may be allowed to go to a fab, but still be only a **spot** worker there
- someone may have neutral region preference, but still be **regular** for a certain phase because of experience or team role
- someone may have high customer preference but still not be part of the normal core crew for that work

So regular / spot should usually be read as:

- **staffing category**
- not only location preference

---

## 3. Can regular / spot replace region/customer preference?

## Short answer

Usually **no, not fully**.

## Why not

Because current preference already carries a stronger meaning:

- `0` means “do not send”
- positive values mean “okay / preferred”

That is broader than regular / spot.

If you replace everything with only regular / spot, you lose the difference between:

- impossible
- possible but neutral
- strongly preferred

So regular / spot is better as an **additional layer**, not a total replacement.

### Good way to think about it

- **skill / manager / availability / pref=0**  
  = “can this person be assigned at all?”

- **region/customer preference level**  
  = “how suitable is this location/customer for this person?”

- **regular / spot**  
  = “is this person part of the normal crew or backup crew for this work?”

That separation is cleaner.

---

## 4. When regular / spot could partly replace preference

There are some cases where it could replace part of current preference logic.

### Case A: very simple business rule
If business only says:

- these workers are core members
- these workers are backup members

and there is no real need for multi-level preference, then regular / spot might be enough.

### Case B: current preference values are being used only as rough labels
If current region/customer preference is not really tuned and mostly used as:

- 0 = no
- 1 = okay
- 2 = preferred

then regular / spot could absorb part of the “preferred” meaning.

But even then, I would still keep:

- `0` as hard no-go
- regular / spot as staff pool meaning

because those 2 ideas are different.

---

## 5. Main pattern choices

# Pattern 1: Global worker type

### Meaning
Each worker is always either:

- regular
- or spot

for every task, fab, region, customer.

### How to use it
Use when business meaning is simple, like:

- senior/core employees = regular
- temporary/support employees = spot

### Benefits
- easiest to understand
- easiest to prepare
- good for first testing
- good if regular/spot is really a company-wide role

### Disadvantages
- too rough
- does not reflect that a worker may be core for one task but backup for another
- may not match real fab/project behavior

### Best use case
A first business draft or a first experiment.

---

# Pattern 2: Task or phase-based type

### Meaning
Worker type depends on task.

Example:
- regular for p1/p2
- spot for p3/p4

### How to use it
Use when “core team” depends more on work type than location.

This works well when:
- certificate / experience differs by phase
- same person is trusted for installation but not utility
- some workers are main members only in certain operations

### Benefits
- practical
- closer to real work than global type
- still not too heavy
- matches your current `p1/p2/p3/p4` structure well 

### Disadvantages
- still ignores fab-specific specialization
- may be too rough if customer/fab matters a lot

### Best use case
Probably the best middle version if you want one realistic but manageable design.

---

# Pattern 3: Fab + task type

### Meaning
Worker is regular only for some fab/task combinations.

Example:
- regular for `f3+p1`
- regular for `f10+p4`
- spot elsewhere

### How to use it
Use when real business depends on:

- site familiarity
- customer method differences
- local certification/site approval
- known core teams per fab

### Benefits
- closest to reality if work is fab-driven
- good for rare “core member only for this site” cases
- fits special customer/site skill pools

### Disadvantages
- heavy to maintain
- many fabs means a lot of data
- easy to miss definitions
- can become messy if done too detailed

### Best use case
When fab/site identity is truly important.

A practical simplification is:
- record only **regular** cases
- treat all not-listed cases as spot

That matches your earlier idea and keeps data smaller.

---

# Pattern 4: Region + task type

### Meaning
Worker is regular by region and task, not by exact fab.

### How to use it
Use when fab count is high, but work style is mostly regional.

Example:
- regular for Japan p1/p2
- spot for Taiwan p1/p2

### Benefits
- smaller data than fab+task
- still captures geography-based core teams
- works well with your existing region logic and transit/stay rules 

### Disadvantages
- not enough if some fabs in same region behave very differently

### Best use case
A good middle ground between task-based and fab-based modeling.

---

# Pattern 5: Customer + task type

### Meaning
Worker is regular based on customer and task.

### How to use it
Use when customer-specific methods matter more than fab identity.

Example:
- regular for Customer A p1/p2
- spot for Customer B p1/p2

### Benefits
- useful when one customer has multiple similar fabs
- matches commercial / relationship-driven staffing

### Disadvantages
- less accurate if actual difference is fab-level, not customer-level

### Best use case
When customer know-how is the real reason for staffing preference.

---

## 6. How regular / spot can be interpreted from current business data

Regular / spot can represent several business meanings, even without changing the whole model.

### A. Certificate-based meaning
A worker may be:

- **regular** because they hold the right certificate and are part of the normal certified pool
- **spot** because they can support, but are not the usual certified choice

This is useful when certificate is not only “can/cannot”, but also “main qualified member vs fallback member”.

### B. Experience-based meaning
A worker may have enough skill to be assignable, but:

- regular = normal experienced member for this type of task
- spot = can help, but not first-line

### C. Region familiarity
A worker may be:

- regular for Japan jobs
- spot for overseas jobs

This is not exactly the same as region preference.
Preference says whether location is acceptable or liked.
Regular/spot says whether the person belongs to the normal staffing pool there.

### D. Customer familiarity
A worker may be:

- regular for a certain customer because of repeated work and communication fit
- spot for others

Again, this is different from customer preference level.
It is more like customer-specific crew membership.

### E. Fab familiarity
A worker may be regular at one fab because they know:

- local rules
- local process
- local people
- startup habits

and only spot at other fabs.

### F. Team composition meaning
Regular / spot can also mean team structure:

- regular = core team
- spot = overflow / shortage filler

This can be useful when you want the solution to first build a stable team, then use backup workers only when needed.

---

## 7. Potential scenarios where regular / spot helps a lot

### Scenario 1: skill is enough, but business still has a normal crew
Current skill map may say several people can do p3.  
But in reality only some are the normal p3 crew.  
Regular / spot can express that difference.

### Scenario 2: certificate is shared, but site trust is different
Two workers may both be legally assignable.  
But one is the usual certified member for that fab/customer, and the other is only emergency support.

### Scenario 3: preference is too general
Region/customer preference may say a worker is okay for Taiwan.  
But that does not tell whether they are part of the main AP6A crew or just a possible helper.  
Regular / spot can cover that gap.

### Scenario 4: want to preserve core members for important phases
For example:
- p1/p2 should mostly use the regular team
- spot fills remaining shortage

That is a staffing policy idea, not only a travel preference idea.

### Scenario 5: want clearer business explanation
“Assigned because he is regular for this fab/task” is easier for business users to understand than only saying “score was better due to preference”.

---

## 8. Potential disadvantages / risks

### A. Too much overlap with existing preference
If regular / spot is not clearly defined, it may duplicate:

- region preference
- customer preference
- fab suitability
- certificate skill

Then data becomes confusing.

### B. Data ownership becomes unclear
Who decides regular vs spot?

- planning team?
- engineering lead?
- certificate owner?
- site owner?

This needs a clear business rule.

### C. Over-modeling
If you define regular/spot by too many dimensions at once, maintenance will become hard.

### D. Hidden hard meaning
Sometimes business says “spot”, but actually means “do not send unless emergency”.  
That is a different rule from ordinary backup preference.

---

## 9. Suggested business interpretation

A clean interpretation would be:

### Keep current preference for location acceptability
Use region/customer preference for:

- no-go
- neutral
- preferred destination

### Use regular / spot for staffing pool identity
Use regular / spot for:

- core member
- backup member

### Use skills/certificates for capability
Use skill/certification for:

- can do / cannot do
- level of technical ability

That gives a nice separation:

- **skill/certificate** = capability
- **preference** = suitability of place/customer
- **regular/spot** = crew role

This separation matches your current model structure better than trying to force one field to do everything. Your current code already treats skill, manager, availability, and preference-0 as gating conditions, while region/customer preference also contributes to soft reward. 

---

## 10. Rough recommendation

### Best first concept
Do not treat regular / spot as a replacement for current preference.

Better to treat it as a new meaning:

- who is the normal crew
- who is fallback crew

### Best pattern to start with
If you want a practical first business version:

- task-based regular/spot  
or
- region+task regular/spot

These usually give good realism without the heavy maintenance of fab+task.

### Best detailed future version
If fab identity is really important, use fab+task, but likely only record regular cases.

---

## 11. Final draft conclusion

Regular / spot is best understood as a **staffing-role concept**, not only a travel/location preference.

It can express:
- core vs backup members
- certificate holders vs support holders
- normal crew vs overflow pool
- fab/customer familiar workers vs general support workers

In the current model, it should usually **sit beside** existing region/customer preference, not replace it completely.

A good separation is:

- preference = where the worker can or likes to go
- regular/spot = whether the worker is part of the normal team for that work
- skill/certificate = whether the worker is capable to do it

That makes the schedule explanation much clearer and gives more room for real business rules later.

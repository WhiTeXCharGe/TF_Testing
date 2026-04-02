# Decoder5 p3/p4 Comparison Notes

## Purpose
This note summarizes the observed characteristics, advantages, disadvantages, and abnormal patterns from two Decoder5 result logs:

- **Version A:** non-limit p3/p4
- **Version B:** 28-day limit for p3+p4

The goal is not to compare every module, but to extract the major tendencies and a few representative examples.

---

## 1. Non-limit p3/p4

### Main behavior
In the non-limit version, once phase 3 starts, the remaining worked days can continue to flow into phase 3 and phase 4 without a hard upper bound. In practice, this means that if the QC trigger happens early, p3 and p4 can occupy a very large portion of the compressed worked timeline.

### Advantage
- It follows the actual SU_Others worked flow more naturally when a long tail really exists.
- It preserves long p3/p4 spans instead of forcing them into a short standardized tail.
- It can be closer to the real field history when QC participation starts early and the module genuinely continues in later-stage work for a long time.

### Disadvantage
- It can create **very small p2** and **very large p3/p4**, which may look operationally unbalanced.
- It is more likely to drift away from the rough structure implied by 譁ｰ隕剰｣ｽ逡ｪ繝ｪ繧ｹ繝�.
- When QC joins early, phase 3 may start too early and the remaining timeline becomes dominated by p3/p4.

### Abnormal / suspicious patterns seen in this version

#### A. p2 becomes too short compared with the whole worked timeline
A strong example is **530900102A**:

- planned: p2=21d, p3=15d, p4=12d
- shifted result (non-limit):
  - p2 = **1d**
  - p3 = **40d**
  - p4 = **39d**
  - worked total = **80d**

This is a strong abnormal sign because the structure becomes almost 窶彗ll late phases,窶� even though the planned structure is much more balanced.

#### B. p3 and p4 become disproportionately long
Example **530N03002A**:

- planned: p2=15d, p3=15d, p4=13d
- shifted result (non-limit):
  - p2 = **14d**
  - p3 = **27d**
  - p4 = **27d**
  - worked total = **68d**

This is not necessarily wrong, but it is clearly more stretched than the original plan and can be read as a late-phase expansion.

#### C. early QC trigger can pull too much work into p3/p4
Example **530900112A**:

- planned: p2=24d, p3=14d, p4=30d
- shifted result (non-limit):
  - p2 = **22d**
  - p3 = **25d**
  - p4 = **26d**
  - worked total = **73d**

Compared with the plan, the shifted result becomes much more even across p2/p3/p4, even though the original plan expected a heavier p4.

### Overall evaluation of non-limit p3/p4
This version is better when the goal is to respect long real execution tails. However, it also produces the most obvious 窶徭hape abnormalities,窶� especially:

- p2 collapsing to a very short period
- p3/p4 becoming overly dominant
- stronger separation from the planned balance in 譁ｰ隕剰｣ｽ逡ｪ繝ｪ繧ｹ繝�

---

## 2. 28-day limit p3+p4

### Main behavior
In this version, p3 and p4 together are limited to **28 worked days** after compression. This effectively pushes more of the earlier worked timeline into p2, while standardizing the tail into a smaller late-phase block.

### Advantage
- It suppresses extreme late-phase stretching.
- It makes p3/p4 easier to read and easier to compare across modules.
- It avoids cases where almost the entire module turns into p3/p4 after one early QC trigger.
- It reduces 窶徭hape explosion窶� in the shifted result.

### Disadvantage
- It can make **p2 too large**, especially when the actual late-stage work was truly long.
- It may hide real long-tail behavior by compressing it into a fixed tail budget.
- It can move the visible start of p3 later than the QC join date suggests.

### Abnormal / suspicious patterns seen in this version

#### A. p2 can become unnaturally long because p3+p4 is forced into 28 worked days
Example **530900102A**:

- shifted result (28-day limit):
  - p2 = **52d**
  - p3 = **14d**
  - p4 = **14d**
  - worked total = **80d**

This is much cleaner than the non-limit result, but it creates the opposite concern: p2 now occupies most of the module.

#### B. late phases become standardized, possibly too standardized
Example **530900112A**:

- shifted result (28-day limit):
  - p2 = **45d**
  - p3 = **14d**
  - p4 = **14d**
  - worked total = **73d**

This is easy to read and stable, but it may be too rigid if the real operation actually had a longer p3/p4 tail.

#### C. plan-vs-shift balance can become distorted in the opposite direction
Example **530N02824A**:

- planned: p2=20d, p3=18d, p4=24d
- shifted result (28-day limit):
  - p2 = **18d**
  - p3 = **14d**
  - p4 = **14d**
  - worked total = **46d**

Compared with the non-limit result (7d / 19d / 20d), this version looks more controlled, but it also flattens the intended difference between p3 and p4.

### Important note about compressed worked days
This version still works on the **compressed worked-day timeline**, not raw calendar span. That means calendar blanks are stitched out before splitting phases. Because of that, apparent calendar gaps can still appear in the final displayed windows.

Example **530L00936A**:

- shifted result (28-day limit):
  - p2 = 2025/01/23 - 2025/02/13
  - p3 = 2025/02/14 - 2025/02/27
  - p4 = 2025/03/07 - 2025/03/20

This does not mean the algorithm failed. It means the split was done on the compressed worked-day list, and the raw calendar had blanks in between.

### Overall evaluation of 28-day limit p3+p4
This version is better when the goal is stability and interpretability. It controls the most extreme p3/p4 growth, but it also introduces a new possible abnormality:

- p2 becoming too dominant
- p3/p4 becoming too uniform or too standardized
- loss of genuine long-tail behavior

---

## 3. Comparison: non-limit vs 28-day limit

### High-level difference
The biggest difference is not in the early cut rules. The biggest difference is in the **shifted phase shape** after phase 3 starts.

- **Non-limit version:** more faithful to long actual tails, but more likely to create extreme p3/p4 sizes.
- **28-day limit version:** more controlled and easier to read, but more likely to over-expand p2.

### What seems unchanged between the two logs
Many abnormalities are effectively the same in both logs because they happen **before** the p3/p4 cap matters. These include:

- blank / N/A phase dates in 譁ｰ隕剰｣ｽ逡ｪ繝ｪ繧ｹ繝�
- wrong p2<=p3<=p4<=p4end order
- phase end < start after applying next-start-1
- planned-window far cut
- unique SU_Others work days < 4
- per-worker single-day outlier cut

Representative examples that appear the same style in both logs:

- **830300199A** 竊� `unique SU_Others work days < 4 (unique_days=3)`
- **830400232A** 竊� `planned-window far cut: 103d away (> 90)` plus single-day outlier cuts
- **530N02621A** 竊� `phase zero workload under Decoder5 span-split check (counts={2: 35, 3: 0, 4: 96})`

So the cap mainly changes the **shape of valid shifted modules**, not the basic data-cleaning anomalies.

### What clearly changes between the two logs
The modules that differ most are the ones where p3 started early enough that the non-limit version allowed a long tail.

#### Example A: 530900102A
- **Non-limit**: 1 / 40 / 39
- **28-day limit**: 52 / 14 / 14

Interpretation:
- Non-limit says: 窶彗lmost everything after QC belongs to late phases.窶�
- 28-day limit says: 窶很eep only a standardized late tail and move the rest into p2.窶�

#### Example B: 530N03002A
- **Non-limit**: 14 / 27 / 27
- **28-day limit**: 40 / 14 / 14

Interpretation:
- Non-limit preserves a long p3/p4 tail.
- 28-day limit forces a compact late-phase block.

#### Example C: 530N02824A
- **Non-limit**: 7 / 19 / 20
- **28-day limit**: 18 / 14 / 14

Interpretation:
- Non-limit emphasizes late stages.
- 28-day limit restores more weight to p2.

#### Example D: 530900112A
- **Non-limit**: 22 / 25 / 26
- **28-day limit**: 45 / 14 / 14

Interpretation:
- Non-limit gives a fairly broad three-phase spread.
- 28-day limit creates a much stronger front-loaded structure.

---

## 4. Suggested abnormal categories for reporting to boss

You can group abnormalities into the following categories.

### A. Source data abnormalities
These come from the input itself.

- missing p2/p3/p4/p4end dates
- wrong date order in 譁ｰ隕剰｣ｽ逡ｪ繝ｪ繧ｹ繝�
- phase end < start after applying next-start-1 rule
- SU_Others worked evidence too short (`unique_days < 4`)

### B. Matching / shifting abnormalities
These come from the relation between 譁ｰ隕剰｣ｽ逡ｪ繝ｪ繧ｹ繝� and SU_Others.

- actual worked span is far from planned window
- QC trigger happens very early, causing phase 3 to start too soon
- actual execution is much longer than the planned overall duration
- compressed worked-day stitching causes calendar gaps in displayed windows

### C. Balance abnormalities
These come from how phase days are split.

#### More visible in non-limit version
- p2 becomes too short
- p3+p4 dominate the module
- shifted balance drifts far from planned phase ratios

#### More visible in 28-day limit version
- p2 becomes too long
- p3/p4 become too flat or too standardized
- real long late-stage work may be under-represented

### D. Structural anomalies worth highlighting separately
- **phase zero workload** cases (for example p3=0 while p4 is very large)
- **single-day worker participation** repeated across many modules
- **planned-window far cut** near or above the 90-day threshold
- **remaining unique worked dates too small vs planned total**

---

## 5. Practical conclusion

If the purpose is to create a result that is easier to explain and less extreme, the **28-day limit** version is stronger.

If the purpose is to preserve long real late-stage execution as much as possible, the **non-limit** version is stronger.

My practical reading is:

- For reporting and control, **28-day limit** is usually easier to defend.
- For historical realism, **non-limit** may reflect actual SU_Others behavior better.
- The most important abnormalities to report are not only the cuts, but also the **phase-balance abnormalities** created after shifting.

A simple summary sentence you can use is:

> Non-limit p3/p4 preserves actual late-stage spread but tends to create oversized p3/p4 and undersized p2, while the 28-day limit version stabilizes the late-stage tail but tends to expand p2 and flatten p3/p4 balance.

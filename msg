# Change

## 1. First pure QC appear → phase 3
If a pure QC worker appears on the task, the first date that pure QC joins is used as the start of phase 3.

## 2. First pure M or E out → phase 3
If there is no pure QC, then the first point where pure M or pure E leaves the task is used as the start of phase 3.

## 3. Module which have work first 10 date of start range → dummy all
If a module appears within the first 10 days from the SU_Others plan start range, the entire module is turned to dummy.

## 4. Module even outside of plan range but still continue task → flexible / used
Even if a module goes outside the planned range, it is still kept as flexible / used when the task is continuing without being broken.

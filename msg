I rewrote vba/CompareSuOthers.bas. It is still not tested in Excel, because I have no Excel here.

Your CompareSuOthers2.bas is an empty file (0 bytes), so I could not see your edits or the compile error. I rebuilt the original from what you described. If an error remains, send me the message text. Every failure now names the step and the error number.

What I changed

Press twice: the macro no longer copies the old sheet into your workbook, which is likely why the first click failed.
Copying the sheet across workbooks merged the roughly 54,000 named styles of two files, and that step is likely what failed.
Now both views are built from a copy of the new sheet inside the same workbook. Only the changed cells are taken over from the old file.
If a step fails, the half-built Compare_* sheets are deleted, so no stray copy stays.
Excel version: I could not see your compile error. I removed the things most likely to differ between versions: the user-defined type, the Office-typed constants, and a plain button macro name. The code uses only basic VBA now. This is a best guess without your error text.
First column marker: Compare_Old and Compare_New get a new first column, “Changed”. A red circle (○) sits on every worker row that has a change, on the same rows in both sheets. The filter arrow is on that header, so you can filter to only the circles.
Summary as a list: Compare_Summary is now one row per bar. The columns are Category (Add / Change / Delete, coloured), Seiban, Worker, When, Before, After, and the text before and after. It is sorted by category, seiban, date and worker. The pop-up message after the run only shows short counts.
Add: a new bar.
Change: a bar that moved, got longer or shorter, or was renamed.
Delete: a bar that is gone.

Checked with the same logic in Python

Moved, longer, renamed, new and deleted bars all give the right row.
FI is never paired with FO.
On your 3-3 input and output, it gives 104 list rows, one per red frame, all Add because it was the first week.

Install: remove your old CompareSuOthers module first, then import this file and run CreateCompareButton once. The README section on the macro is updated.

I left CompareSuOthers2.bas alone; you can delete it.

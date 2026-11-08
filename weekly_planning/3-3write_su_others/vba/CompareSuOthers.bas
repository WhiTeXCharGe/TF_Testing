Attribute VB_Name = "CompareSuOthers"
Option Explicit

' =====================================================================================================
' CompareSuOthers  -  compare the REWRITTEN SU_Others (step 3-3) with the OLD one (its input)
'
' How to use
'   1. Open the workbook that holds this module (the rewritten SU_Others, or the real one the rewritten file comes from).
'   2. Run CreateCompareButton once: it makes a sheet "Compare" with a button.
'   3. Click the button, choose the OLD SU_Others file (the input of step 3-3).
'   4. Three sheets are added to THIS workbook:
'        Compare_Old      copy of the old task sheet; every bar that is going to change has a red frame
'        Compare_New      copy of this workbook's task sheet; every added / changed bar has a red frame
'        Compare_Summary  one row per worker + seiban that changed:  Date, Seiban, Worker, Status, Before, After, Text before, Text after
'                         Status = Changed (was in the old version: Before and After), New (only After), Removed (only Before)
'   The task sheet of THIS workbook is the NEW version, the file you choose is the OLD version.
'   Nothing in the existing sheets is changed. Delete the three sheets when you are done (the next click recreates them).
'
' (ASCII only on purpose: Japanese names are built with ChrW so the file imports correctly on any Windows code page.)
' =====================================================================================================

Private Const SHEET_OLD As String = "Compare_Old"
Private Const SHEET_NEW As String = "Compare_New"
Private Const SHEET_SUM As String = "Compare_Summary"
Private Const SHEET_BTN As String = "Compare"
Private Const SEP As String = "|"

Private Type SheetData
    Values As Variant      ' 2D array: Values(sheetRow, sheetCol) for rows 1..LastRow, cols 1..LastCol
    DateRow As Long
    FirstRow As Long
    LastRow As Long
    LastCol As Long
    NameCol As Long
    CompCol As Long
    RowKey As Object       ' "company|name" (+ "#2" for a repeated one) -> sheet row
    DateCol As Object      ' date serial -> sheet column
End Type

' ---------------------------------------------------------------------------------- names
Private Function TaskSheetName() As String
    ' the task sheet of SU_Others: ChrW(4E88) ChrW(5B9A) ChrW(8868) & "_2026"
    TaskSheetName = ChrW(&H4E88) & ChrW(&H5B9A) & ChrW(&H8868) & "_2026"
End Function

' ---------------------------------------------------------------------------------- button
Public Sub CreateCompareButton()
    Dim ws As Worksheet, b As Object
    On Error Resume Next
    Set ws = ThisWorkbook.Worksheets(SHEET_BTN)
    On Error GoTo 0
    If ws Is Nothing Then
        Set ws = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
        ws.Name = SHEET_BTN
    End If
    For Each b In ws.Buttons
        b.Delete
    Next b
    Set b = ws.Buttons.Add(ws.Range("B2").Left, ws.Range("B2").Top, 240, 36)
    b.Caption = "Compare with old version"
    b.OnAction = "CompareWithOldVersion"
    ws.Range("B5").Value = "1. Click the button.  2. Choose the OLD SU_Others (the input of step 3-3)."
    ws.Range("B6").Value = "3. Sheets Compare_Old, Compare_New and Compare_Summary are added to this workbook."
    ws.Range("B7").Value = "Red frame = a bar that is changed or added. This workbook's task sheet is the NEW version."
End Sub

' ---------------------------------------------------------------------------------- main
Public Sub CompareWithOldVersion()
    Dim oldPath As Variant
    Dim wbOld As Workbook
    Dim secOld As MsoAutomationSecurity
    Dim dOld As SheetData, dNew As SheetData
    Dim wsOldSrc As Worksheet, wsNewSrc As Worksheet, wsOldCopy As Worksheet, wsNewCopy As Worksheet, wsSum As Worksheet
    Dim chgOld() As String, chgNew() As String
    Dim nDiff As Long, nMissing As Long, nRows As Long, nOutlines As Long

    oldPath = Application.GetOpenFilename("Excel files (*.xlsm;*.xlsx),*.xlsm;*.xlsx", , "Select the OLD SU_Others (the input of step 3-3)")
    If VarType(oldPath) = vbBoolean Then Exit Sub
    If LCase$(CStr(oldPath)) = LCase$(ThisWorkbook.FullName) Then
        MsgBox "Choose the OLD file, not this workbook.", vbExclamation
        Exit Sub
    End If

    secOld = Application.AutomationSecurity
    On Error GoTo Fail
    Application.ScreenUpdating = False
    Application.EnableEvents = False
    Application.DisplayAlerts = False

    Application.AutomationSecurity = msoAutomationSecurityForceDisable      ' do not run the macros of the old file
    Set wbOld = Workbooks.Open(Filename:=CStr(oldPath), UpdateLinks:=0, ReadOnly:=True)
    Application.AutomationSecurity = secOld

    Set wsOldSrc = wbOld.Worksheets(TaskSheetName())
    Set wsNewSrc = ThisWorkbook.Worksheets(TaskSheetName())
    dOld = LoadSheet(wsOldSrc)
    dNew = LoadSheet(wsNewSrc)

    ' ---- which cells differ (rows matched by company|name, columns by date)
    ReDim chgOld(1 To dOld.LastRow, 1 To dOld.LastCol)
    ReDim chgNew(1 To dNew.LastRow, 1 To dNew.LastCol)
    Dim key As Variant, dt As Variant
    Dim rO As Long, rN As Long, cO As Long, cN As Long, tO As String, tN As String
    For Each key In dNew.RowKey.Keys
        If dOld.RowKey.Exists(key) Then
            rN = dNew.RowKey(key)
            rO = dOld.RowKey(key)
            For Each dt In dNew.DateCol.Keys
                If dOld.DateCol.Exists(dt) Then
                    cN = dNew.DateCol(dt)
                    cO = dOld.DateCol(dt)
                    tO = CellText(dOld.Values(rO, cO))
                    tN = CellText(dNew.Values(rN, cN))
                    If StrComp(tO, tN, vbBinaryCompare) <> 0 Then
                        chgOld(rO, cO) = tO & Chr(1) & tN       ' same text on both sides = same run
                        chgNew(rN, cN) = tO & Chr(1) & tN
                        nDiff = nDiff + 1
                    End If
                End If
            Next dt
        Else
            nMissing = nMissing + 1
        End If
    Next key

    ' ---- the three sheets
    DeleteSheetIfExists SHEET_OLD
    DeleteSheetIfExists SHEET_NEW
    DeleteSheetIfExists SHEET_SUM

    wsOldSrc.Copy After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count)
    Set wsOldCopy = ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count)
    wsOldCopy.Name = SHEET_OLD
    RemoveControls wsOldCopy

    wsNewSrc.Copy After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count)
    Set wsNewCopy = ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count)
    wsNewCopy.Name = SHEET_NEW
    RemoveControls wsNewCopy

    nOutlines = OutlineRuns(wsOldCopy, dOld, chgOld) + OutlineRuns(wsNewCopy, dNew, chgNew)

    Set wsSum = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
    wsSum.Name = SHEET_SUM
    nRows = WriteSummary(wsSum, dOld, dNew, chgOld, chgNew)

    wbOld.Close SaveChanges:=False
    Set wbOld = Nothing

    Application.DisplayAlerts = True
    Application.EnableEvents = True
    Application.ScreenUpdating = True
    wsSum.Activate
    MsgBox "Changed cells: " & nDiff & vbCrLf & "Red frames: " & nOutlines & " (old + new sheet)" & vbCrLf & _
           "Summary rows: " & nRows & IIf(nMissing > 0, vbCrLf & "Workers only in the new file (not compared): " & nMissing, ""), vbInformation, "Compare done"
    Exit Sub

Fail:
    Dim msg As String
    msg = Err.Description
    On Error Resume Next
    If Not wbOld Is Nothing Then wbOld.Close SaveChanges:=False
    Application.AutomationSecurity = secOld
    Application.DisplayAlerts = True
    Application.EnableEvents = True
    Application.ScreenUpdating = True
    MsgBox "Compare failed: " & msg, vbCritical
End Sub

' ---------------------------------------------------------------------------------- reading a task sheet
Private Function LoadSheet(ws As Worksheet) As SheetData
    Dim d As SheetData
    Dim r As Long, c As Long, n As Long, i As Long
    Dim wName As String, wComp As String, k As String

    ' date header row: the first of rows 1..6 that has at least 20 dates in columns 10..60
    For r = 1 To 6
        n = 0
        For c = 10 To 60
            If IsDate(ws.Cells(r, c).Value) Then n = n + 1
        Next c
        If n >= 20 Then
            d.DateRow = r
            Exit For
        End If
    Next r
    If d.DateRow = 0 Then Err.Raise vbObjectError + 1, , "Date header row not found in " & ws.Parent.Name

    d.NameCol = 3
    d.CompCol = 1
    For c = 1 To 14
        Select Case CellText(ws.Cells(d.DateRow + 1, c).Value)
            Case ChrW(&H59D3) & ChrW(&H540D)                       ' name header
                d.NameCol = c
            Case ChrW(&H4F01) & ChrW(&H696D) & ChrW(&H540D)        ' company header
                d.CompCol = c
        End Select
    Next c

    d.LastCol = ws.Cells(d.DateRow, ws.Columns.Count).End(xlToLeft).Column
    d.FirstRow = d.DateRow + 2
    d.LastRow = ws.Cells(ws.Rows.Count, d.NameCol).End(xlUp).Row
    If d.LastRow < d.FirstRow Then Err.Raise vbObjectError + 2, , "No worker rows found in " & ws.Parent.Name

    d.Values = ws.Range(ws.Cells(1, 1), ws.Cells(d.LastRow, d.LastCol)).Value

    Set d.DateCol = CreateObject("Scripting.Dictionary")
    For c = 1 To d.LastCol
        If IsDate(d.Values(d.DateRow, c)) Then
            k = CStr(CLng(CDate(d.Values(d.DateRow, c))))
            If Not d.DateCol.Exists(k) Then d.DateCol.Add k, c
        End If
    Next c

    Set d.RowKey = CreateObject("Scripting.Dictionary")
    For r = d.FirstRow To d.LastRow
        wName = CellText(d.Values(r, d.NameCol))
        wComp = CellText(d.Values(r, d.CompCol))
        If Len(wName) > 0 And Len(wComp) > 0 Then
            k = wComp & SEP & wName
            i = 1
            Do While d.RowKey.Exists(k)
                i = i + 1
                k = wComp & SEP & wName & "#" & i
            Loop
            d.RowKey.Add k, r
        End If
    Next r
    LoadSheet = d
End Function

Private Function CellText(ByVal v As Variant) As String
    If IsError(v) Then Exit Function
    If IsEmpty(v) Then Exit Function
    CellText = Trim$(CStr(v))
End Function

' ---------------------------------------------------------------------------------- the sheets
Private Sub DeleteSheetIfExists(ByVal nm As String)
    Dim ws As Worksheet
    On Error Resume Next
    Set ws = ThisWorkbook.Worksheets(nm)
    On Error GoTo 0
    If Not ws Is Nothing Then ws.Delete
End Sub

Private Sub RemoveControls(ws As Worksheet)
    ' the copies are views: remove buttons / form controls so they do not run macros of the other workbook
    Dim i As Long
    For i = ws.Shapes.Count To 1 Step -1
        If ws.Shapes(i).Type = msoFormControl Then ws.Shapes(i).Delete
    Next i
End Sub

' red frame around every run of changed cells in a row (a run = neighbouring columns with the same old and new text)
Private Function OutlineRuns(ws As Worksheet, d As SheetData, chg() As String) As Long
    Dim r As Long, c As Long, c2 As Long, n As Long
    For r = d.FirstRow To d.LastRow
        c = 1
        Do While c <= d.LastCol
            If Len(chg(r, c)) > 0 Then
                c2 = c
                Do While c2 < d.LastCol
                    If chg(r, c2 + 1) = chg(r, c) Then c2 = c2 + 1 Else Exit Do
                Loop
                ws.Range(ws.Cells(r, c), ws.Cells(r, c2)).BorderAround LineStyle:=xlContinuous, Weight:=xlMedium, Color:=RGB(255, 0, 0)
                n = n + 1
                c = c2 + 1
            Else
                c = c + 1
            End If
        Loop
    Next r
    OutlineRuns = n
End Function

' ---------------------------------------------------------------------------------- summary
' "label" of a cell text: the seiban (module code), "FI/FO" for a mark, otherwise "(other) text"
Private Function LabelOf(ByVal t As String, re As Object, cache As Object) As String
    Dim s As String, m As Object, res As String
    If cache.Exists(t) Then
        LabelOf = cache(t)
        Exit Function
    End If
    s = t
    On Error Resume Next
    s = StrConv(t, vbNarrow)          ' full-width -> half-width (Japanese locale)
    On Error GoTo 0
    s = UCase$(Trim$(s))
    If s = "FI" Or s = "FO" Then
        res = "FI/FO"
    Else
        Set m = re.Execute(s)
        If m.Count > 0 Then
            res = m(0).Value
        Else
            res = "(other) " & Left$(t, 30)
        End If
    End If
    cache.Add t, res
    LabelOf = res
End Function

' every run (neighbouring date columns, same text) of every worker, grouped by worker|label:  runs(pairKey)(runString) = True
Private Sub CollectRuns(d As SheetData, re As Object, cache As Object, runs As Object, pairWorker As Object, pairLabel As Object)
    Dim colDate() As Long, key As Variant, dt As Variant
    Dim r As Long, c As Long, t As String, curText As String, curStart As Long, curEnd As Long, lastC As Long

    ReDim colDate(1 To d.LastCol)
    For Each dt In d.DateCol.Keys
        colDate(d.DateCol(dt)) = CLng(dt)
    Next dt

    For Each key In d.RowKey.Keys
        r = d.RowKey(key)
        curText = ""
        lastC = 0
        For c = 1 To d.LastCol
            If colDate(c) > 0 Then
                t = CellText(d.Values(r, c))
                If Len(t) > 0 And t = curText And c = lastC + 1 Then
                    curEnd = colDate(c)
                Else
                    If Len(curText) > 0 Then AddRun runs, pairWorker, pairLabel, CStr(key), curText, curStart, curEnd, re, cache
                    If Len(t) > 0 Then
                        curText = t
                        curStart = colDate(c)
                        curEnd = colDate(c)
                    Else
                        curText = ""
                    End If
                End If
                lastC = c
            End If
        Next c
        If Len(curText) > 0 Then AddRun runs, pairWorker, pairLabel, CStr(key), curText, curStart, curEnd, re, cache
    Next key
End Sub

Private Sub AddRun(runs As Object, pairWorker As Object, pairLabel As Object, ByVal rowKey As String, ByVal txt As String, _
                   ByVal startSerial As Long, ByVal endSerial As Long, re As Object, cache As Object)
    Dim lbl As String, pk As String, rs As String
    lbl = LabelOf(txt, re, cache)
    pk = rowKey & Chr(2) & lbl
    rs = Format$(startSerial, "000000") & SEP & Format$(endSerial, "000000") & SEP & txt     ' sorts by date as text
    If Not runs.Exists(pk) Then runs.Add pk, CreateObject("Scripting.Dictionary")
    If Not pairWorker.Exists(pk) Then
        pairWorker.Add pk, Mid$(rowKey, InStr(rowKey, SEP) + 1)
        pairLabel.Add pk, lbl
    End If
    runs(pk)(rs) = True
End Sub

Private Function RunsText(ByVal runList As Object, ByVal other As Object, ByRef firstSerial As Long, ByRef texts As String) As String
    ' the runs of runList that are not in other (same start, end and text), as "yyyy/mm/dd - yyyy/mm/dd; ..."
    Dim arr() As String, rs As Variant, n As Long, i As Long, j As Long, tmp As String, p() As String, s As String
    Dim res As String
    Dim seenText As Object
    Set seenText = CreateObject("Scripting.Dictionary")
    If Not runList Is Nothing Then
        For Each rs In runList.Keys
            If other Is Nothing Then
                n = n + 1
                ReDim Preserve arr(1 To n)
                arr(n) = CStr(rs)
            ElseIf Not other.Exists(rs) Then
                n = n + 1
                ReDim Preserve arr(1 To n)
                arr(n) = CStr(rs)
            End If
        Next rs
    End If
    If n = 0 Then Exit Function
    For i = 1 To n - 1                       ' small bubble sort (text sorts by date)
        For j = i + 1 To n
            If arr(j) < arr(i) Then tmp = arr(i): arr(i) = arr(j): arr(j) = tmp
        Next j
    Next i
    For i = 1 To n
        p = Split(arr(i), SEP, 3)
        If i = 1 Then firstSerial = CLng(p(0))
        s = Format$(CDate(CLng(p(0))), "yyyy/mm/dd")
        If p(1) <> p(0) Then s = s & " - " & Format$(CDate(CLng(p(1))), "yyyy/mm/dd")
        res = res & IIf(Len(res) > 0, "; ", "") & s
        If Not seenText.Exists(p(2)) Then
            seenText.Add p(2), True
            texts = texts & IIf(Len(texts) > 0, "; ", "") & p(2)
        End If
    Next i
    RunsText = res
End Function

Private Function WriteSummary(ws As Worksheet, dOld As SheetData, dNew As SheetData, chgOld() As String, chgNew() As String) As Long
    Dim re As Object, cache As Object
    Dim runsOld As Object, runsNew As Object, pairWorker As Object, pairLabel As Object, affected As Object
    Dim pk As Variant

    Set re = CreateObject("VBScript.RegExp")
    re.Pattern = "\d{3}[A-Z0-9]\d{5}A"
    re.Global = False
    Set cache = CreateObject("Scripting.Dictionary")
    Set runsOld = CreateObject("Scripting.Dictionary")
    Set runsNew = CreateObject("Scripting.Dictionary")
    Set pairWorker = CreateObject("Scripting.Dictionary")
    Set pairLabel = CreateObject("Scripting.Dictionary")
    Set affected = CreateObject("Scripting.Dictionary")

    CollectRuns dOld, re, cache, runsOld, pairWorker, pairLabel
    CollectRuns dNew, re, cache, runsNew, pairWorker, pairLabel

    ' worker + seiban pairs that have at least one changed cell (old text or new text)
    MarkAffected dOld, chgOld, re, cache, affected
    MarkAffected dNew, chgNew, re, cache, affected

    Dim n As Long
    Dim outRows() As Variant
    n = 0
    ReDim outRows(1 To affected.Count + 1, 1 To 8)

    Dim oldSet As Object, newSet As Object, bef As String, aft As String, txB As String, txA As String
    Dim fB As Long, fA As Long, first As Long, status As String
    Dim order() As String
    ReDim order(1 To affected.Count + 1)

    For Each pk In affected.Keys
        Set oldSet = Nothing: Set newSet = Nothing
        If runsOld.Exists(pk) Then Set oldSet = runsOld(pk)
        If runsNew.Exists(pk) Then Set newSet = runsNew(pk)
        bef = "": aft = "": txB = "": txA = "": fB = 0: fA = 0
        bef = RunsText(oldSet, newSet, fB, txB)
        aft = RunsText(newSet, oldSet, fA, txA)
        If Len(bef) = 0 And Len(aft) = 0 Then GoTo NextPair
        If Len(bef) > 0 And Len(aft) > 0 Then
            status = "Changed"
        ElseIf Len(aft) > 0 Then
            status = "New"
        Else
            status = "Removed"
        End If
        If fA > 0 Then first = fA Else first = fB
        n = n + 1
        outRows(n, 1) = CDate(first)
        outRows(n, 2) = pairLabel(pk)
        outRows(n, 3) = pairWorker(pk)
        outRows(n, 4) = status
        outRows(n, 5) = bef
        outRows(n, 6) = aft
        outRows(n, 7) = txB
        outRows(n, 8) = txA
        order(n) = Format$(first, "000000") & SEP & pairLabel(pk) & SEP & pairWorker(pk) & SEP & Format$(n, "000000")
NextPair:
    Next pk

    ws.Range("A1:H1").Value = Array("Date", "Seiban", "Worker", "Status", "Before", "After", "Text before", "Text after")
    ws.Range("A1:H1").Font.Bold = True
    If n > 0 Then
        Dim i As Long, j As Long, tmp As String, idx As Long
        For i = 1 To n - 1                   ' sort rows by date, seiban, worker
            For j = i + 1 To n
                If order(j) < order(i) Then tmp = order(i): order(i) = order(j): order(j) = tmp
            Next j
        Next i
        Dim sorted() As Variant
        ReDim sorted(1 To n, 1 To 8)
        For i = 1 To n
            idx = CLng(Mid$(order(i), InStrRev(order(i), SEP) + 1))
            For j = 1 To 8
                sorted(i, j) = outRows(idx, j)
            Next j
        Next i
        ws.Range(ws.Cells(2, 1), ws.Cells(n + 1, 8)).Value = sorted
        ws.Range(ws.Cells(2, 1), ws.Cells(n + 1, 1)).NumberFormat = "yyyy/mm/dd"
    End If
    ws.Columns("A:H").AutoFit
    If ws.Columns("E").ColumnWidth > 60 Then ws.Columns("E").ColumnWidth = 60
    If ws.Columns("F").ColumnWidth > 60 Then ws.Columns("F").ColumnWidth = 60
    If ws.Columns("G").ColumnWidth > 50 Then ws.Columns("G").ColumnWidth = 50
    If ws.Columns("H").ColumnWidth > 50 Then ws.Columns("H").ColumnWidth = 50
    ws.Range("A1:H1").AutoFilter
    WriteSummary = n
End Function

Private Sub MarkAffected(d As SheetData, chg() As String, re As Object, cache As Object, affected As Object)
    Dim key As Variant, r As Long, c As Long, parts() As String, t As Variant, pk As String
    For Each key In d.RowKey.Keys
        r = d.RowKey(key)
        For c = 1 To d.LastCol
            If Len(chg(r, c)) > 0 Then
                parts = Split(chg(r, c), Chr(1))
                For Each t In parts
                    If Len(CStr(t)) > 0 Then
                        pk = CStr(key) & Chr(2) & LabelOf(CStr(t), re, cache)
                        If Not affected.Exists(pk) Then affected.Add pk, True
                    End If
                Next t
            End If
        Next c
    Next key
End Sub

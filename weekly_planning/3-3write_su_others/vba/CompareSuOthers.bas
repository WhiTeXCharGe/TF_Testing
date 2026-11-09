Attribute VB_Name = "CompareSuOthers"
Option Explicit

' =====================================================================================================
' CompareSuOthers  -  compare the REWRITTEN SU_Others (step 3-3) with the OLD one (its input)
'
' Install : Alt+F11 -> File -> Import File -> this .bas   (remove an older "CompareSuOthers" module first)
'           then run CreateCompareButton once (Alt+F8): it makes a sheet "Compare" with a button.
' Use     : open the rewritten file, click the button, choose the OLD SU_Others (the input of step 3-3).
'           The task sheet of THIS workbook is the NEW version, the file you choose is the OLD version.
'
' Three sheets are added to THIS workbook (the old file is only read; nothing existing is changed):
'   Compare_Old      the old task sheet. Every bar that is going to change has a red frame.
'   Compare_New      the new task sheet. Every added or changed bar has a red frame.
'                    Both sheets have a new first column "Changed": a circle on every worker row that has a change
'                    (and the filter arrow is on, so you can show only the rows that changed).
'   Compare_Summary  a list, one row per bar:  Category (Add / Change / Delete), Seiban, Worker, When,
'                    Before, After, Text before, Text after.
'                      Add     = a bar that is new in the new version        (When = After)
'                      Change  = a bar that moved / got longer / was renamed (Before and After)
'                      Delete  = a bar that is gone in the new version       (When = Before)
'
' The old sheet is NOT copied across workbooks (that was heavy and failed on the first click): both views are built
' from a copy of the NEW sheet inside this workbook, and only the changed cells are taken over from the old file.
' If anything fails, the message names the step, and the half-built Compare_* sheets are deleted again.
'
' (ASCII only on purpose: Japanese names are built with ChrW so the file imports correctly on any Windows code page.)
' =====================================================================================================

Private Const SHEET_OLD As String = "Compare_Old"
Private Const SHEET_NEW As String = "Compare_New"
Private Const SHEET_SUM As String = "Compare_Summary"
Private Const SHEET_BTN As String = "Compare"
Private Const SEP As String = "|"

' indexes of the Variant array that LoadSheet returns
Private Const I_VALUES As Long = 0
Private Const I_DATEROW As Long = 1
Private Const I_FIRSTROW As Long = 2
Private Const I_LASTROW As Long = 3
Private Const I_LASTCOL As Long = 4
Private Const I_NAMECOL As Long = 5
Private Const I_COMPCOL As Long = 6
Private Const I_ROWKEY As Long = 7
Private Const I_DATECOL As Long = 8

' ---------------------------------------------------------------------------------- names
Private Function TaskSheetName() As String
    ' the task sheet of SU_Others: ChrW(4E88) ChrW(5B9A) ChrW(8868) & "_2026"
    TaskSheetName = ChrW(&H4E88) & ChrW(&H5B9A) & ChrW(&H8868) & "_2026"
End Function

Private Function MarkSymbol() As String
    MarkSymbol = ChrW(&H25CB)        ' a circle
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
    b.OnAction = "'" & ThisWorkbook.Name & "'!CompareWithOldVersion"
    ws.Range("B5").Value = "1. Click the button.  2. Choose the OLD SU_Others (the input of step 3-3)."
    ws.Range("B6").Value = "3. Sheets Compare_Old, Compare_New and Compare_Summary are added to this workbook."
    ws.Range("B7").Value = "Red frame = a bar that is changed or added. Column A of the Compare sheets marks the rows that changed."
    ws.Range("B8").Value = "This workbook's task sheet is the NEW version."
End Sub

' ---------------------------------------------------------------------------------- main
Public Sub CompareWithOldVersion()
    Dim oldPath As Variant
    Dim wbOld As Workbook
    Dim secOld As Long
    Dim stepName As String
    Dim oInfo As Variant, nInfo As Variant, oVals As Variant, nVals As Variant
    Dim oRowKey As Object, nRowKey As Object, oDateCol As Object, nDateCol As Object
    Dim wsOldSrc As Worksheet, wsNewSrc As Worksheet, wsOldView As Worksheet, wsNewView As Worksheet, wsSum As Worksheet
    Dim chg() As String
    Dim dRowO() As Long, dColO() As Long, dRowN() As Long, dColN() As Long
    Dim cap As Long, nDiff As Long, nFrames As Long, nMissing As Long, i As Long
    Dim key As Variant, dt As Variant
    Dim rO As Long, rN As Long, cO As Long, cN As Long, tO As String, tN As String
    Dim counts(1 To 3) As Long, nSummary As Long

    oldPath = Application.GetOpenFilename("Excel files (*.xlsm;*.xlsx),*.xlsm;*.xlsx", , "Select the OLD SU_Others (the input of step 3-3)")
    If VarType(oldPath) = vbBoolean Then Exit Sub
    If LCase$(CStr(oldPath)) = LCase$(ThisWorkbook.FullName) Then
        MsgBox "Choose the OLD file, not this workbook.", vbExclamation
        Exit Sub
    End If

    secOld = Application.AutomationSecurity
    stepName = "preparing"
    On Error GoTo Fail
    Application.ScreenUpdating = False
    Application.EnableEvents = False
    Application.DisplayAlerts = False

    stepName = "opening the old file (read only, macros off)"
    Application.AutomationSecurity = 3                  ' msoAutomationSecurityForceDisable: do not run the macros of the old file
    Set wbOld = Workbooks.Open(Filename:=CStr(oldPath), UpdateLinks:=0, ReadOnly:=True)
    Application.AutomationSecurity = secOld

    stepName = "reading the task sheets"
    Set wsOldSrc = wbOld.Worksheets(TaskSheetName())
    Set wsNewSrc = ThisWorkbook.Worksheets(TaskSheetName())
    oInfo = LoadSheet(wsOldSrc)
    nInfo = LoadSheet(wsNewSrc)
    oVals = oInfo(I_VALUES)
    nVals = nInfo(I_VALUES)
    Set oRowKey = oInfo(I_ROWKEY)
    Set nRowKey = nInfo(I_ROWKEY)
    Set oDateCol = oInfo(I_DATECOL)
    Set nDateCol = nInfo(I_DATECOL)

    stepName = "finding the changed cells"
    ' rows are matched by company|name, columns by date. chg(row, col) (coordinates of the NEW sheet) = "old text" & Chr(1) & "new text"
    ReDim chg(1 To nInfo(I_LASTROW), 1 To nInfo(I_LASTCOL))
    cap = 1024
    ReDim dRowO(1 To cap): ReDim dColO(1 To cap): ReDim dRowN(1 To cap): ReDim dColN(1 To cap)
    For Each key In nRowKey.Keys
        If oRowKey.Exists(key) Then
            rN = nRowKey(key)
            rO = oRowKey(key)
            For Each dt In nDateCol.Keys
                If oDateCol.Exists(dt) Then
                    cN = nDateCol(dt)
                    cO = oDateCol(dt)
                    tO = CellText(oVals(rO, cO))
                    tN = CellText(nVals(rN, cN))
                    If StrComp(tO, tN, vbBinaryCompare) <> 0 Then
                        nDiff = nDiff + 1
                        If nDiff > cap Then
                            cap = cap * 2
                            ReDim Preserve dRowO(1 To cap): ReDim Preserve dColO(1 To cap)
                            ReDim Preserve dRowN(1 To cap): ReDim Preserve dColN(1 To cap)
                        End If
                        dRowO(nDiff) = rO: dColO(nDiff) = cO: dRowN(nDiff) = rN: dColN(nDiff) = cN
                        chg(rN, cN) = tO & Chr(1) & tN
                    End If
                End If
            Next dt
        Else
            nMissing = nMissing + 1
        End If
    Next key

    stepName = "building Compare_New (a copy of the new sheet in this workbook)"
    DeleteCompareSheets
    wsNewSrc.Copy After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count)
    Set wsNewView = ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count)
    wsNewView.Name = SHEET_NEW
    RemoveControls wsNewView

    stepName = "building Compare_Old (the same sheet, with the old cells put back)"
    wsNewSrc.Copy After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count)
    Set wsOldView = ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count)
    wsOldView.Name = SHEET_OLD
    RemoveControls wsOldView
    For i = 1 To nDiff                                  ' only the changed cells come from the old file (value + format)
        wsOldSrc.Cells(dRowO(i), dColO(i)).Copy Destination:=wsOldView.Cells(dRowN(i), dColN(i))
    Next i
    Application.CutCopyMode = False

    stepName = "drawing the red frames"
    nFrames = OutlineRuns(wsOldView, nInfo, chg)
    nFrames = nFrames + OutlineRuns(wsNewView, nInfo, chg)

    stepName = "adding the Changed column"
    MarkChangedRows wsOldView, nInfo, chg
    MarkChangedRows wsNewView, nInfo, chg

    stepName = "building Compare_Summary"
    Set wsSum = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
    wsSum.Name = SHEET_SUM
    nSummary = WriteSummary(wsSum, oInfo, nInfo, chg, counts)

    stepName = "closing the old file"
    wbOld.Close SaveChanges:=False
    Set wbOld = Nothing

    Application.DisplayAlerts = True
    Application.EnableEvents = True
    Application.ScreenUpdating = True
    On Error Resume Next
    ThisWorkbook.Activate
    wsSum.Activate
    wsSum.Range("A2").Select
    ActiveWindow.FreezePanes = True
    On Error GoTo 0
    MsgBox "Add: " & counts(1) & "   Change: " & counts(2) & "   Delete: " & counts(3) & vbCrLf & _
           "(changed cells: " & nDiff & ", red frames: " & nFrames & ")" & _
           IIf(nMissing > 0, vbCrLf & "Workers only in the new file, not compared: " & nMissing, "") & vbCrLf & _
           "See the list on the sheet " & SHEET_SUM & ".", vbInformation, "Compare done"
    Exit Sub

Fail:
    Dim msg As String
    msg = "Compare failed at step: " & stepName & vbCrLf & "Error " & Err.Number & ": " & Err.Description
    On Error Resume Next
    Application.CutCopyMode = False
    If Not wbOld Is Nothing Then wbOld.Close SaveChanges:=False
    Application.AutomationSecurity = secOld
    DeleteCompareSheets                                 ' do not leave half-built copies behind
    Application.DisplayAlerts = True
    Application.EnableEvents = True
    Application.ScreenUpdating = True
    MsgBox msg, vbCritical
End Sub

' ---------------------------------------------------------------------------------- reading a task sheet
' returns Array(Values, DateRow, FirstRow, LastRow, LastCol, NameCol, CompCol, RowKey, DateCol)
Private Function LoadSheet(ws As Worksheet) As Variant
    Dim vals As Variant, rowKey As Object, dateCol As Object
    Dim dateRow As Long, firstRow As Long, lastRow As Long, lastCol As Long, nameCol As Long, compCol As Long
    Dim r As Long, c As Long, n As Long, i As Long
    Dim wName As String, wComp As String, k As String

    ' date header row: the first of rows 1..6 that has at least 20 dates in columns 10..60
    dateRow = 0
    For r = 1 To 6
        n = 0
        For c = 10 To 60
            If IsDate(ws.Cells(r, c).Value) Then n = n + 1
        Next c
        If n >= 20 Then
            dateRow = r
            Exit For
        End If
    Next r
    If dateRow = 0 Then Err.Raise vbObjectError + 1, , "Date header row not found in " & ws.Parent.Name

    nameCol = 3
    compCol = 1
    For c = 1 To 14
        Select Case CellText(ws.Cells(dateRow + 1, c).Value)
            Case ChrW(&H59D3) & ChrW(&H540D)                       ' name header
                nameCol = c
            Case ChrW(&H4F01) & ChrW(&H696D) & ChrW(&H540D)        ' company header
                compCol = c
        End Select
    Next c

    lastCol = ws.Cells(dateRow, ws.Columns.Count).End(xlToLeft).Column
    firstRow = dateRow + 2
    lastRow = ws.Cells(ws.Rows.Count, nameCol).End(xlUp).Row
    If lastRow < firstRow Then Err.Raise vbObjectError + 2, , "No worker rows found in " & ws.Parent.Name

    vals = ws.Range(ws.Cells(1, 1), ws.Cells(lastRow, lastCol)).Value

    Set dateCol = CreateObject("Scripting.Dictionary")
    For c = 1 To lastCol
        If IsDate(vals(dateRow, c)) Then
            k = CStr(CLng(CDate(vals(dateRow, c))))
            If Not dateCol.Exists(k) Then dateCol.Add k, c
        End If
    Next c

    Set rowKey = CreateObject("Scripting.Dictionary")
    For r = firstRow To lastRow
        wName = CellText(vals(r, nameCol))
        wComp = CellText(vals(r, compCol))
        If Len(wName) > 0 And Len(wComp) > 0 Then
            k = wComp & SEP & wName
            i = 1
            Do While rowKey.Exists(k)
                i = i + 1
                k = wComp & SEP & wName & "#" & i
            Loop
            rowKey.Add k, r
        End If
    Next r
    LoadSheet = Array(vals, dateRow, firstRow, lastRow, lastCol, nameCol, compCol, rowKey, dateCol)
End Function

Private Function CellText(ByVal v As Variant) As String
    If IsError(v) Then Exit Function
    If IsEmpty(v) Then Exit Function
    CellText = Trim$(CStr(v))
End Function

' ---------------------------------------------------------------------------------- the sheets
Private Sub DeleteCompareSheets()
    Dim nm As Variant, ws As Worksheet
    For Each nm In Array(SHEET_OLD, SHEET_NEW, SHEET_SUM)
        Set ws = Nothing
        On Error Resume Next
        Set ws = ThisWorkbook.Worksheets(CStr(nm))
        On Error GoTo 0
        If Not ws Is Nothing Then
            Application.DisplayAlerts = False
            ws.Delete
        End If
    Next nm
End Sub

Private Sub RemoveControls(ws As Worksheet)
    ' the copies are views: remove buttons / form controls so they do not run macros
    Dim i As Long
    On Error Resume Next
    For i = ws.Shapes.Count To 1 Step -1
        If ws.Shapes(i).Type = 8 Then ws.Shapes(i).Delete          ' 8 = msoFormControl
    Next i
End Sub

' red frame around every run of changed cells in a row (a run = neighbouring columns with the same old and new text)
Private Function OutlineRuns(ws As Worksheet, info As Variant, chg() As String) As Long
    Dim r As Long, c As Long, c2 As Long, n As Long, lastCol As Long
    lastCol = info(I_LASTCOL)
    For r = info(I_FIRSTROW) To info(I_LASTROW)
        c = 1
        Do While c <= lastCol
            If Len(chg(r, c)) > 0 Then
                c2 = c
                Do While c2 < lastCol
                    If chg(r, c2 + 1) = chg(r, c) Then
                        c2 = c2 + 1
                    Else
                        Exit Do
                    End If
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

' a new first column "Changed": a circle on every worker row that has a changed cell, with the filter on the header row
Private Sub MarkChangedRows(ws As Worksheet, info As Variant, chg() As String)
    Dim r As Long, c As Long, lastCol As Long, lastRow As Long, labelRow As Long
    Dim hasChg() As Boolean
    lastCol = info(I_LASTCOL)
    lastRow = info(I_LASTROW)
    labelRow = info(I_DATEROW) + 1
    ReDim hasChg(1 To lastRow)
    For r = info(I_FIRSTROW) To lastRow
        For c = 1 To lastCol
            If Len(chg(r, c)) > 0 Then
                hasChg(r) = True
                Exit For
            End If
        Next c
    Next r

    ws.AutoFilterMode = False
    ws.Columns(1).Insert Shift:=xlToRight
    ws.Columns(1).ColumnWidth = 9
    ws.Columns(1).HorizontalAlignment = xlCenter
    ws.Cells(labelRow, 1).Value = "Changed"
    ws.Cells(labelRow, 1).Font.Bold = True
    For r = info(I_FIRSTROW) To lastRow
        If hasChg(r) Then
            ws.Cells(r, 1).Value = MarkSymbol()
            ws.Cells(r, 1).Font.Color = RGB(255, 0, 0)
        End If
    Next r
    ws.Range(ws.Cells(labelRow, 1), ws.Cells(lastRow, lastCol + 1)).AutoFilter
End Sub

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
' runString = "SSSSSS|EEEEEE|text" (start / end date serial, 6 digits, so the strings sort by date)
Private Sub CollectRuns(info As Variant, re As Object, cache As Object, runs As Object, pairWorker As Object, pairLabel As Object)
    Dim vals As Variant, rowKey As Object, dateCol As Object
    Dim colDate() As Long, key As Variant, dt As Variant
    Dim r As Long, c As Long, lastCol As Long, t As String, curText As String, curStart As Long, curEnd As Long, lastC As Long

    vals = info(I_VALUES)
    Set rowKey = info(I_ROWKEY)
    Set dateCol = info(I_DATECOL)
    lastCol = info(I_LASTCOL)
    ReDim colDate(1 To lastCol)
    For Each dt In dateCol.Keys
        colDate(dateCol(dt)) = CLng(dt)
    Next dt

    For Each key In rowKey.Keys
        r = rowKey(key)
        curText = ""
        lastC = 0
        For c = 1 To lastCol
            If colDate(c) > 0 Then
                t = CellText(vals(r, c))
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
    rs = Format$(startSerial, "000000") & SEP & Format$(endSerial, "000000") & SEP & txt
    If Not runs.Exists(pk) Then runs.Add pk, CreateObject("Scripting.Dictionary")
    If Not pairWorker.Exists(pk) Then
        pairWorker.Add pk, Mid$(rowKey, InStr(rowKey, SEP) + 1)
        pairLabel.Add pk, lbl
    End If
    runs(pk)(rs) = True
End Sub

' runs of runList that are not in other (same start, end and text), sorted by date; returns how many
Private Function MissingRuns(runList As Object, other As Object, arr() As String) As Long
    Dim rs As Variant, n As Long, i As Long, j As Long, tmp As String
    If runList Is Nothing Then Exit Function
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
    For i = 2 To n                                     ' insertion sort (the strings sort by date)
        tmp = arr(i)
        j = i - 1
        Do While j >= 1
            If arr(j) <= tmp Then Exit Do
            arr(j + 1) = arr(j)
            j = j - 1
        Loop
        arr(j + 1) = tmp
    Next i
    MissingRuns = n
End Function

Private Function RunDates(ByVal rs As String) As String
    RunDates = Left$(rs, 13)                           ' "SSSSSS|EEEEEE"
End Function

Private Function RunText(ByVal rs As String) As String
    RunText = Mid$(rs, 15)
End Function

Private Function RunWhen(ByVal rs As String) As String
    Dim s As String, e As String
    s = Format$(CDate(CLng(Left$(rs, 6))), "yyyy/mm/dd")
    e = Format$(CDate(CLng(Mid$(rs, 8, 6))), "yyyy/mm/dd")
    If s = e Then RunWhen = s Else RunWhen = s & " - " & e
End Function

' one list row. remRun = the bar of the old version ("" for Add), addRun = the bar of the new version ("" for Delete)
Private Sub AddListRow(rowsOut As Collection, counts() As Long, ByVal lbl As String, ByVal worker As String, _
                       ByVal remRun As String, ByVal addRun As String)
    Dim cat As String, catNo As Long, whenTxt As String, firstSerial As Long, sortLabel As String
    Dim befTxt As String, aftTxt As String, textB As String, textA As String
    If Len(remRun) > 0 Then
        befTxt = RunWhen(remRun)
        textB = RunText(remRun)
    End If
    If Len(addRun) > 0 Then
        aftTxt = RunWhen(addRun)
        textA = RunText(addRun)
    End If
    If Len(remRun) = 0 Then
        cat = "Add": catNo = 1
    ElseIf Len(addRun) = 0 Then
        cat = "Delete": catNo = 3
    Else
        cat = "Change": catNo = 2
    End If
    counts(catNo) = counts(catNo) + 1
    If Len(addRun) > 0 Then
        whenTxt = RunWhen(addRun)
        firstSerial = CLng(Left$(addRun, 6))
    Else
        whenTxt = RunWhen(remRun)
        firstSerial = CLng(Left$(remRun, 6))
    End If
    sortLabel = lbl
    If lbl = "FI/FO" Then sortLabel = "~" & lbl
    If Left$(lbl, 1) = "(" Then sortLabel = "~~" & lbl
    rowsOut.Add Array(catNo & SEP & sortLabel & SEP & lbl & SEP & Format$(firstSerial, "000000") & SEP & worker & SEP & Format$(rowsOut.Count + 1, "000000"), _
                      cat, lbl, worker, whenTxt, befTxt, aftTxt, textB, textA)
End Sub

' the bars of one worker + seiban that differ between the versions -> Add / Change / Delete rows
Private Sub EmitPair(ByVal lbl As String, ByVal worker As String, oldSet As Object, newSet As Object, rowsOut As Collection, counts() As Long)
    Dim remArr() As String, addArr() As String, usedR() As Boolean, usedA() As Boolean
    Dim nR As Long, nA As Long, i As Long, j As Long, pass As Long, ok As Boolean
    nR = MissingRuns(oldSet, newSet, remArr)
    nA = MissingRuns(newSet, oldSet, addArr)
    If nR > 0 Then ReDim usedR(1 To nR)
    If nA > 0 Then ReDim usedA(1 To nA)

    ' pair a removed bar with an added bar: 1 = same days (the text changed), 2 = same text (moved / longer / shorter),
    ' 3 = anything left in order (not for FI/FO marks)
    For pass = 1 To 3
        If pass < 3 Or lbl <> "FI/FO" Then
            For i = 1 To nR
                If Not usedR(i) Then
                    For j = 1 To nA
                        If Not usedA(j) Then
                            Select Case pass
                                Case 1: ok = (RunDates(remArr(i)) = RunDates(addArr(j)))
                                Case 2: ok = (RunText(remArr(i)) = RunText(addArr(j)))
                                Case Else: ok = True
                            End Select
                            If ok Then
                                usedR(i) = True
                                usedA(j) = True
                                AddListRow rowsOut, counts, lbl, worker, remArr(i), addArr(j)
                                Exit For
                            End If
                        End If
                    Next j
                End If
            Next i
        End If
    Next pass
    For j = 1 To nA
        If Not usedA(j) Then AddListRow rowsOut, counts, lbl, worker, "", addArr(j)
    Next j
    For i = 1 To nR
        If Not usedR(i) Then AddListRow rowsOut, counts, lbl, worker, remArr(i), ""
    Next i
End Sub

Private Function WriteSummary(ws As Worksheet, oInfo As Variant, nInfo As Variant, chg() As String, counts() As Long) As Long
    Dim re As Object, cache As Object
    Dim runsOld As Object, runsNew As Object, pairWorker As Object, pairLabel As Object, affected As Object
    Dim rowsOut As Collection
    Dim pk As Variant, oldSet As Object, newSet As Object
    Dim n As Long, i As Long, j As Long, idx As Long, tmp As String, lastRow As Long
    Dim sk() As String, ix() As Long, itm As Variant, outArr() As Variant, tt As Long

    Set re = CreateObject("VBScript.RegExp")
    re.Pattern = "\d{3}[A-Z0-9]\d{5}A"
    re.Global = False
    Set cache = CreateObject("Scripting.Dictionary")
    Set runsOld = CreateObject("Scripting.Dictionary")
    Set runsNew = CreateObject("Scripting.Dictionary")
    Set pairWorker = CreateObject("Scripting.Dictionary")
    Set pairLabel = CreateObject("Scripting.Dictionary")
    Set affected = CreateObject("Scripting.Dictionary")
    Set rowsOut = New Collection

    CollectRuns oInfo, re, cache, runsOld, pairWorker, pairLabel
    CollectRuns nInfo, re, cache, runsNew, pairWorker, pairLabel
    MarkAffected nInfo, chg, re, cache, affected

    For Each pk In affected.Keys
        Set oldSet = Nothing
        Set newSet = Nothing
        If runsOld.Exists(pk) Then Set oldSet = runsOld(pk)
        If runsNew.Exists(pk) Then Set newSet = runsNew(pk)
        EmitPair CStr(pairLabel(pk)), CStr(pairWorker(pk)), oldSet, newSet, rowsOut, counts
    Next pk

    ws.Range("A1:H1").Value = Array("Category", "Seiban", "Worker", "When", "Before", "After", "Text before", "Text after")
    ws.Range("A1:H1").Font.Bold = True
    ws.Range("A:H").NumberFormat = "@"                 ' dates are written as text: Excel must not turn them into dates
    n = rowsOut.Count
    If n > 0 Then
        ReDim sk(1 To n): ReDim ix(1 To n)
        For i = 1 To n
            itm = rowsOut(i)
            sk(i) = itm(0)
            ix(i) = i
        Next i
        For i = 2 To n                                 ' insertion sort of the keys (Category, Seiban, Date, Worker)
            tmp = sk(i)
            tt = ix(i)
            j = i - 1
            Do While j >= 1
                If sk(j) <= tmp Then Exit Do
                sk(j + 1) = sk(j)
                ix(j + 1) = ix(j)
                j = j - 1
            Loop
            sk(j + 1) = tmp
            ix(j + 1) = tt
        Next i
        ReDim outArr(1 To n, 1 To 8)
        For i = 1 To n
            itm = rowsOut(ix(i))
            For j = 1 To 8
                outArr(i, j) = itm(j)
            Next j
        Next i
        ws.Range(ws.Cells(2, 1), ws.Cells(n + 1, 8)).Value = outArr
        For i = 1 To n
            Select Case outArr(i, 1)
                Case "Add": ws.Cells(i + 1, 1).Interior.Color = RGB(198, 239, 206)
                Case "Change": ws.Cells(i + 1, 1).Interior.Color = RGB(255, 235, 156)
                Case "Delete": ws.Cells(i + 1, 1).Interior.Color = RGB(255, 199, 206)
            End Select
        Next i
    End If
    ws.Columns("A:H").AutoFit
    If ws.Columns("E").ColumnWidth > 45 Then ws.Columns("E").ColumnWidth = 45
    If ws.Columns("F").ColumnWidth > 45 Then ws.Columns("F").ColumnWidth = 45
    If ws.Columns("G").ColumnWidth > 50 Then ws.Columns("G").ColumnWidth = 50
    If ws.Columns("H").ColumnWidth > 50 Then ws.Columns("H").ColumnWidth = 50
    ws.Range("A1:H1").AutoFilter
    WriteSummary = n
End Function

' worker + seiban pairs that have at least one changed cell (old text or new text)
Private Sub MarkAffected(info As Variant, chg() As String, re As Object, cache As Object, affected As Object)
    Dim rowKey As Object, key As Variant, r As Long, c As Long, lastCol As Long, parts() As String, t As Variant, pk As String
    Set rowKey = info(I_ROWKEY)
    lastCol = info(I_LASTCOL)
    For Each key In rowKey.Keys
        r = rowKey(key)
        For c = 1 To lastCol
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

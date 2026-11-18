Attribute VB_Name = "CompareSuOthers"
Option Explicit

Private Const SHEET_OLD As String = "Compare_Old"
Private Const SHEET_NEW As String = "Compare_New"
Private Const SHEET_SUM As String = "Compare_Summary"
Private Const SHEET_BTN As String = "Compare"

Private Type SheetData
    DateRow As Long
    FirstRow As Long
    LastRow As Long
    LastCol As Long
End Type

Private Function TaskSheetName() As String
    TaskSheetName = ChrW(&H4E88) & ChrW(&H5B9A) & ChrW(&H8868) & "_2026"
End Function

Public Sub CreateCompareButton()
    Dim ws As Worksheet, b As Object, i As Long
    
    On Error Resume Next
    Set ws = ThisWorkbook.Worksheets(SHEET_BTN)
    On Error GoTo 0
    
    If ws Is Nothing Then
        Set ws = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
        ws.Name = SHEET_BTN
    End If
    
    For i = ws.Shapes.Count To 1 Step -1
        If ws.Shapes(i).Type = msoFormControl Then ws.Shapes(i).Delete
    Next i
    
    Set b = ws.Buttons.Add(ws.Range("B2").Left, ws.Range("B2").Top, 240, 36)
    b.Caption = "Compare with old version"
    b.OnAction = "CompareWithOldVersion"
    
    ws.Range("B5").Value = "1. Click the button.  2. Choose the OLD SU_Others (the input of step 3-3)."
    ws.Range("B6").Value = "3. Three sheets are added: Compare_Old, Compare_New (with red bar outlines), Compare_Summary"
    
    MsgBox "Compare button created!", vbInformation
End Sub

Public Sub CompareWithOldVersion()
    On Error GoTo ErrHandler
    
    Dim oldPath As Variant, sheetNum As String, sheetIdx As Long, sheetName As String
    Dim wbOld As Workbook, wbNew As Workbook, secOld As MsoAutomationSecurity
    Dim wsOldSrc As Worksheet, wsNewSrc As Worksheet, wsOldCopy As Worksheet, wsNewCopy As Worksheet
    Dim dOld As SheetData, dNew As SheetData
    Dim chgOld() As String, chgNew() As String
    Dim nOutlines As Long, nDiff As Long
    
    oldPath = Application.GetOpenFilename("Excel (*.xlsm;*.xlsx),*.xlsm;*.xlsx")
    If VarType(oldPath) = vbBoolean Then Exit Sub
    If LCase$(CStr(oldPath)) = LCase$(ThisWorkbook.FullName) Then
        MsgBox "Choose the OLD file, not this workbook.", vbExclamation
        Exit Sub
    End If
    
    secOld = Application.AutomationSecurity
    Application.AutomationSecurity = msoAutomationSecurityForceDisable
    Set wbOld = Workbooks.Open(Filename:=CStr(oldPath), UpdateLinks:=0, ReadOnly:=True)
    Application.AutomationSecurity = secOld
    Set wbNew = ThisWorkbook
    
    ' Select sheet
    sheetNum = InputBox("Which sheet? (1-" & wbOld.Sheets.Count & ", default=3)", "Select", "3")
    If sheetNum = "" Then
        wbOld.Close SaveChanges:=False
        Exit Sub
    End If
    
    sheetIdx = CLng(sheetNum)
    If sheetIdx < 1 Or sheetIdx > wbOld.Sheets.Count Then
        MsgBox "Invalid sheet number", vbCritical
        wbOld.Close SaveChanges:=False
        Exit Sub
    End If
    
    sheetName = wbOld.Sheets(sheetIdx).Name
    Set wsOldSrc = wbOld.Sheets(sheetIdx)
    Set wsNewSrc = wbNew.Sheets(sheetName)
    
    If wsNewSrc Is Nothing Then
        MsgBox "Sheet '" & sheetName & "' not found in NEW file", vbCritical
        wbOld.Close SaveChanges:=False
        Exit Sub
    End If
    
    Application.ScreenUpdating = False
    Application.DisplayAlerts = False
    
    ' Load sheet data
    dOld = LoadSheetData(wsOldSrc)
    dNew = LoadSheetData(wsNewSrc)
    
    ' Compare cells
    ReDim chgOld(1 To dOld.LastRow, 1 To dOld.LastCol)
    ReDim chgNew(1 To dNew.LastRow, 1 To dNew.LastCol)
    
    CompareCells wsOldSrc, wsNewSrc, dOld, dNew, chgOld, chgNew, nDiff
    
    ' Delete old sheets
    On Error Resume Next
    wbNew.Sheets(SHEET_OLD).Delete
    wbNew.Sheets(SHEET_NEW).Delete
    wbNew.Sheets(SHEET_SUM).Delete
    On Error GoTo ErrHandler
    
    ' Copy sheets
    MsgBox "Copying sheets (please wait)...", vbInformation
    wsOldSrc.Copy After:=wbNew.Sheets(wbNew.Sheets.Count)
    wbNew.Sheets(wbNew.Sheets.Count).Name = SHEET_OLD
    Set wsOldCopy = wbNew.Sheets(SHEET_OLD)
    RemoveControls wsOldCopy
    
    wsNewSrc.Copy After:=wbNew.Sheets(wbNew.Sheets.Count)
    wbNew.Sheets(wbNew.Sheets.Count).Name = SHEET_NEW
    Set wsNewCopy = wbNew.Sheets(SHEET_NEW)
    RemoveControls wsNewCopy
    
    ' Outline runs of changed cells (bars, not individual cells)
    nOutlines = OutlineRuns(wsOldCopy, dOld, chgOld) + OutlineRuns(wsNewCopy, dNew, chgNew)
    
    ' Create summary
    Dim wsSum As Worksheet
    Set wsSum = wbNew.Worksheets.Add(After:=wbNew.Sheets(wbNew.Sheets.Count))
    wsSum.Name = SHEET_SUM
    WriteSummary wsSum, nDiff, nOutlines
    
    wbOld.Close SaveChanges:=False
    
    Application.ScreenUpdating = True
    Application.DisplayAlerts = True
    
    wsNewCopy.Activate
    MsgBox "Compare Done!" & vbCrLf & _
           "Changed cells: " & nDiff & vbCrLf & _
           "Red bar outlines: " & nOutlines, vbInformation
    Exit Sub
    
ErrHandler:
    Application.ScreenUpdating = True
    Application.DisplayAlerts = True
    MsgBox "ERROR: " & Err.Description, vbCritical
    On Error Resume Next
    If Not wbOld Is Nothing Then wbOld.Close SaveChanges:=False
End Sub

Private Function LoadSheetData(ws As Worksheet) As SheetData
    Dim d As SheetData, r As Long, c As Long, n As Long
    
    ' Find date row
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
    
    If d.DateRow = 0 Then d.DateRow = 3
    
    d.FirstRow = d.DateRow + 2
    d.LastRow = ws.Cells(ws.Rows.Count, 2).End(xlUp).Row
    d.LastCol = ws.Cells(d.DateRow, ws.Columns.Count).End(xlToLeft).Column
    
    LoadSheetData = d
End Function

Private Sub CompareCells(wsOld As Worksheet, wsNew As Worksheet, dOld As SheetData, dNew As SheetData, _
                         chgOld() As String, chgNew() As String, ByRef nDiff As Long)
    Dim r As Long, c As Long
    Dim oldVal As String, newVal As String
    
    For r = 1 To Application.WorksheetFunction.Max(dOld.LastRow, dNew.LastRow)
        For c = 1 To Application.WorksheetFunction.Max(dOld.LastCol, dNew.LastCol)
            On Error Resume Next
            oldVal = CStr(wsOld.Cells(r, c).Value)
            newVal = CStr(wsNew.Cells(r, c).Value)
            On Error GoTo 0
            
            If oldVal <> newVal Then
                chgOld(r, c) = oldVal & Chr(1) & newVal
                chgNew(r, c) = oldVal & Chr(1) & newVal
                nDiff = nDiff + 1
            End If
        Next c
    Next r
End Sub

Private Function OutlineRuns(ws As Worksheet, d As SheetData, chg() As String) As Long
    Dim r As Long, c As Long, c2 As Long, n As Long
    
    For r = d.FirstRow To d.LastRow
        c = 1
        Do While c <= d.LastCol
            If Len(chg(r, c)) > 0 Then
                c2 = c
                ' Find consecutive changed cells
                Do While c2 < d.LastCol
                    If chg(r, c2 + 1) = chg(r, c) Then c2 = c2 + 1 Else Exit Do
                Loop
                ' Thicker red border around the run (xlThick = thicker than xlMedium)
                ws.Range(ws.Cells(r, c), ws.Cells(r, c2)).BorderAround LineStyle:=xlContinuous, Weight:=xlThick, Color:=RGB(255, 0, 0)
                n = n + 1
                c = c2 + 1
            Else
                c = c + 1
            End If
        Loop
    Next r
    OutlineRuns = n
End Function

Private Sub WriteSummary(ws As Worksheet, nDiff As Long, nOutlines As Long)
    ws.Range("A1").Value = "Comparison Summary"
    ws.Range("A2").Value = "Changed cells"
    ws.Range("B2").Value = nDiff
    ws.Range("A3").Value = "Red bar outlines"
    ws.Range("B3").Value = nOutlines
    ws.Range("A5").Value = "Details:"
    ws.Range("A6").Value = "- Compare_Old: original file (before rewrite)"
    ws.Range("A7").Value = "- Compare_New: rewritten file (with red outlines on changes)"
    ws.Range("A8").Value = "- Red outline = continuous bar of changed/added cells"
    
    ws.Range("A1").Font.Bold = True
    ws.Range("A1").Font.Size = 14
    ws.Columns("A").ColumnWidth = 35
    ws.Columns("B").ColumnWidth = 15
End Sub

Private Sub RemoveControls(ws As Worksheet)
    Dim i As Long
    On Error Resume Next
    For i = ws.Shapes.Count To 1 Step -1
        If ws.Shapes(i).Type = msoFormControl Then ws.Shapes(i).Delete
    Next i
    On Error GoTo 0
End Sub
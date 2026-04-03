# Excel Keyword Search (VBA) - Quick Guide  
  
## Purpose  
Search a keyword (e.g. task ID like `1001A`) in the Schedule sheet and:  
- Create a new sheet with only matching workers  
- Highlight the continuous working period (bar) with red border  
  
---  
  
## Setup (One-time only)  
  
1. Open Excel file  
2. Save as:  
   - **.xlsm (Macro-enabled workbook)**    
   OR    
   - **.xlsb (Binary workbook, also supports macro)**  
  
3. Open VBA editor:  
   - Press `Alt + F11`  
  
4. Insert module:  
   - Right click workbook → Insert → Module  
  
5. Paste VBA code (the one provided)  
  
6. Close editor  
  
---  
  
## Add Button (optional but recommended)  
  
1. Go to **Developer tab**  
2. Click **Insert → Button**  
3. Place button on `Schedule` sheet  
4. Assign macro:  
   - `SearchKeywordAndCreateSheet`  
  
---  
  
## How to Use  
  
1. Go to sheet: `Schedule`  
2. Input keyword in:  
   - Cell **B1**  
   - Example: `1001A`  
  
3. Run macro:  
   - Click button    
   OR    
   - Press `Alt + F8` → run `SearchKeywordAndCreateSheet`  
  
---  
  
## Result  
  
- New sheet created:

Search_1001A

  
- Contains:  
- Only workers that have that task  
- Full row data preserved  
- Same colors as original  
  
- Highlight:  
- Continuous working period (e.g. 01–10)  
- Red border around the whole block (not cell-by-cell)  
  
---  
  
## Notes  
  
- Search range:

F4 → SC808

  
- Search is:  
- Case-insensitive  
  
- If same sheet exists:  
- It will be overwritten  
  
- Must click:

Enable Content (コンテンツの有効化)

when opening file  
  
---  

## Code
```
Option Explicit

Sub SearchKeywordAndCreateSheet()

    Dim srcWs As Worksheet
    Dim outWs As Worksheet
    Dim keyword As String
    Dim keywordCell As Range
    Dim outName As String
    Dim lastOutRow As Long
    Dim r As Long, c As Long
    Dim hasMatch As Boolean
    Dim firstDataRow As Long, lastDataRow As Long
    Dim firstDataCol As Long, lastDataCol As Long
    Dim outRow As Long
    
    Set srcWs = ThisWorkbook.Worksheets("Schedule")
    
    ' ===== keyword input cell =====
    Set keywordCell = srcWs.Range("B1")
    keyword = Trim(CStr(keywordCell.Value))
    
    If keyword = "" Then
        MsgBox "Please enter a keyword in cell B1.", vbExclamation
        Exit Sub
    End If
    
    ' Search area: F4:SC808
    firstDataRow = 4
    lastDataRow = 808
    firstDataCol = srcWs.Range("F1").Column
    lastDataCol = srcWs.Range("SC1").Column
    
    outName = "Search_" & CleanSheetName(keyword)
    If Len(outName) > 31 Then outName = Left(outName, 31)
    
    Application.ScreenUpdating = False
    Application.DisplayAlerts = False
    
    ' Delete old result sheet if exists
    On Error Resume Next
    ThisWorkbook.Worksheets(outName).Delete
    On Error GoTo 0
    
    Application.DisplayAlerts = True
    
    ' Create output sheet
    Set outWs = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Worksheets(ThisWorkbook.Worksheets.Count))
    outWs.Name = outName
    
    ' Copy header area A1:SC3
    srcWs.Range("A1:SC3").Copy Destination:=outWs.Range("A1")
    
    ' Copy column widths
    For c = 1 To lastDataCol
        outWs.Columns(c).ColumnWidth = srcWs.Columns(c).ColumnWidth
    Next c
    
    ' Copy row heights for header
    outWs.Rows(1).RowHeight = srcWs.Rows(1).RowHeight
    outWs.Rows(2).RowHeight = srcWs.Rows(2).RowHeight
    outWs.Rows(3).RowHeight = srcWs.Rows(3).RowHeight
    
    ' Freeze panes same style
    outWs.Activate
    outWs.Range("F4").Select
    ActiveWindow.FreezePanes = True
    
    ' Autofilter
    outWs.Range("A3:SC3").AutoFilter
    
    outRow = 4
    
    ' Copy only matching rows
    For r = firstDataRow To lastDataRow
        hasMatch = RowContainsKeyword(srcWs, r, firstDataCol, lastDataCol, keyword)
        
        If hasMatch Then
            srcWs.Range("A" & r & ":SC" & r).Copy Destination:=outWs.Range("A" & outRow)
            outWs.Rows(outRow).RowHeight = srcWs.Rows(r).RowHeight
            
            ' Draw red outline around each continuous matching bar
            Call OutlineKeywordRuns(outWs, outRow, firstDataCol, lastDataCol, keyword)
            
            outRow = outRow + 1
        End If
    Next r
    
    lastOutRow = outWs.Cells(outWs.Rows.Count, "A").End(xlUp).Row
    
    If lastOutRow < 4 Then
        MsgBox "No rows found for keyword: " & keyword, vbInformation
    Else
        outWs.Range("A3:SC" & lastOutRow).AutoFilter
        MsgBox "Created sheet: " & outName, vbInformation
    End If
    
    Application.ScreenUpdating = True

End Sub


Function RowContainsKeyword(ws As Worksheet, rowNum As Long, firstCol As Long, lastCol As Long, keyword As String) As Boolean
    Dim c As Long
    Dim v As String
    
    For c = firstCol To lastCol
        v = CStr(ws.Cells(rowNum, c).Value)
        If InStr(1, v, keyword, vbTextCompare) > 0 Then
            RowContainsKeyword = True
            Exit Function
        End If
    Next c
    
    RowContainsKeyword = False
End Function


Sub OutlineKeywordRuns(ws As Worksheet, rowNum As Long, firstCol As Long, lastCol As Long, keyword As String)

    Dim c As Long
    Dim runStart As Long
    Dim inRun As Boolean
    Dim v As String
    
    inRun = False
    runStart = 0
    
    For c = firstCol To lastCol
        v = CStr(ws.Cells(rowNum, c).Value)
        
        If InStr(1, v, keyword, vbTextCompare) > 0 Then
            If Not inRun Then
                inRun = True
                runStart = c
            End If
        Else
            If inRun Then
                Call DrawRunBorder(ws, rowNum, runStart, c - 1)
                inRun = False
                runStart = 0
            End If
        End If
    Next c
    
    If inRun Then
        Call DrawRunBorder(ws, rowNum, runStart, lastCol)
    End If

End Sub


Sub DrawRunBorder(ws As Worksheet, rowNum As Long, startCol As Long, endCol As Long)

    Dim rng As Range
    Set rng = ws.Range(ws.Cells(rowNum, startCol), ws.Cells(rowNum, endCol))
    
    With rng.Borders(xlEdgeLeft)
        .LineStyle = xlContinuous
        .Color = RGB(255, 0, 0)
        .Weight = xlMedium
    End With
    
    With rng.Borders(xlEdgeTop)
        .LineStyle = xlContinuous
        .Color = RGB(255, 0, 0)
        .Weight = xlMedium
    End With
    
    With rng.Borders(xlEdgeBottom)
        .LineStyle = xlContinuous
        .Color = RGB(255, 0, 0)
        .Weight = xlMedium
    End With
    
    With rng.Borders(xlEdgeRight)
        .LineStyle = xlContinuous
        .Color = RGB(255, 0, 0)
        .Weight = xlMedium
    End With

End Sub


Function CleanSheetName(ByVal s As String) As String
    Dim badChars As Variant
    Dim i As Long
    
    badChars = Array("/", "\", "[", "]", "*", "?", ":", "'")
    
    For i = LBound(badChars) To UBound(badChars)
        s = Replace(s, badChars(i), "_")
    Next i
    
    CleanSheetName = s
End Function
```
---
  
## When to Run  
  
- Every time you want to search a new keyword  
- No need to rerun Python  
  
---  
  
## Troubleshooting  
  
| Problem | Fix |  
|--------|-----|  
| Button not working | Enable macro |  
| No result | Check keyword spelling |  
| Macro missing | Reopen .xlsm file |  
  
---
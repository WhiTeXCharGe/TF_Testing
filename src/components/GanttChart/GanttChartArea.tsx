import { useRef, useCallback } from 'react';
import { CELL_WIDTH, ROW_HEIGHT, ROW_HEADER_WIDTH, SHOW_WEEKEND_SHADING } from '../../config/appConfig';
import { formatDateShort, isWeekend, diffDays, addDays } from '../../utils/dateUtils';
import { UI } from '../../config/uiText';

const NAVY = '#1c2b3a';
const NAVY_MID = '#243447';
const NAVY_LIGHT = '#2c3d52';
const NAVY_TEXT = '#d0dce8';
const EDGE_PX = 8;

export interface GanttBar {
  id: string;
  assignmentIndex: number;
  label: string;
  startDate: string;
  endDate: string;
  color: string;
  isSelected: boolean;
  hasViolation: boolean;
}

export interface GanttRow {
  id: string;
  label: string;
  indent: number;      // 0=device header, 1=phase/summary, 2=slot
  isHeader?: boolean;  // dark header row (device level)
  isExpandable: boolean;
  isExpanded: boolean;
  bars: GanttBar[];
}

interface DragState {
  mode: 'move' | 'resize-start' | 'resize-end';
  assignmentIndex: number;
  originalStart: string;
  originalEnd: string;
  startX: number;
  barEl: HTMLDivElement;
  barInitialLeft: number;
  barInitialWidth: number;
}

interface Props {
  rows: GanttRow[];
  dates: string[];
  onToggleRow: (id: string) => void;
  onBarClick: (assignmentIndex: number) => void;
  onBarDragEnd: (assignmentIndex: number, newStart: string, newEnd: string) => void;
  isDragDisabled?: boolean;
}

export function GanttChartArea({ rows, dates, onToggleRow, onBarClick, onBarDragEnd, isDragDisabled }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const leftBodyRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<DragState | null>(null);

  const handleScroll = useCallback(() => {
    if (leftBodyRef.current && scrollRef.current) {
      leftBodyRef.current.scrollTop = scrollRef.current.scrollTop;
    }
  }, []);

  const handleBarMouseDown = useCallback(
    (e: React.MouseEvent<HTMLDivElement>, bar: GanttBar) => {
      if (isDragDisabled) {
        onBarClick(bar.assignmentIndex);
        return;
      }

      e.preventDefault();
      e.stopPropagation();

      const barEl = e.currentTarget;
      const barRect = barEl.getBoundingClientRect();
      const offsetX = e.clientX - barRect.left;
      const mode: DragState['mode'] =
        offsetX <= EDGE_PX ? 'resize-start' :
        offsetX >= barRect.width - EDGE_PX ? 'resize-end' :
        'move';

      dragRef.current = {
        mode,
        assignmentIndex: bar.assignmentIndex,
        originalStart: bar.startDate,
        originalEnd: bar.endDate,
        startX: e.clientX,
        barEl,
        barInitialLeft: barEl.offsetLeft,
        barInitialWidth: barEl.offsetWidth,
      };

      barEl.style.cursor = mode === 'move' ? 'grabbing' : 'ew-resize';
      barEl.style.zIndex = '20';
      barEl.style.opacity = '0.85';

      const onMouseMove = (me: MouseEvent) => {
        const d = dragRef.current;
        if (!d) return;
        const dx = me.clientX - d.startX;

        if (d.mode === 'move') {
          d.barEl.style.transform = `translateX(${dx}px)`;
        } else if (d.mode === 'resize-start') {
          const clamped = Math.min(dx, d.barInitialWidth - CELL_WIDTH);
          d.barEl.style.left = `${d.barInitialLeft + clamped}px`;
          d.barEl.style.width = `${d.barInitialWidth - clamped}px`;
        } else {
          const newW = Math.max(CELL_WIDTH, d.barInitialWidth + dx);
          d.barEl.style.width = `${newW}px`;
        }
      };

      const onMouseUp = (me: MouseEvent) => {
        const d = dragRef.current;
        if (!d) return;

        // Reset DOM styles — React will re-render with correct values
        d.barEl.style.transform = '';
        d.barEl.style.left = `${d.barInitialLeft}px`;
        d.barEl.style.width = `${d.barInitialWidth}px`;
        d.barEl.style.cursor = '';
        d.barEl.style.zIndex = '';
        d.barEl.style.opacity = '';

        const dx = me.clientX - d.startX;
        const daysDelta = Math.round(dx / CELL_WIDTH);

        if (daysDelta === 0 && d.mode === 'move') {
          onBarClick(d.assignmentIndex);
        } else if (daysDelta !== 0) {
          if (d.mode === 'move') {
            onBarDragEnd(d.assignmentIndex, addDays(d.originalStart, daysDelta), addDays(d.originalEnd, daysDelta));
          } else if (d.mode === 'resize-start') {
            const newStart = addDays(d.originalStart, daysDelta);
            if (newStart < d.originalEnd) {
              onBarDragEnd(d.assignmentIndex, newStart, d.originalEnd);
            }
          } else {
            const newEnd = addDays(d.originalEnd, daysDelta);
            if (newEnd > d.originalStart) {
              onBarDragEnd(d.assignmentIndex, d.originalStart, newEnd);
            }
          }
        }

        dragRef.current = null;
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
      };

      window.addEventListener('mousemove', onMouseMove);
      window.addEventListener('mouseup', onMouseUp);
    },
    [onBarClick, onBarDragEnd, isDragDisabled],
  );

  const handleBarMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (isDragDisabled || dragRef.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ox = e.clientX - rect.left;
    e.currentTarget.style.cursor =
      ox <= EDGE_PX ? 'w-resize' :
      ox >= rect.width - EDGE_PX ? 'e-resize' :
      'grab';
  }, [isDragDisabled]);

  if (dates.length === 0) return null;
  const viewStart = dates[0];
  const totalWidth = dates.length * CELL_WIDTH;

  return (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
      {/* ── Left: row headers ─────────────────────────────────────── */}
      <div
        style={{
          width: ROW_HEADER_WIDTH,
          flexShrink: 0,
          borderRight: '2px solid #0a1520',
          backgroundColor: NAVY,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {/* Fixed header — matches date header height */}
        <div style={{
          height: ROW_HEIGHT,
          flexShrink: 0,
          borderBottom: '1px solid #0a1520',
          backgroundColor: '#0f1e2b',
          display: 'flex',
          alignItems: 'center',
          paddingLeft: 8,
        }}>
          <span style={{ fontSize: 10, color: '#7a9bb5', fontFamily: 'MS Gothic, monospace' }}>{UI.deviceCornerLabel}</span>
        </div>

        {/* Scrollable body — overflow hidden, scrollTop driven by right panel */}
        <div ref={leftBodyRef} style={{ flex: 1, overflow: 'hidden' }}>
          {rows.map(row => {
            const bg = row.indent === 0
              ? (row.isHeader ? '#162433' : NAVY)
              : row.indent === 1
              ? NAVY_MID
              : NAVY_LIGHT;

            return (
              <div
                key={row.id}
                style={{
                  height: ROW_HEIGHT,
                  display: 'flex',
                  alignItems: 'center',
                  paddingLeft: 8 + row.indent * 14,
                  borderBottom: '1px solid #0a1520',
                  backgroundColor: bg,
                  fontSize: row.indent === 0 ? 12 : 11,
                  fontFamily: 'MS Gothic, monospace',
                  color: NAVY_TEXT,
                  userSelect: 'none',
                  cursor: row.isExpandable ? 'pointer' : 'default',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
                onClick={() => row.isExpandable && onToggleRow(row.id)}
                title={row.label}
              >
                {row.isExpandable && (
                  <span style={{ marginRight: 5, fontSize: 13, color: '#7ab0d0', flexShrink: 0, fontWeight: 'bold' }}>
                    {row.isExpanded ? '−' : '+'}
                  </span>
                )}
                {row.indent > 0 && !row.isExpandable && (
                  <span style={{ marginRight: 4, color: '#4a6a85', flexShrink: 0 }}>▸</span>
                )}
                <span>{row.label}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Right: scrollable Gantt area ──────────────────────────── */}
      <div ref={scrollRef} onScroll={handleScroll} style={{ flex: 1, overflow: 'auto' }}>
        {/* Date header */}
        <div
          style={{
            display: 'flex',
            height: ROW_HEIGHT,
            position: 'sticky',
            top: 0,
            backgroundColor: '#1a2e3f',
            borderBottom: '2px solid #0a1520',
            zIndex: 10,
            minWidth: totalWidth,
          }}
        >
          {dates.map(d => {
            const weekend = isWeekend(d);
            return (
              <div
                key={d}
                style={{
                  width: CELL_WIDTH,
                  flexShrink: 0,
                  borderRight: '1px solid #243447',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: 10,
                  fontFamily: 'MS Gothic, monospace',
                  color: weekend ? '#7ab0d0' : '#a0bccc',
                  backgroundColor: SHOW_WEEKEND_SHADING && weekend ? 'rgba(100,150,200,0.15)' : undefined,
                }}
              >
                {formatDateShort(d)}
              </div>
            );
          })}
        </div>

        {/* Gantt rows */}
        {rows.map(row => {
          const rowBg = row.indent === 0
            ? (row.isHeader ? '#f0f4f8' : '#f4f6f9')
            : row.indent === 1
            ? '#f8f9fb'
            : '#ffffff';

          return (
            <div
              key={row.id}
              style={{
                position: 'relative',
                height: ROW_HEIGHT,
                borderBottom: '1px solid #e2e8f0',
                backgroundColor: rowBg,
                minWidth: totalWidth,
              }}
            >
              {/* Column grid lines */}
              {dates.map((d, di) => {
                const weekend = isWeekend(d);
                return (
                  <div
                    key={d}
                    style={{
                      position: 'absolute',
                      left: di * CELL_WIDTH,
                      top: 0,
                      width: CELL_WIDTH,
                      height: ROW_HEIGHT,
                      borderRight: '1px solid #e8edf2',
                      backgroundColor: SHOW_WEEKEND_SHADING && weekend ? 'rgba(200,210,230,0.25)' : undefined,
                      pointerEvents: 'none',
                    }}
                  />
                );
              })}

              {/* Bars */}
              {row.bars.map(bar => {
                const leftDays = diffDays(viewStart, bar.startDate);
                const widthDays = Math.max(1, diffDays(bar.startDate, bar.endDate) + 1);
                if (leftDays + widthDays < 0 || leftDays > dates.length) return null;

                const left = leftDays * CELL_WIDTH + 1;
                const width = widthDays * CELL_WIDTH - 2;

                return (
                  <div
                    key={bar.id}
                    onMouseDown={e => bar.assignmentIndex >= 0 && handleBarMouseDown(e, bar)}
                    onMouseMove={handleBarMouseMove}
                    onMouseLeave={e => { e.currentTarget.style.cursor = ''; }}
                    style={{
                      position: 'absolute',
                      left,
                      width,
                      top: 5,
                      height: ROW_HEIGHT - 10,
                      backgroundColor: bar.color,
                      borderRadius: 4,
                      cursor: bar.assignmentIndex >= 0 && !isDragDisabled ? 'grab' : 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      paddingLeft: 6,
                      paddingRight: 6,
                      fontSize: 11,
                      fontFamily: 'MS Gothic, monospace',
                      color: '#fff',
                      overflow: 'hidden',
                      whiteSpace: 'nowrap',
                      boxSizing: 'border-box',
                      outline: bar.isSelected
                        ? '2px solid #1565c0'
                        : bar.hasViolation
                        ? '2px solid #c62828'
                        : 'none',
                      outlineOffset: bar.isSelected ? 1 : 0,
                      zIndex: bar.isSelected ? 5 : 1,
                      boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
                      textShadow: '0 1px 2px rgba(0,0,0,0.4)',
                    }}
                    title={`${bar.label}\n${bar.startDate} → ${bar.endDate}`}
                  >
                    {/* Left resize handle hint */}
                    {!isDragDisabled && (
                      <div style={{
                        position: 'absolute', left: 0, top: 0, width: EDGE_PX, height: '100%',
                        cursor: 'w-resize', borderRadius: '4px 0 0 4px',
                      }} />
                    )}
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', pointerEvents: 'none' }}>
                      {bar.label}
                    </span>
                    {/* Right resize handle hint */}
                    {!isDragDisabled && (
                      <div style={{
                        position: 'absolute', right: 0, top: 0, width: EDGE_PX, height: '100%',
                        cursor: 'e-resize', borderRadius: '0 4px 4px 0',
                      }} />
                    )}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}
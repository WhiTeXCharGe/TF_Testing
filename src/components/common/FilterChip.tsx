import { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { SelectOption } from './SearchableSelect';
import { filterBarStyles as S } from '../../styles/toolbar';
import { UI } from '../../config/uiText';

interface Props {
  label: string;
  options: SelectOption[];
  selected: string[];
  onChange: (values: string[]) => void;
}

export function FilterChip({ label, options, selected, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const btnRef = useRef<HTMLButtonElement>(null);
  const dropRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const active = selected.length > 0;
  const chipLabel = active
    ? selected.length === 1
      ? (options.find(o => o.value === selected[0])?.label ?? selected[0])
      : UI.filterItemCount(selected.length)
    : label;

  const filtered = search
    ? options.filter(o => o.label.toLowerCase().includes(search.toLowerCase()))
    : options;

  // Position dropdown below button
  const [pos, setPos] = useState({ top: 0, left: 0 });

  const openDropdown = () => {
    if (!btnRef.current) return;
    const r = btnRef.current.getBoundingClientRect();
    setPos({ top: r.bottom + 4, left: r.left });
    setOpen(true);
    setSearch('');
    setTimeout(() => searchRef.current?.focus(), 30);
  };

  const toggle = (value: string) => {
    onChange(
      selected.includes(value)
        ? selected.filter(v => v !== value)
        : [...selected, value],
    );
  };

  const clear = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onChange([]);
  }, [onChange]);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (
        !btnRef.current?.contains(e.target as Node) &&
        !dropRef.current?.contains(e.target as Node)
      ) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  return (
    <>
      <button
        ref={btnRef}
        onClick={() => (open ? setOpen(false) : openDropdown())}
        style={S.chip(active)}
      >
        {chipLabel}
        {active ? (
          <span style={S.chipClear} onClick={clear} title={UI.filterClear}>×</span>
        ) : (
          <span style={{ fontSize: 10, opacity: 0.6 }}>▾</span>
        )}
      </button>

      {open && createPortal(
        <div ref={dropRef} style={{ ...S.dropdown, top: pos.top, left: pos.left }}>
          <input
            ref={searchRef}
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder={UI.filterSearchPlaceholder}
            style={S.dropdownSearch}
          />
          <div style={S.dropdownList}>
            {filtered.length === 0 ? (
              <div style={S.dropdownEmpty}>{UI.filterNoValues}</div>
            ) : (
              filtered.map(opt => {
                const checked = selected.includes(opt.value);
                return (
                  <div
                    key={opt.value}
                    style={S.dropdownItem(checked)}
                    onClick={() => toggle(opt.value)}
                  >
                    <input
                      type="checkbox"
                      readOnly
                      checked={checked}
                      style={{ margin: 0, flexShrink: 0, cursor: 'pointer' }}
                    />
                    <span>{opt.label}</span>
                    {opt.sub && <span style={{ color: '#888', fontSize: 10, marginLeft: 'auto' }}>{opt.sub}</span>}
                  </div>
                );
              })
            )}
          </div>
          {selected.length > 0 && (
            <div
              style={{ padding: '5px 10px', borderTop: '1px solid #eee', fontSize: 11, color: '#1565c0', cursor: 'pointer', textAlign: 'center', fontWeight: 700 }}
              onClick={() => onChange([])}
            >
              {UI.filterClearAll}
            </div>
          )}
        </div>,
        document.body,
      )}
    </>
  );
}

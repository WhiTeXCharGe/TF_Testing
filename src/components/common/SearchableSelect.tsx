import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

export interface SelectOption {
  value: string;
  label: string;
  sub?: string;
}

interface Props {
  value: string;
  options: SelectOption[];
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  style?: React.CSSProperties;
}

export function SearchableSelect({ value, options, onChange, placeholder = '---', disabled = false, style }: Props) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const triggerRef = useRef<HTMLButtonElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const [pos, setPos] = useState({ left: 0, top: 0, width: 0 });

  const selected = options.find(o => o.value === value);

  const filtered = useMemo(() => {
    if (!search.trim()) return options;
    const q = search.toLowerCase();
    return options.filter(o =>
      o.label.toLowerCase().includes(q) || (o.sub ?? '').toLowerCase().includes(q),
    );
  }, [options, search]);

  useEffect(() => {
    if (!open || !triggerRef.current) return;
    const rect = triggerRef.current.getBoundingClientRect();
    const dropW = Math.max(rect.width, 240);
    let left = rect.left;
    if (left + dropW > window.innerWidth - 8) left = window.innerWidth - dropW - 8;
    const dropH = 280;
    const top = rect.bottom + window.scrollY + 2 > window.innerHeight - dropH
      ? rect.top + window.scrollY - dropH - 2
      : rect.bottom + window.scrollY + 2;
    setPos({ left, top, width: dropW });
    setTimeout(() => searchRef.current?.focus(), 10);

    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      const portal = document.getElementById('searchable-select-portal');
      if (!portal?.contains(t) && !triggerRef.current?.contains(t)) {
        setOpen(false);
        setSearch('');
      }
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open]);

  const triggerStyle: React.CSSProperties = {
    width: '100%',
    padding: '4px 24px 4px 6px',
    border: '1px solid #ccc',
    borderRadius: 3,
    fontSize: 11,
    fontFamily: 'MS Gothic, monospace',
    backgroundColor: disabled ? '#f5f5f5' : '#fff',
    color: disabled ? '#aaa' : value ? '#1c2b3a' : '#999',
    cursor: disabled ? 'default' : 'pointer',
    textAlign: 'left',
    position: 'relative',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    boxSizing: 'border-box',
    ...style,
  };

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        disabled={disabled}
        onClick={() => { if (!disabled) setOpen(o => !o); }}
        style={triggerStyle}
        title={selected?.label ?? placeholder}
      >
        {selected ? (
          <span>{selected.label}{selected.sub ? <span style={{ color: '#888', marginLeft: 4 }}>{selected.sub}</span> : null}</span>
        ) : (
          placeholder
        )}
        <span style={{ position: 'absolute', right: 5, top: '50%', transform: 'translateY(-50%)', fontSize: 9, color: '#999', pointerEvents: 'none' }}>▼</span>
      </button>

      {open && createPortal(
        <div
          id="searchable-select-portal"
          style={{
            position: 'fixed',
            left: pos.left,
            top: pos.top,
            width: pos.width,
            zIndex: 9999,
            backgroundColor: '#fff',
            border: '1px solid #c6d3e2',
            borderRadius: 5,
            boxShadow: '0 6px 20px rgba(0,0,0,0.15)',
            overflow: 'hidden',
            fontFamily: 'MS Gothic, monospace',
            fontSize: 11,
          }}
        >
          <div style={{ padding: '6px 8px', borderBottom: '1px solid #eee' }}>
            <input
              ref={searchRef}
              type="text"
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="検索..."
              style={{
                width: '100%',
                padding: '4px 6px',
                border: '1px solid #c0d0e0',
                borderRadius: 3,
                fontSize: 11,
                fontFamily: 'MS Gothic, monospace',
                boxSizing: 'border-box',
                outline: 'none',
              }}
            />
          </div>
          <div style={{ maxHeight: 220, overflowY: 'auto' }}>
            <div
              style={{ padding: '5px 10px', cursor: 'pointer', color: '#999' }}
              onMouseDown={() => { onChange(''); setOpen(false); setSearch(''); }}
            >
              {placeholder}
            </div>
            {filtered.map(o => (
              <div
                key={o.value}
                style={{
                  padding: '5px 10px',
                  cursor: 'pointer',
                  backgroundColor: o.value === value ? '#e8f0fb' : 'transparent',
                  color: '#1c2b3a',
                  display: 'flex',
                  gap: 6,
                  alignItems: 'baseline',
                }}
                onMouseDown={() => { onChange(o.value); setOpen(false); setSearch(''); }}
                onMouseEnter={e => (e.currentTarget.style.backgroundColor = o.value === value ? '#e8f0fb' : '#f5f8ff')}
                onMouseLeave={e => (e.currentTarget.style.backgroundColor = o.value === value ? '#e8f0fb' : 'transparent')}
              >
                <span>{o.label}</span>
                {o.sub && <span style={{ color: '#888', fontSize: 10 }}>{o.sub}</span>}
              </div>
            ))}
            {filtered.length === 0 && (
              <div style={{ padding: '8px 10px', color: '#aaa', textAlign: 'center' }}>該当なし</div>
            )}
          </div>
        </div>,
        document.body,
      )}
    </>
  );
}

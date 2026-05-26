/**
 * Simple modal dialog. One title, body, and a row of buttons.
 * Used for the Gantt-editor placeholder, "result not ready" notice,
 * and the Delete confirm prompt.
 */
import type { ReactNode } from 'react';
import { Icon } from './Icon';

export interface DialogButton {
  label: string;
  onClick: () => void;
  variant?: 'primary' | 'secondary' | 'danger';
}

interface DialogProps {
  title: string;
  body: ReactNode;
  buttons: DialogButton[];
  onClose: () => void;
}

const VARIANT_CLASS: Record<NonNullable<DialogButton['variant']>, string> = {
  primary:   'btn btn-primary btn-sm',
  secondary: 'btn btn-secondary btn-sm',
  danger:    'btn btn-danger btn-sm',
};

export function Dialog({ title, body, buttons, onClose }: DialogProps) {
  return (
    <div className="modal-overlay" onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="modal">
        <div className="modal-hd">
          <div className="modal-title">{title}</div>
          <button className="modal-close" onClick={onClose}><Icon name="x" size={16} /></button>
        </div>
        <div className="modal-body" style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-sec)', lineHeight: 1.55 }}>
          {body}
        </div>
        <div className="modal-footer">
          {buttons.map((b, i) => (
            <button key={i} className={VARIANT_CLASS[b.variant ?? 'secondary']} onClick={b.onClick}>
              {b.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

import { UI } from '@/config/uiConfig';

export function Topbar() {
  return (
    <header className="topbar">
      <div className="topbar-brand">
        <div className="topbar-title">{UI.app.title}</div>
        <div className="topbar-sub">{UI.app.subtitle}</div>
      </div>
    </header>
  );
}

import { NavLink } from 'react-router-dom';
import { UI } from '@/config/uiConfig';
import { Icon } from '@/components/common/Icon';

export function Sidebar() {
  return (
    <nav className="sidebar">
      <NavLink to="/" end className={({ isActive }) => 'sidebar-item' + (isActive ? ' active' : '')}>
        <Icon name="dataset" size={16} />
        {UI.sidebar.runLog}
      </NavLink>
      <NavLink to="/settings" className={({ isActive }) => 'sidebar-item' + (isActive ? ' active' : '')}>
        <Icon name="settings" size={16} />
        {UI.sidebar.settings}
      </NavLink>
    </nav>
  );
}

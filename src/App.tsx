import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from '@/components/layout/Layout';
import { RunLogPage } from '@/pages/RunLogPage';
import { GanttPage }  from '@/pages/GanttPage';
import { SettingsPage } from '@/pages/SettingsPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index                       element={<RunLogPage />} />
          <Route path="/gantt/:runId"         element={<GanttPage />} />
          <Route path="/gantt/:runId/:view"   element={<GanttPage />} />
          <Route path="/settings"             element={<SettingsPage />} />
          <Route path="*"                     element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

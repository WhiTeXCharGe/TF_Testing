import { AppProvider } from './context/AppContext';
import { GanttPage } from './pages/GanttPage';
import { ErrorDialog } from './components/Dialogs/ErrorDialog';
import { TaskAddDialog } from './components/Dialogs/TaskAddDialog';
import { NewScheduleDialog } from './components/Dialogs/NewScheduleDialog';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { useConstraintCheck } from './hooks/useConstraintCheck';

function AppContent() {
  useKeyboardShortcuts();
  useConstraintCheck();
  return (
    <>
      <GanttPage />
      <TaskAddDialog />
      <NewScheduleDialog />
      <ErrorDialog />
    </>
  );
}

export default function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}

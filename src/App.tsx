import { AppProvider } from './context/AppContext';
import { GanttPage } from './pages/GanttPage';
import { ErrorDialog } from './components/Dialogs/ErrorDialog';
import { TaskAddDialog } from './components/Dialogs/TaskAddDialog';
import { NewScheduleDialog } from './components/Dialogs/NewScheduleDialog';
import { ConstraintResultDialog } from './components/Dialogs/ConstraintResultDialog';
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
      <ConstraintResultDialog />
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

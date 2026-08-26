import { useEffect } from 'react';
import { useAppContext } from '../context/AppContext';
import { overwriteSaveFiles } from '../services/fileService';
import { UI } from '../config/uiText';

export function useKeyboardShortcuts() {
  const { state, dispatch } = useAppContext();

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key.toLowerCase()) {
          case 'z':
            e.preventDefault();
            dispatch({ type: 'UNDO' });
            break;
          case 'y':
            e.preventDefault();
            dispatch({ type: 'REDO' });
            break;
          case 's':
            e.preventDefault();
            if (!e.shiftKey && state.schedule && state.envConfig && state.currentEnvPath && state.currentSchedulePath) {
              overwriteSaveFiles(state.envConfig, state.schedule, state.currentEnvPath, state.currentSchedulePath)
                .catch(err => console.error('Save failed:', err));
            }
            break;
          case 'o':
            e.preventDefault();
            if (!state.session) dispatch({ type: 'OPEN_FILE_DIALOG' });
            break;
        }
      }
      if (e.key === 'Delete' && state.selectedAssignmentIndex !== null) {
        if (window.confirm(UI.deleteConfirm)) {
          dispatch({ type: 'DELETE_ASSIGNMENT', payload: state.selectedAssignmentIndex });
        }
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [state.schedule, state.envConfig, state.selectedAssignmentIndex, state.currentSchedulePath, state.currentEnvPath, state.session, dispatch]);
}

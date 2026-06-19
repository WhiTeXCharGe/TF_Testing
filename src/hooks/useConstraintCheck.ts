import { useEffect } from 'react';
import { useAppContext } from '../context/AppContext';
import { checkConstraints } from '../services/constraintService';

// Runs constraint check whenever schedule or envConfig changes
export function useConstraintCheck() {
  const { state, dispatch } = useAppContext();

  useEffect(() => {
    if (!state.envConfig || !state.schedule) {
      dispatch({ type: 'SET_VIOLATIONS', payload: [] });
      return;
    }
    const violations = checkConstraints(state.envConfig, state.schedule);
    dispatch({ type: 'SET_VIOLATIONS', payload: violations });
  }, [state.schedule, state.envConfig, dispatch]);
}

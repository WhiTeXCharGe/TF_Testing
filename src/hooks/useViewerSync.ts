import { useEffect } from 'react';
import { useAppContext } from '../context/AppContext';
import { connectAsViewer } from '../services/viewBroadcastService';

// Read-only side: connects to whichever host is broadcasting on this network
// and mirrors their current schedule/envConfig into local state as it arrives.
export function useViewerSync(): void {
  const { dispatch } = useAppContext();

  useEffect(() => {
    const disconnect = connectAsViewer(
      (snapshot) => {
        dispatch({
          type: 'SET_VIEW_STATE',
          payload: { schedule: snapshot.schedule, envConfig: snapshot.envConfig, currentView: snapshot.currentView },
        });
      },
      (status) => dispatch({ type: 'SET_VIEW_CONNECTION_STATUS', payload: status }),
    );
    return disconnect;
  }, [dispatch]);
}

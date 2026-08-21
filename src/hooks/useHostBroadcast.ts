import { useEffect, useRef } from 'react';
import { useAppContext } from '../context/AppContext';
import { sendHostUpdate, startHostBroadcast, stopHostBroadcast } from '../services/viewBroadcastService';

const DEBOUNCE_MS = 300;

// While the host has Live View sharing turned on, pushes the current
// schedule/envConfig to the broadcast server whenever it changes (debounced
// so a drag doesn't flood the socket), so viewers see it live.
export function useHostBroadcast(): void {
  const { state } = useAppContext();
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!state.isSharingLiveView) return;
    startHostBroadcast();
    return () => stopHostBroadcast();
  }, [state.isSharingLiveView]);

  useEffect(() => {
    if (!state.isSharingLiveView || !state.schedule || !state.envConfig) return;
    const schedule = state.schedule;
    const envConfig = state.envConfig;
    const currentView = state.currentView;
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      sendHostUpdate(schedule, envConfig, currentView);
    }, DEBOUNCE_MS);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [state.isSharingLiveView, state.schedule, state.envConfig, state.currentView]);
}

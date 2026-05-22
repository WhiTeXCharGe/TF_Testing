import { useState, useEffect } from 'react';
import { formatTimer } from '@/utils/dateUtils';

interface UseTimer {
  elapsedSeconds: number;
  timerLabel: string;
}

/**
 * Live timer hook.
 * @param startEpoch - Unix timestamp (ms) when the run started. Pass 0 to disable.
 */
export function useTimer(startEpoch: number): UseTimer {
  const [elapsedSeconds, setElapsedSeconds] = useState(
    startEpoch ? Math.floor((Date.now() - startEpoch) / 1000) : 0
  );

  useEffect(() => {
    if (!startEpoch) return;
    const id = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startEpoch) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, [startEpoch]);

  return { elapsedSeconds, timerLabel: formatTimer(elapsedSeconds) };
}

/**
 * @jest-environment jsdom
 */
import { render } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppProvider, useAppContext } from '../../context/AppContext';
import { useKeyboardShortcuts } from '../../hooks/useKeyboardShortcuts';
import * as fileService from '../../services/fileService';
import { ScheduleData } from '../../types/schedule';
import { EnvConfig } from '../../types/envConfig';

jest.mock('../../services/fileService');
const mockedFileService = fileService as jest.Mocked<typeof fileService>;

const SCHEDULE: ScheduleData = {
  planRange: { startDate: '2026-01-01', endDate: '2026-01-31' },
  workflowTaskList: [],
  assignmentList: [],
};
const ENV_CONFIG: EnvConfig = {
  workflowList: [], fabList: [], regionList: [], customerCompanyList: [], workerCompanyList: [], workerList: [], transiteDayMap: [],
};

function Harness({ inSession }: { inSession: boolean }) {
  useKeyboardShortcuts();
  const { dispatch } = useAppContext();
  return (
    <button
      onClick={() => {
        dispatch({ type: 'LOAD_FILES', payload: { schedule: SCHEDULE, envConfig: ENV_CONFIG, envPath: 'E.yaml', schedulePath: 'S.yaml' } });
        if (inSession) {
          dispatch({
            type: 'SET_SESSION',
            payload: { id: 's1', name: 'Test', role: 'edit', connectionStatus: 'connected', participants: [], status: 'open' },
          });
        }
      }}
    >
      setup
    </button>
  );
}

function renderHarness(inSession: boolean) {
  return render(<AppProvider><Harness inSession={inSession} /></AppProvider>);
}

beforeEach(() => {
  jest.clearAllMocks();
  mockedFileService.overwriteSaveFiles.mockResolvedValue(undefined);
});

describe('Ctrl+S', () => {
  it('saves to the known file paths when there is no active session', async () => {
    const { getByText } = renderHarness(false);
    await userEvent.click(getByText('setup'));

    await userEvent.keyboard('{Control>}s{/Control}');

    expect(mockedFileService.overwriteSaveFiles).toHaveBeenCalledWith(ENV_CONFIG, SCHEDULE, 'E.yaml', 'S.yaml');
  });

  it('does nothing while in an online session, matching the 上書き保存 menu item being disabled then', async () => {
    const { getByText } = renderHarness(true);
    await userEvent.click(getByText('setup'));

    await userEvent.keyboard('{Control>}s{/Control}');

    expect(mockedFileService.overwriteSaveFiles).not.toHaveBeenCalled();
  });
});

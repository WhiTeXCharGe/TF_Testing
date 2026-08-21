import { useAppContext } from '../../context/AppContext';
import { fetchShareableLink } from '../../services/viewBroadcastService';
import { UI } from '../../config/uiText';

export function ShareViewButton() {
  const { state, dispatch } = useAppContext();
  const has = !!state.schedule;

  const handleClick = async () => {
    dispatch({ type: 'OPEN_SHARE_VIEW_DIALOG' });
    if (!state.isSharingLiveView) {
      dispatch({ type: 'SET_SHARING_LIVE_VIEW', payload: true });
    }
    try {
      const link = await fetchShareableLink();
      dispatch({ type: 'SET_LIVE_VIEW_SHARE_LINK', payload: link });
    } catch (err) {
      dispatch({ type: 'SET_ERROR', payload: `共有リンクの取得に失敗しました: ${String(err)}` });
    }
  };

  return (
    <button
      disabled={!has}
      onClick={handleClick}
      style={{
        padding: '4px 10px',
        backgroundColor: state.isSharingLiveView ? '#00796b' : '#5c6bc0',
        color: '#fff',
        border: 'none',
        borderRadius: 3,
        cursor: has ? 'pointer' : 'default',
        fontSize: 12,
        fontFamily: 'MS Gothic, monospace',
        opacity: has ? 1 : 0.4,
      }}
    >
      {state.isSharingLiveView ? `📡 ${UI.shareLiveViewBtn}` : UI.shareLiveViewBtn}
    </button>
  );
}

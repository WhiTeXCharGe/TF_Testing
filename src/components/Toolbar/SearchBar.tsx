import { useState } from 'react';
import type { CSSProperties } from 'react';
import { useAppContext } from '../../context/AppContext';
import { UI } from '../../config/uiText';

export function SearchBar() {
  const { dispatch } = useAppContext();
  const [keyword, setKeyword] = useState('');
  const [mode, setMode] = useState<'device' | 'worker'>('device');

  const handleSearch = () => {
    dispatch({ type: 'SET_SEARCH_QUERY', payload: { keyword, mode } });
  };

  const handleClear = () => {
    setKeyword('');
    dispatch({ type: 'SET_SEARCH_QUERY', payload: { keyword: '', mode: '' } });
  };

  const inputStyle: CSSProperties = {
    padding: '3px 6px',
    border: '1px solid #999',
    borderRadius: 3,
    fontSize: 12,
    fontFamily: 'MS Gothic, monospace',
    width: 120,
  };

  const btn: CSSProperties = {
    padding: '4px 8px',
    border: '1px solid #999',
    borderRadius: 3,
    cursor: 'pointer',
    fontSize: 12,
    fontFamily: 'MS Gothic, monospace',
    backgroundColor: '#fff',
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      <input
        style={inputStyle}
        placeholder={UI.workerNamePlaceholder}
        value={keyword}
        onChange={e => setKeyword(e.target.value)}
        onKeyDown={e => e.key === 'Enter' && handleSearch()}
      />
      <select
        style={{ ...inputStyle, width: 80 }}
        value={mode}
        onChange={e => setMode(e.target.value as 'device' | 'worker')}
      >
        <option value="device">{UI.deviceCodeLabel}</option>
        <option value="worker">{UI.workerNamePlaceholder}</option>
      </select>
      <button style={{ ...btn, backgroundColor: '#1976d2', color: '#fff' }} onClick={handleSearch}>{UI.search}</button>
      <button style={btn} onClick={handleClear}>{UI.clear}</button>
    </div>
  );
}

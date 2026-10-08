'use client';

import { useCallback, useEffect, useState } from 'react';
import { EMPTY_DATA, errorMessage, readLocalData, type LocalData } from './local-data';

export function useLocalData() {
  const [data, setData] = useState<LocalData>(EMPTY_DATA);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const refresh = useCallback(async () => {
    try {
      setData(await readLocalData());
      setError('');
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    let active = true;
    const reload = async () => {
      try {
        const next = await readLocalData();
        if (active) {
          setData(next);
          setError('');
        }
      } catch (err) {
        if (active) setError(errorMessage(err));
      } finally {
        if (active) setLoading(false);
      }
    };
    void reload();
    window.addEventListener('adam:data', reload);
    window.addEventListener('storage', reload);
    return () => {
      active = false;
      window.removeEventListener('adam:data', reload);
      window.removeEventListener('storage', reload);
    };
  }, []);
  return { data, loading, error, refresh };
}

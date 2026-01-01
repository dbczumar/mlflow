import { useCallback, useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';

const STORAGE_KEY_PREFIX = 'mlflow.trace.pinnedFields';

export interface PinnedFieldConfig {
  path: string;
  displayName: string;
}

const getStorageKey = (experimentId: string | undefined) => {
  return experimentId ? `${STORAGE_KEY_PREFIX}.${experimentId}` : STORAGE_KEY_PREFIX;
};

const getStoredPinnedFields = (experimentId: string | undefined): PinnedFieldConfig[] => {
  try {
    const stored = localStorage.getItem(getStorageKey(experimentId));
    if (stored) {
      const parsed = JSON.parse(stored);
      if (Array.isArray(parsed)) {
        return parsed;
      }
    }
  } catch {
    // Invalid data, return empty array
  }
  return [];
};

const setStoredPinnedFields = (experimentId: string | undefined, fields: PinnedFieldConfig[]) => {
  try {
    localStorage.setItem(getStorageKey(experimentId), JSON.stringify(fields));
  } catch {
    // localStorage unavailable, ignore
  }
  // Notify all subscribers
  listeners.forEach((listener) => listener());
};

// Simple pub/sub for cross-component sync
const listeners = new Set<() => void>();

export interface UsePinnedFieldsReturn {
  pinnedFields: PinnedFieldConfig[];
  isPinned: (path: string) => boolean;
  pinField: (path: string, displayName: string) => void;
  unpinField: (path: string) => void;
  togglePin: (path: string, displayName: string) => void;
}

export const usePinnedFields = (): UsePinnedFieldsReturn => {
  const { experimentId } = useParams<{ experimentId?: string }>();
  const [pinnedFields, setPinnedFields] = useState<PinnedFieldConfig[]>(() => getStoredPinnedFields(experimentId));

  useEffect(() => {
    // Update state when experimentId changes
    setPinnedFields(getStoredPinnedFields(experimentId));
  }, [experimentId]);

  useEffect(() => {
    const handleUpdate = () => {
      setPinnedFields(getStoredPinnedFields(experimentId));
    };

    listeners.add(handleUpdate);
    return () => {
      listeners.delete(handleUpdate);
    };
  }, [experimentId]);

  const isPinned = useCallback((path: string) => pinnedFields.some((f) => f.path === path), [pinnedFields]);

  const pinField = useCallback(
    (path: string, displayName: string) => {
      if (!isPinned(path)) {
        const newFields = [...pinnedFields, { path, displayName }];
        setPinnedFields(newFields);
        setStoredPinnedFields(experimentId, newFields);
      }
    },
    [pinnedFields, isPinned, experimentId],
  );

  const unpinField = useCallback(
    (path: string) => {
      const newFields = pinnedFields.filter((f) => f.path !== path);
      setPinnedFields(newFields);
      setStoredPinnedFields(experimentId, newFields);
    },
    [pinnedFields, experimentId],
  );

  const togglePin = useCallback(
    (path: string, displayName: string) => {
      if (isPinned(path)) {
        unpinField(path);
      } else {
        pinField(path, displayName);
      }
    },
    [isPinned, pinField, unpinField],
  );

  return {
    pinnedFields,
    isPinned,
    pinField,
    unpinField,
    togglePin,
  };
};

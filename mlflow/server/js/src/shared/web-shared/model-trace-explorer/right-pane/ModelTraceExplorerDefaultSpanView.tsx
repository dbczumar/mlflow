import { isNil } from 'lodash';
import { useMemo } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceSpanNode, SearchMatch } from '../ModelTrace.types';
import { createListFromObject } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { PinButton, usePinnedFields } from '../pinned-fields';

export function ModelTraceExplorerDefaultSpanView({
  activeSpan,
  className,
  searchFilter,
  activeMatch,
}: {
  activeSpan: ModelTraceSpanNode | undefined;
  className?: string;
  searchFilter: string;
  activeMatch: SearchMatch | null;
}) {
  const { theme } = useDesignSystemTheme();
  const { isPinned, togglePin } = usePinnedFields();
  const inputList = useMemo(() => createListFromObject(activeSpan?.inputs), [activeSpan]);
  const outputList = useMemo(() => createListFromObject(activeSpan?.outputs), [activeSpan]);

  if (isNil(activeSpan)) {
    return null;
  }

  const containsInputs = inputList.length > 0;
  const containsOutputs = outputList.length > 0;

  const isActiveMatchSpan = !isNil(activeMatch) && activeMatch.span.key === activeSpan.key;

  return (
    <div data-testid="model-trace-explorer-default-span-view">
      {containsInputs && (
        <ModelTraceExplorerCollapsibleSection
          withBorder
          css={{ marginBottom: theme.spacing.sm }}
          sectionKey="input"
          title={
            <FormattedMessage
              defaultMessage="Inputs"
              description="Model trace explorer > selected span > inputs header"
            />
          }
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            {inputList.map(({ key, value }, index) => {
              // If there's a key, pin the specific field. If no key (scalar), pin the entire inputs.
              const fieldPath = key ? `inputs.${key}` : 'inputs';
              const displayName = key || 'Inputs';
              const pinned = isPinned(fieldPath);
              return (
                <div
                  key={key || index}
                  css={{
                    '&:hover .pin-button-inline': {
                      opacity: 1,
                    },
                  }}
                >
                  <ModelTraceExplorerCodeSnippet
                    title={key}
                    data={value}
                    searchFilter={searchFilter}
                    activeMatch={activeMatch}
                    containsActiveMatch={
                      isActiveMatchSpan && activeMatch.section === 'inputs' && activeMatch.key === key
                    }
                    titleActions={
                      <div
                        className="pin-button-inline"
                        css={{
                          opacity: pinned ? 1 : 0,
                          transition: 'opacity 0.15s ease-in-out',
                        }}
                      >
                        <PinButton isPinned={pinned} onToggle={() => togglePin(fieldPath, displayName)} />
                      </div>
                    }
                  />
                </div>
              );
            })}
          </div>
        </ModelTraceExplorerCollapsibleSection>
      )}
      {containsOutputs && (
        <ModelTraceExplorerCollapsibleSection
          withBorder
          sectionKey="output"
          title={
            <FormattedMessage
              defaultMessage="Outputs"
              description="Model trace explorer > selected span > outputs header"
            />
          }
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            {outputList.map(({ key, value }, index) => {
              // If there's a key, pin the specific field. If no key (scalar), pin the entire outputs.
              const fieldPath = key ? `outputs.${key}` : 'outputs';
              const displayName = key || 'Outputs';
              const pinned = isPinned(fieldPath);
              return (
                <div
                  key={key || index}
                  css={{
                    '&:hover .pin-button-inline': {
                      opacity: 1,
                    },
                  }}
                >
                  <ModelTraceExplorerCodeSnippet
                    title={key}
                    data={value}
                    searchFilter={searchFilter}
                    activeMatch={activeMatch}
                    containsActiveMatch={
                      isActiveMatchSpan && activeMatch.section === 'outputs' && activeMatch.key === key
                    }
                    titleActions={
                      <div
                        className="pin-button-inline"
                        css={{
                          opacity: pinned ? 1 : 0,
                          transition: 'opacity 0.15s ease-in-out',
                        }}
                      >
                        <PinButton isPinned={pinned} onToggle={() => togglePin(fieldPath, displayName)} />
                      </div>
                    }
                  />
                </div>
              );
            })}
          </div>
        </ModelTraceExplorerCollapsibleSection>
      )}
    </div>
  );
}

import { useState } from 'react';

import { Typography, useDesignSystemTheme } from '@databricks/design-system';

import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import {
  ModelTraceExplorerFieldRenderer,
  DEFAULT_MAX_VISIBLE_CHAT_MESSAGES,
} from '../field-renderers/ModelTraceExplorerFieldRenderer';
import { PinButton, usePinnedFields } from '../pinned-fields';

const DEFAULT_MAX_VISIBLE_ITEMS = 3;

export const ModelTraceExplorerSummarySection = ({
  title,
  data,
  renderMode,
  sectionKey,
  fieldPathPrefix,
  maxVisibleItems = DEFAULT_MAX_VISIBLE_ITEMS,
  maxVisibleChatMessages = DEFAULT_MAX_VISIBLE_CHAT_MESSAGES,
  className,
  chatMessageFormat,
}: {
  title: React.ReactElement;
  data: { key: string; value: string }[];
  renderMode: 'default' | 'json' | 'text';
  sectionKey: string;
  fieldPathPrefix?: 'inputs' | 'outputs';
  maxVisibleItems?: number;
  maxVisibleChatMessages?: number;
  className?: string;
  chatMessageFormat?: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);
  const { isPinned, togglePin } = usePinnedFields();
  const shouldTruncateItems = data.length > maxVisibleItems;

  const visibleItems = shouldTruncateItems && !expanded ? data.slice(-maxVisibleItems) : data;
  const hiddenItemCount = shouldTruncateItems ? data.length - visibleItems.length : 0;

  return (
    <ModelTraceExplorerCollapsibleSection withBorder title={title} className={className} sectionKey={sectionKey}>
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        {shouldTruncateItems && (
          <Typography.Link
            css={{ alignSelf: 'flex-start', marginLeft: theme.spacing.xs }}
            componentId="shared.model-trace-explorer.conversation-toggle"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? 'Show less' : `Show ${hiddenItemCount} more`}
          </Typography.Link>
        )}
        {visibleItems.map(({ key, value }, index) => {
          // If there's a key, pin the specific field. If no key (scalar), pin the entire section.
          const fieldPath = fieldPathPrefix ? (key ? `${fieldPathPrefix}.${key}` : fieldPathPrefix) : undefined;
          const displayName = key || (fieldPathPrefix === 'inputs' ? 'Inputs' : 'Outputs');
          const pinned = fieldPath ? isPinned(fieldPath) : false;

          return (
            <div
              key={key || index}
              css={{
                position: 'relative',
                '&:hover .pin-button-wrapper': {
                  opacity: 1,
                },
              }}
            >
              {fieldPath && (
                <div
                  className="pin-button-wrapper"
                  css={{
                    position: 'absolute',
                    top: theme.spacing.xs,
                    right: theme.spacing.xs,
                    zIndex: 1,
                    opacity: pinned ? 1 : 0,
                    transition: 'opacity 0.15s ease-in-out',
                  }}
                >
                  <PinButton isPinned={pinned} onToggle={() => togglePin(fieldPath!, displayName)} />
                </div>
              )}
              <ModelTraceExplorerFieldRenderer
                title={key}
                data={value}
                renderMode={renderMode}
                chatMessageFormat={chatMessageFormat}
                maxVisibleMessages={maxVisibleChatMessages}
              />
            </div>
          );
        })}
      </div>
    </ModelTraceExplorerCollapsibleSection>
  );
};

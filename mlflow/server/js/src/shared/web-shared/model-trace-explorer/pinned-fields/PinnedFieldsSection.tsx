import { PinFillIcon, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { ModelTraceExplorerFieldRenderer } from '../field-renderers/ModelTraceExplorerFieldRenderer';
import type { ModelTraceExplorerRenderMode } from '../ModelTrace.types';
import type { PinnedFieldConfig } from './usePinnedFields';
import { usePinnedFields } from './usePinnedFields';

const getFieldValue = (data: any, fieldPath: string): string | null => {
  if (!data) return null;

  // Handle "inputs" or "outputs" (entire object)
  if (fieldPath === 'inputs' || fieldPath === 'outputs') {
    return JSON.stringify(data, null, 2);
  }

  // Handle "inputs.fieldName" or "outputs.fieldName"
  const parts = fieldPath.split('.');
  if (parts.length === 2) {
    const fieldName = parts[1];
    if (data && typeof data === 'object' && fieldName in data) {
      return JSON.stringify(data[fieldName], null, 2);
    }
  }

  return null;
};

interface PinnedFieldItemProps {
  field: PinnedFieldConfig;
  value: string;
  renderMode: ModelTraceExplorerRenderMode;
  chatMessageFormat?: string;
  onUnpin: () => void;
}

const PinnedFieldItem = ({ field, value, renderMode, chatMessageFormat, onUnpin }: PinnedFieldItemProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        position: 'relative',
        '&:hover .unpin-button': {
          opacity: 1,
        },
      }}
    >
      <Tooltip
        componentId="shared.model-trace-explorer.unpin-button-tooltip"
        content={
          <FormattedMessage defaultMessage="Unpin" description="Tooltip for unpin button in pinned fields section" />
        }
      >
        <button
          type="button"
          className="unpin-button"
          onClick={onUnpin}
          css={{
            position: 'absolute',
            top: theme.spacing.sm,
            right: theme.spacing.sm,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: theme.spacing.xs,
            border: 'none',
            background: 'transparent',
            cursor: 'pointer',
            borderRadius: theme.borders.borderRadiusSm,
            color: theme.colors.actionPrimaryBackgroundDefault,
            zIndex: 1,
            transition: 'background 0.15s ease-in-out',
            ':hover': {
              background: theme.colors.actionTertiaryBackgroundHover,
            },
          }}
          aria-label="Unpin"
        >
          <PinFillIcon />
        </button>
      </Tooltip>
      <ModelTraceExplorerFieldRenderer
        title={field.path}
        data={value}
        renderMode={renderMode}
        chatMessageFormat={chatMessageFormat}
      />
    </div>
  );
};

export interface PinnedFieldsSectionProps {
  rootInputs: any;
  rootOutputs: any;
  renderMode: ModelTraceExplorerRenderMode;
  chatMessageFormat?: string;
}

export const PinnedFieldsSection = ({
  rootInputs,
  rootOutputs,
  renderMode,
  chatMessageFormat,
}: PinnedFieldsSectionProps) => {
  const { theme } = useDesignSystemTheme();
  const { pinnedFields, unpinField } = usePinnedFields();

  // Filter to only show pinned fields that exist in the current trace
  const visiblePinnedFields = pinnedFields
    .map((field) => {
      const isInputField = field.path === 'inputs' || field.path.startsWith('inputs.');
      const data = isInputField ? rootInputs : rootOutputs;
      const value = getFieldValue(data, field.path);
      return { field, value };
    })
    .filter(({ value }) => value !== null && value !== 'null');

  if (visiblePinnedFields.length === 0) {
    return null;
  }

  return (
    <ModelTraceExplorerCollapsibleSection
      withBorder
      sectionKey="pinned-fields"
      css={{ marginBottom: theme.spacing.md }}
      title={
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <FormattedMessage
            defaultMessage="Pinned Fields"
            description="Header for pinned fields section in trace explorer"
          />
          <Typography.Hint css={{ fontWeight: 'normal' }}>({visiblePinnedFields.length})</Typography.Hint>
        </div>
      }
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        {visiblePinnedFields.map(({ field, value }) => (
          <PinnedFieldItem
            key={field.path}
            field={field}
            value={value!}
            renderMode={renderMode}
            chatMessageFormat={chatMessageFormat}
            onUnpin={() => unpinField(field.path)}
          />
        ))}
      </div>
    </ModelTraceExplorerCollapsibleSection>
  );
};

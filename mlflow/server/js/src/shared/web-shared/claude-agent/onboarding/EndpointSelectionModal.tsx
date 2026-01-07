/**
 * Modal for selecting or creating an endpoint for judges.
 */

import { useCallback, useState } from 'react';
import { Modal, Button, Typography, useDesignSystemTheme, Spinner, Alert, PlusIcon } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useEndpointsQuery } from '../../../../gateway/hooks/useEndpointsQuery';
import { CreateEndpointModal } from '../../../../gateway/components/endpoint-form';
import { getEndpointDisplayInfo } from '../../../../gateway/utils/gatewayUtils';
import type { Endpoint } from '../../../../gateway/types';

const COMPONENT_ID_PREFIX = 'mlflow.onboarding.endpoint-selection-modal';

interface EndpointSelectionModalProps {
  open: boolean;
  onClose: () => void;
  onEndpointSelect: (endpointName: string) => void;
  currentEndpointName?: string;
}

export const EndpointSelectionModal: React.FC<EndpointSelectionModalProps> = ({
  open,
  onClose,
  onEndpointSelect,
  currentEndpointName,
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const { data: endpoints, isLoading, error, refetch } = useEndpointsQuery();
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);

  const handleOpenCreateModal = useCallback(() => {
    setIsCreateModalOpen(true);
  }, []);

  const handleCloseCreateModal = useCallback(() => {
    setIsCreateModalOpen(false);
  }, []);

  const handleCreateEndpointSuccess = useCallback(
    async (endpoint: Endpoint) => {
      await refetch();
      onEndpointSelect(endpoint.name);
      setIsCreateModalOpen(false);
      onClose();
    },
    [refetch, onEndpointSelect, onClose],
  );

  const handleEndpointClick = useCallback(
    (endpointName: string) => {
      onEndpointSelect(endpointName);
      onClose();
    },
    [onEndpointSelect, onClose],
  );

  return (
    <>
      <Modal
        componentId={`${COMPONENT_ID_PREFIX}.modal`}
        title={intl.formatMessage({
          defaultMessage: 'Select Judge Endpoint',
          description: 'Title for endpoint selection modal',
        })}
        visible={open}
        onCancel={onClose}
        footer={
          <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
            <Button componentId={`${COMPONENT_ID_PREFIX}.cancel`} onClick={onClose}>
              <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
            </Button>
          </div>
        }
        size="normal"
      >
        <div css={{ minHeight: 300 }}>
          <Typography.Text css={{ display: 'block', marginBottom: theme.spacing.md }}>
            <FormattedMessage
              defaultMessage="Select an LLM endpoint that will power your judges, or create a new one."
              description="Description for endpoint selection"
            />
          </Typography.Text>

          {isLoading && (
            <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: theme.spacing.lg }}>
              <Spinner size="large" />
            </div>
          )}

          {error && (
            <Alert
              componentId={`${COMPONENT_ID_PREFIX}.error`}
              type="warning"
              message={
                <FormattedMessage
                  defaultMessage="Unable to load endpoints. Please try again or create a new endpoint."
                  description="Error message when endpoints fail to load"
                />
              }
              closable
            />
          )}

          {!isLoading && !error && (
            <>
              {endpoints.length === 0 ? (
                <div
                  css={{
                    padding: theme.spacing.lg,
                    textAlign: 'center',
                    backgroundColor: theme.colors.backgroundSecondary,
                    borderRadius: theme.borders.borderRadiusMd,
                    marginBottom: theme.spacing.md,
                  }}
                >
                  <Typography.Text color="secondary">
                    <FormattedMessage
                      defaultMessage="No endpoints available. Create one to get started."
                      description="Empty state message"
                    />
                  </Typography.Text>
                </div>
              ) : (
                <div
                  css={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: theme.spacing.sm,
                    marginBottom: theme.spacing.md,
                  }}
                >
                  {endpoints.map((endpoint) => {
                    const displayInfo = getEndpointDisplayInfo(endpoint);
                    const isSelected = currentEndpointName === endpoint.name;

                    return (
                      <button
                        key={endpoint.name}
                        onClick={() => handleEndpointClick(endpoint.name)}
                        css={{
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'flex-start',
                          padding: theme.spacing.md,
                          backgroundColor: isSelected
                            ? theme.colors.actionPrimaryBackgroundDefault
                            : theme.colors.backgroundSecondary,
                          border: `1px solid ${
                            isSelected ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.border
                          }`,
                          borderRadius: theme.borders.borderRadiusMd,
                          cursor: 'pointer',
                          textAlign: 'left',
                          transition: 'all 0.2s',
                          '&:hover': {
                            borderColor: theme.colors.actionPrimaryBackgroundDefault,
                            backgroundColor: isSelected
                              ? theme.colors.actionPrimaryBackgroundDefault
                              : theme.colors.backgroundPrimary,
                          },
                        }}
                      >
                        <Typography.Text
                          bold
                          css={{
                            color: isSelected ? theme.colors.actionPrimaryTextDefault : theme.colors.textPrimary,
                          }}
                        >
                          {endpoint.name}
                        </Typography.Text>
                        {displayInfo?.provider && displayInfo?.modelName && (
                          <Typography.Text
                            size="sm"
                            css={{
                              color: isSelected ? theme.colors.actionPrimaryTextDefault : theme.colors.textSecondary,
                            }}
                          >
                            {displayInfo.provider} / {displayInfo.modelName}
                          </Typography.Text>
                        )}
                      </button>
                    );
                  })}
                </div>
              )}

              {/* Create new endpoint button */}
              <Button
                componentId={`${COMPONENT_ID_PREFIX}.create-new`}
                type="primary"
                icon={<PlusIcon />}
                onClick={handleOpenCreateModal}
                css={{ width: '100%' }}
              >
                <FormattedMessage defaultMessage="Create New Endpoint" description="Create new endpoint button" />
              </Button>
            </>
          )}
        </div>
      </Modal>

      <CreateEndpointModal
        open={isCreateModalOpen}
        onClose={handleCloseCreateModal}
        onSuccess={handleCreateEndpointSuccess}
      />
    </>
  );
};

import type { QueryFunctionContext } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { ListSecretInfosResponse } from '../types';

const queryFn = ({ queryKey }: QueryFunctionContext<SecretsQueryKey>) => {
  const [, { provider }] = queryKey;
  return GatewayApi.listSecrets(provider);
};

type SecretsQueryKey = ['gateway_secrets', { provider?: string }];

export const useSecretsQuery = ({ provider, enabled = true }: { provider?: string; enabled?: boolean } = {}) => {
  const queryResult = useQuery<ListSecretInfosResponse, Error, ListSecretInfosResponse, SecretsQueryKey>(
    ['gateway_secrets', { provider }],
    {
      queryFn,
      retry: false,
      enabled,
    },
  );

  return {
    data: queryResult.data?.secrets ?? [],
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    refetch: queryResult.refetch,
  };
};

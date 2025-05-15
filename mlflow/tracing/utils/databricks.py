import logging
import os
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from mlflow.entities import Trace, TraceData, TraceInfo

_logger = logging.getLogger(__name__)

# Constants for request handling
TRANSIENT_FAILURE_RESPONSE_CODES = [
    408,  # Request Timeout
    429,  # Too Many Requests
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
]
REQUEST_TIMEOUT = 120


def download_trace(trace_id, databricks_host, databricks_auth_headers):
    """
    Download trace data for a single trace ID using the Databricks API.

    Args:
        trace_id: The trace ID to download
        databricks_host: The Databricks host URL
        databricks_auth_headers: Authentication headers for Databricks API

    Returns:
        A dictionary with trace_id and trace_data or None if download failed
    """
    url_path = f"/api/3.0/mlflow/traces/{trace_id}/credentials-for-data-download"
    url = f"{databricks_host.rstrip('/')}{url_path}"

    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=8, backoff_factor=0.25, status_forcelist=TRANSIENT_FAILURE_RESPONSE_CODES
            )
        ),
    )

    try:
        credentials_response = session.get(
            url, headers=databricks_auth_headers, timeout=REQUEST_TIMEOUT
        )
        credentials_response.raise_for_status()  # Raise an error for bad responses
        credential_info = credentials_response.json()["credential_info"]
        data_response = session.get(
            credential_info["signed_uri"],
            headers=credential_info.get("headers"),
            timeout=REQUEST_TIMEOUT,
        )
        data_response.raise_for_status()  # Raise an error for bad responses
        return {"trace_id": trace_id, "trace_data": data_response.json()}
    except Exception as e:
        if "429" in str(e):
            return download_trace(trace_id, databricks_host, databricks_auth_headers)
        _logger.warning(f"Exception while downloading trace {trace_id}: {e}")
        return None


def get_full_traces_databricks(trace_infos: list[TraceInfo]):
    from databricks.sdk import WorkspaceClient

    databricks_client = WorkspaceClient()
    databricks_host = databricks_client.config.host
    databricks_auth_headers = databricks_client.config.authenticate()

    # Extract trace IDs from trace_infos
    trace_ids = [trace_info.request_id for trace_info in trace_infos]

    # Use ThreadPoolExecutor to download traces in parallel
    max_workers = min(64, os.cpu_count() * 4 if os.cpu_count() else 16)
    trace_id_to_data = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_trace, trace_id, databricks_host, databricks_auth_headers)
            for trace_id in trace_ids
        ]

        # Process results as they complete
        for future in futures:
            result = future.result()
            if result is not None:
                trace_id_to_data[result["trace_id"]] = result["trace_data"]

    # Create full traces with the downloaded data
    full_traces = []
    for trace_info in trace_infos:
        trace_data = trace_id_to_data.get(trace_info.request_id)
        if trace_data is not None:
            trace_data = TraceData.from_dict(trace_data)
        else:
            trace_data = TraceData()
            _logger.warning(
                f"Trace data not found for trace ID {trace_info.request_id}. "
                "This may indicate a failure in the download process."
            )

        full_traces.append(Trace(info=trace_info, data=trace_data))

    return full_traces

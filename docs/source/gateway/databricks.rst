.. _gateway:

=====================
Databricks AI Gateway
=====================

.. warning::

    - **Support for using the Databricks AI Gateway is in private preview.**
    - Operability and support channels might be insufficient for production use. Support is best
      effort and informal—please reach out to your account or external Slack channel for best
      effort support.
    - This private preview will be free of charge at this time. We may charge for it in the future.
    - You will still incur charges for DBUs.
    - This product may change or may never be released.
    - We may terminate the preview or your access to it with 2 weeks of notice.
    - There may be API changes before Public Preview or General Availability. We will give you at
      least 2 weeks notice before any significant changes so that you have time to update your
      projects.
    - Non-public information about the preview (including the fact that there is a preview for the
      feature/product itself) is confidential.

The Databricks AI Gateway service is a powerful tool designed to streamline the usage and management of
various large language model (LLM) providers, such as OpenAI and Anthropic, within an organization.
It offers a high-level interface that simplifies the interaction with these services by providing
a unified endpoint to handle specific LLM related requests.

A major advantage of using the Databricks AI Gateway service is its centralized credential management.
By storing API keys in one secure location, organizations can significantly enhance their
security posture by minimizing the exposure of sensitive API keys throughout the system. It also
helps to prevent exposing these keys within code or requiring end-users to manage keys safely.

The gateway is designed to be flexible and adaptable, capable of easily defining and managing routes and rate limits
using a straightforward REST API. This enables the easy incorporation
of new LLM providers or provider LLM types into the system without necessitating changes to
applications that interface with the gateway. This level of adaptability makes the Databricks AI Gateway
Service an invaluable tool in environments that require agility and quick response to changes.

This simplification and centralization of language model interactions, coupled with the added
layer of security for API key management, rate limiting for cost control, make the Databricks AI Gateway service an ideal choice for
organizations that use LLMs on a regular basis.

.. _gateway-quickstart:

Quickstart
==========

The following guide will assist you in getting up and running, using a 3-route configuration to
OpenAI services for chat, completions, and embeddings.

Step 1: Install the Databricks AI Gateway client
------------------------------------------------
First, you need to install the Databricks AI Gateway Python client. You can do this using ``%pip`` in
your Databricks notebook as follows:

Installing from PyPI
~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh

    %pip install 'mlflow[gateway]'

Step 2: Set the OpenAI API Key(s) for each provider
---------------------------------------------------
The Gateway service needs to communicate with the OpenAI API. To do this, it requires an API key.
You can create an API key from the OpenAI dashboard.

For this example, we're only connecting with OpenAI. If there are additional providers within the
configuration, these keys will need to be set as well.

Once you have the key, we recommend storing it using
`Databricks Secrets <https://docs.databricks.com/security/secrets/index.html>`_. In this quickstart,
we assume that the OpenAI key is available in secret scope ``example`` with key ``openai-api-key``.

Step 3: Create Gateway Routes
------------------------------
The next step is to create Gateway Routes for each LLM you want to use. In this example, we call
the :py:func:`mlflow.gateway.create_route()` API. For more information, see the
:ref:`gateway_fluent_api` and :ref:`gateway_client_api` sections.

If you are using the AI Gateway in a Databricks Notebook or Databricks Job, you can set the gateway URI as follows:

.. code-block:: python

    from mlflow.gateway import set_gateway_uri

    set_gateway_uri(gateway_uri="databricks")

If you are using the AI Gateway outside of a Databricks Notebook or Databricks Job, you will need to configure
your Databricks host name and personal access token in your current environment before making requests to
the Gateway. You can do this using the ``DATABRICKS_HOST`` and ``DATABRICKS_TOKEN`` environment variables.
For example:

.. code-block:: python

    import os
    from mlflow.gateway import set_gateway_uri

    os.environ["DATABRICKS_HOST"] = "http://your.workspace.databricks.com"
    os.environ["DATABRICKS_TOKEN"] = "<your_personal_access_token>"

    set_gateway_uri(gateway_uri="databricks")

Now that you have set the Gateway URI in your Python environment, you can create routes as follows:

.. code-block:: python

    from mlflow.gateway import create_route

    openai_api_key = dbutils.secrets.get(
        scope="example",
        key="openai-api-key"
    )

    # Create a Route for completions with OpenAI GPT-4
    create_route(
        name="completions",
        route_type="llm/v1/completions",
        model={
            "name": "gpt-4",
            "provider": "openai",
            "openai_config": {
                "openai_api_key": openai_api_key
            }
        }
    )

    # Create a Route for chat with OpenAI GPT-4
    create_route(
        name="chat",
        route_type="llm/v1/chat",
        model={
            "name": "gpt-4",
            "provider": "openai",
            "openai_config": {
                "openai_api_key": openai_api_key
            }
        }
    )

    # Create a Route for embeddings with OpenAI text-embedding-ada-002
    create_route(
        name="embeddings",
        route_type="llm/v1/embeddings",
        model={
            "name": "text-embedding-ada-002",
            "provider": "openai",
            "openai_config": {
                "openai_api_key": openai_api_key
            }
        }
    )

Step 4: Set rate limits on AI Gateway routes
--------------------------------------------------

After you created AI Gateway routes, you can set rate limits on AI Gateway routes for cost control, ensuring production availability and fair sharing.  For example:

.. code-block:: python

    from mlflow.gateway import set_limits

    # Set a per user limit (5 calls per minute) for the route created
    set_limits(
        route="completions",
        limits=[
            {
                "key": "user",
                "calls": "5",
                "renewal_period": "minute"
            }
        ]
    )

For more details, see :ref:`rate_limits` sections.

Step 5: Send Requests Using the Fluent API
------------------------------------------

The next step is to query the Routes using the :ref:`gateway_fluent_api`.
For information on formatting requirements and how to pass parameters, see :ref:`gateway_query`.

Completions
~~~~~~~~~~~
Here's an example of how to send a completions request using the :ref:`gateway_fluent_api` :

.. code-block:: python

    from mlflow.gateway import set_gateway_uri, query

    set_gateway_uri("databricks")

    response = query(
        route="completions",
        data={"prompt": "What is the best day of the week?", "temperature": 0.3}
    )

    print(response)

The returned response will have the following structure (the actual content and token values will likely be different):

.. code-block:: python

    {
         "candidates": [
           {
             "text": "It's hard to say what the best day of the week is.",
             "metadata": {
               "finish_reason": "stop"
             }
           }
        ],
        "metadata": {
            "input_tokens": 13,
            "output_tokens": 15,
            "total_tokens": 28,
            "model": "gpt-4",
            "route_type": "llm/v1/completions"
        }
    }


Chat
~~~~
Here's an example of how to send a chat request using the :ref:`gateway_fluent_api` :

.. code-block:: python

    from mlflow.gateway import set_gateway_uri, query

    set_gateway_uri("databricks")

    response = query(
        route="chat",
        data={"messages": [{"role": "user", "content": "What is the best day of the week?"}]}
    )

    print(response)

The returned response will have the following structure (the actual content and token values will likely be different):

.. code-block:: python

    {
        "candidates": [
            {
                "message": {
                    "role": "assistant",
                    "content": "\n\nIt's hard to say what the best day of the week is.",
                },
                "metadata": {"finish_reason": "stop"}
            }
        ],
        "metadata": {
            "input_tokens": 13,
            "output_tokens": 15,
            "total_tokens": 28,
            "model": "gpt-4",
            "route_type": "llm/v1/completions"
        }
    }

Embeddings
~~~~~~~~~~

Here's an example of how to send an embeddings request using the :ref:`gateway_fluent_api` :

.. code-block:: python

    from mlflow.gateway import set_gateway_uri, query

    set_gateway_uri("databricks")

    response = query(
        route="embeddings",
        data={"text": ["Example text to embed"]}
    )

    print(response)

The returned response will have the following structure (the actual content and token values will likely be different):

.. code-block:: python

    {
        "embeddings": [
          0.010169279,
          -0.0053696977,
          -0.018654726,
          -0.03396831,
          3.1851505e-05,
          -0.03341145,
          -0.023189139,
          ...
        ],
        "metadata": {
            "input_tokens": 6,
            "total_tokens": 6,
            "model": "text-embedding-ada-002",
            "route_type": "llm/v1/embeddings"
        }
    }

Step 6: Send Requests Using the Client API
------------------------------------------
See the :ref:`gateway_client_api` section for further information.

Step 7: Send Requests to Routes via REST API
--------------------------------------------
See the :ref:`REST examples <gateway_rest_api>` section for further information.

Step 8: Compare Provider Models
-------------------------------
Here's an example of adding and querying a new model from a different provider - in this case
Anthropic - to determine which model is better for a given use case. We assume that the
Anthropic API key is stored in `Databricks Secrets <https://docs.databricks.com/security/secrets/index.html>`_
with scope ``example`` and key ``anthropic-api-key``.

.. code-block:: python

    from mlflow.gateway import set_gateway_uri, create_route, query

    set_gateway_uri("databricks")

    anthropic_api_key = dbutils.secrets.get(
        scope="example",
        key="anthropic-api-key"
    )

    # Create a Route for completions with OpenAI GPT-4
    create_route(
        name="claude-completions",
        route_type="llm/v1/completions",
        model={
            "name": "claude-v1.3",
            "provider": "anthropic",
            "anthropic_config": {
                "anthropic_api_key": anthropic_api_key
            }
        }
    )

    completions_response = query(
        route="claude-completions",
        data={"prompt": "What is MLflow? Be concise.", "temperature": 0.3}
    )

The returned response will have the following structure (the actual content and token values will likely be different):

.. code-block:: python

    {
        "candidates": [
            {
                "text": "MLflow is an open source platform for machine learning...",
                "metadata": {
                    "finish_reason": "stop"
                }
            }
        ],
        "metadata": {
            "input_tokens": 8,
            "output_tokens": 15,
            "total_tokens": 23,
            "model": "claude-v1.3",
            "route_type": "llm/v1/completions"
        }
    }

Finally, if you no longer need a route, you can delete it using the
:py:func:`mlflow.gateway.delete_route` API. For more information, see the
:ref:`gateway_fluent_api` and :ref:`gateway_client_api` sections.

Step 9: Use AI Gateway routes for model development
---------------------------------------------------

Now that you have created several AI Gateway routes, you can create MLflow Models that query these
routes to build application-specific logic using techniques like prompt engineering. For more
information, see :ref:`AI Gateway and MLflow Models <gateway_mlflow_models>`.


.. _gateway-concepts:

Concepts
========

There are several concepts that are referred to within the Databricks AI Gateway APIs, the configuration definitions, examples, and documentation.
Becoming familiar with these terms will help in configuring new endpoints (routes) and ease the use of the interface APIs for the AI Gateway.

.. _providers:

Providers
---------
The Databricks AI Gateway is designed to support a variety of model providers.
A provider represents the source of the machine learning models, such as OpenAI, Anthropic, and so on.
Each provider has its specific characteristics and configurations that are encapsulated within the model part of a route in the Databricks AI Gateway.

Supported Provider Models
~~~~~~~~~~~~~~~~~~~~~~~~~
The table below presents a non-exhaustive list of models and a corresponding route type within the Databricks AI Gateway.
With the rapid development of LLMs, there is no guarantee that this list will be up to date at all times. However, the associations listed
below can be used as a helpful guide when configuring a given route for any newly released model types as they become available with a given provider.
Customers are responsible for ensuring compliance with applicable model licenses.

.. list-table::
   :header-rows: 1

   * - Route Type
     - Provider
     - Model Examples
     - Supported
   * - llm/v1/completions
     - OpenAI
     - gpt-3.5-turbo, gpt-4
     - Yes
   * - llm/v1/completions
     - MosaicML
     - mpt-7b-instruct, mpt-30b-instruct, llama2-70b-chat†
     - Yes
   * - llm/v1/completions
     - Anthropic
     - claude-1, claude-1.3-100k
     - Yes
   * - llm/v1/completions
     - Cohere
     - command, command-light-nightly
     - Yes
   * - llm/v1/completions
     - Azure OpenAI
     - text-davinci-003, gpt-35-turbo
     - Yes
   * - llm/v1/completions
     - Databricks Model Serving
     - Endpoints with compatible schemas
     - Yes
   * - llm/v1/chat
     - OpenAI
     - gpt-3.5-turbo, gpt-4
     - Yes
   * - llm/v1/chat
     - MosaicML
     -
     - No
   * - llm/v1/chat
     - Anthropic
     -
     - No
   * - llm/v1/chat
     - Cohere
     -
     - No
   * - llm/v1/chat
     - Azure OpenAI
     - gpt-35-turbo, gpt-4
     - Yes
   * - llm/v1/chat
     - Databricks Model Serving
     -
     - No
   * - llm/v1/embeddings
     - OpenAI
     - text-embedding-ada-002
     - Yes
   * - llm/v1/embeddings
     - MosaicML
     - instructor-large, instructor-xl
     - Yes
   * - llm/v1/embeddings
     - Anthropic
     -
     - No
   * - llm/v1/embeddings
     - Cohere
     - embed-english-v2.0, embed-multilingual-v2.0
     - Yes
   * - llm/v1/embeddings
     - Azure OpenAI
     - text-embedding-ada-002
     - Yes
   * - llm/v1/embeddings
     - Databricks Model Serving
     - Endpoints with compatible schemas
     - Yes

† Llama 2 is licensed under the [LLAMA 2 Community License](https://ai.meta.com/llama/license/), Copyright © Meta Platforms, Inc. All Rights Reserved. 

When creating a route, the provider field is used to specify the name
of the provider for that model. This is a string value that needs to correspond to a provider
the Databricks AI Gateway supports.

Here's an example demonstrating how a provider is specified when creating a route with the
:py:func:`mlflow.gateway.create_route` API:

.. code-block:: python

    create_route(
        name="chat",
        route_type="llm/v1/chat",
        model={
            "name": "gpt-4",
            "provider": "openai",
            "openai_config": {
                "openai_api_key": "<YOUR_OPENAI_API_KEY>"
            }
        }
    )

In the above example, ``openai`` is the `provider` for the model.

As of now, the Databricks AI Gateway supports the following providers:

* **openai**: This is used for models offered by `OpenAI <https://platform.openai.com/>`_ and the `Azure <https://learn.microsoft.com/en-gb/azure/cognitive-services/openai/>`_ integrations for Azure OpenAI and Azure OpenAI with AAD.
* **mosaicml**: This is used for models offered by `MosaicML <https://docs.mosaicml.com/en/latest/>`_.
* **anthropic**: This is used for models offered by `Anthropic <https://docs.anthropic.com/claude/docs>`_.
* **cohere**: This is used for models offered by `Cohere <https://docs.cohere.com/docs>`_.
* **databricks-model-serving**: This is used for Databricks Model Serving endpoints with compatible schemas. See :ref:`config_databricks_model_serving`.

More providers are being added continually. Check the latest version of the Databricks AI Gateway Docs for the
most up-to-date list of supported providers.

Remember, the provider you specify must be one that the Databricks AI Gateway supports. If the provider
is not supported, the Gateway will return an error when trying to route requests to that provider.

.. _routes:

Routes
------

`Routes` are central to how the Databricks AI Gateway functions. Each route acts as a proxy endpoint for the
user, forwarding requests to its configured :ref:`provider <providers>`.

A route in the Databricks AI Gateway consists of the following fields:

* **name**: This is the unique identifier for the route. This will be part of the URL when making API calls via the Databricks AI Gateway.

* **route_type**: The type of the route corresponds to the type of language model interaction you desire. For instance, ``llm/v1/completions`` for text completion operations, ``llm/v1/embeddings`` for text embeddings, and ``llm/v1/chat`` for chat operations.

  - "llm/v1/completions"
  - "llm/v1/chat"
  - "llm/v1/embeddings"

* **model**: Defines the model to which this route will forward requests. The model contains the following details:

    * **provider**: Specifies the name of the :ref:`provider <providers>` for this model. For example, ``openai`` for OpenAI's ``GPT-3.5`` models.

      - "openai"
      - "mosaicml"
      - "anthropic"
      - "cohere"
      - "azure" / "azuread"

    * **name**: The name of the model to use. For example, ``gpt-3.5-turbo`` for OpenAI's ``GPT-3.5-Turbo`` model.
    * **config**: Contains any additional configuration details required for the model. This includes specifying the API base URL and the API key. See :ref:`configure_route_provider`.

  .. important::

      When specifying a model, it is critical that the provider supports the model you are requesting.
      For instance, ``openai`` as a provider supports models like ``text-embedding-ada-002``, but other providers
      may not. If the model is not supported by the provider, the Databricks AI Gateway will return an HTTP 4xx error
      when trying to route requests to that model.

Remember, the model you choose directly affects the results of the responses you'll get from the
API calls. Therefore, choose a model that fits your use-case requirements. For instance,
for generating conversational responses, you would typically choose a chat model.
Conversely, for generating embeddings of text, you would choose an embedding model.

Here's an example of route creation with the :py:func:`mlflow.gateway.create_route` API:

.. code-block:: python

    create_route(
        name="embeddings",
        route_type="llm/v1/embeddings",
        model={
            "name": "text-embedding-ada-002",
            "provider": "openai",
            "openai_config": {
                "openai_api_key": "<YOUR_OPENAI_API_KEY>"
            }
        }
    )

In the example above, a request sent to the embeddings route would be forwarded to the
``text-embedding-ada-002`` model provided by ``openai``.

.. _configure_route_provider:

Configuring the Provider for a Route
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When creating a Route, it's important to supply the required configurations for the specified
:ref:`provider <providers>`. This section provides an overview of the configuration parameters
available for each provider.

Provider-Specific Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenAI
++++++

+-------------------------+----------+-------------------------------+-------------------------------------------------------------+
| Configuration Parameter | Required | Default                       | Description                                                 |
+=========================+==========+===============================+=============================================================+
| **openai_api_key**      | Yes      |                               | This is the API key for the OpenAI service.                 |
+-------------------------+----------+-------------------------------+-------------------------------------------------------------+
| **openai_api_type**     | No       |                               | This is an optional field to specify the type of OpenAI API |
|                         |          |                               | to use.                                                     |
+-------------------------+----------+-------------------------------+-------------------------------------------------------------+
| **openai_api_base**     | No       | `https://api.openai.com/v1`   | This is the base URL for the OpenAI API.                    |
+-------------------------+----------+-------------------------------+-------------------------------------------------------------+
| **openai_api_version**  | No       |                               | This is an optional field to specify the OpenAI API         |
|                         |          |                               | version.                                                    |
+-------------------------+----------+-------------------------------+-------------------------------------------------------------+
| **openai_organization** | No       |                               | This is an optional field to specify the organization in    |
|                         |          |                               | OpenAI.                                                     |
+-------------------------+----------+-------------------------------+-------------------------------------------------------------+

MosaicML
+++++++++

+-------------------------+----------+--------------------------+-------------------------------------------------------+
| Configuration Parameter | Required | Default                  | Description                                           |
+=========================+==========+==========================+=======================================================+
| **mosaicml_api_key**    | Yes      | N/A                      | This is the API key for the MosaicML service.         |
+-------------------------+----------+--------------------------+-------------------------------------------------------+


Cohere
++++++

+-------------------------+----------+--------------------------+-------------------------------------------------------+
| Configuration Parameter | Required | Default                  | Description                                           |
+=========================+==========+==========================+=======================================================+
| **cohere_api_key**      | Yes      | N/A                      | This is the API key for the Cohere service.           |
+-------------------------+----------+--------------------------+-------------------------------------------------------+


Anthropic
+++++++++

+-------------------------+----------+--------------------------+-------------------------------------------------------+
| Configuration Parameter | Required | Default                  | Description                                           |
+=========================+==========+==========================+=======================================================+
| **anthropic_api_key**   | Yes      | N/A                      | This is the API key for the Anthropic service.        |
+-------------------------+----------+--------------------------+-------------------------------------------------------+

Azure OpenAI
++++++++++++

Azure provides two different mechanisms for integrating with OpenAI, each corresponding to a different type of security validation. One relies on an access token for validation, referred to as ``azure``, while the other uses Azure Active Directory (Azure AD) integration for authentication, termed as ``azuread``.

To match your user's interaction and security access requirements, adjust the ``openai_api_type`` parameter to represent the preferred security validation model. This will ensure seamless interaction and reliable security for your Azure-OpenAI integration.

+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+
| Configuration Parameter    | Required | Default | Description                                                                                   |
+============================+==========+=========+===============================================================================================+
| **openai_api_key**         | Yes      |         | This is the API key for the Azure OpenAI service.                                             |
+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+
| **openai_api_type**        | Yes      |         | This field must be either ``azure`` or ``azuread`` depending on the security access protocol. |
+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+
| **openai_api_base**        | Yes      |         | This is the base URL for the Azure OpenAI API service provided by Azure.                      |
+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+
| **openai_api_version**     | Yes      |         | The version of the Azure OpenAI service to utilize, specified by a date.                      |
+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+
| **openai_deployment_name** | Yes      |         | This is the name of the deployment resource for the Azure OpenAI service.                     |
+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+
| **openai_organization**    | No       |         | This is an optional field to specify the organization in OpenAI.                              |
+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+

The following example demonstrates how to create a route with Azure OpenAI:

.. code-block:: python

    create_route(
        name="completions",
        route_type="llm/v1/completions",
        model={
            "name": "gpt-35-turbo",
            "provider": "openai",
            "openai_config": {
                "openai_api_type": "azuread"
                "openai_api_key": "<YOUR_AZURE_OPENAI_API_KEY>"
                "openai_deployment_name": "{your_azure_openai_deployment_name}"
                "openai_api_base": "https://{your_azure_openai_resource_name}-azureopenai.openai.azure.com/"
                "openai_api_version": "2023-05-15"
            }
        }
    )

.. note::

    Azure OpenAI has distinct features as compared with the direct OpenAI service. For an overview, please see `the comparison documentation <https://learn.microsoft.com/en-gb/azure/cognitive-services/openai/how-to/switching-endpoints>`_.

.. _databricks_serving_provider_fields:

Databricks Model Serving (open source models)
+++++++++++++++++++++++++++++++++++++++++++++

+-------------------------------+----------+--------------------------+-------------------------------------------------------+
| Configuration Parameter       | Required | Default                  | Description                                           |
+===============================+==========+==========================+=======================================================+
|                               |          |                          | A Databricks access token corresponding to a user or  |
| **databricks_api_token**      | Yes      | N/A                      | service principal that has **Can Query** access to the|
|                               |          |                          | Model Serving endpoint associated with the route.     |
+-------------------------------+----------+--------------------------+-------------------------------------------------------+
| **databricks_workspace_url**  | Yes      | N/A                      | The URL of the workspace containing the Model Serving |
|                               |          |                          | endpoint associated with the route.                   |
+-------------------------------+----------+--------------------------+-------------------------------------------------------+

The following example demonstrates how to create a route with a Databricks Model Serving endpoint:

.. code-block:: python

    create_route(
        name="databricks-completions",
        route_type="llm/v1/completions",
        model={
            "name": "mpt-7b-instruct",
            "provider": "databricks-model-serving",
            "openai_config": {
                "databricks_api_token": "<YOUR_DATABRICKS_ACCESS_TOKEN>"
                "databricks_workspace_url": "<URL_OF_DATABRICKS_WORKSPACE_CONTAINING_ENDPOINT>"
            }
        }
    )

For more information about creating routes with Databricks Model Serving endpoints, see :ref:`config_databricks_model_serving`.

.. _rate_limits:

Rate Limits on AI Gateway Routes
================================

The parameters for define a rate limit are:

+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| Limit Parameter               | Type           | Required | Default       | Description                                           |
+===============================+================+==========+===============+=======================================================+
| **key**                       | string         | No       | route         | The limit key defines whether the rate limit is per   |
|                               |                |          |               | databricks user or per route (across all users).      |
|                               |                |          |               | Allowed options are: user, route.                     |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| **calls**                     | integer        | Yes      | N/A           | The maximum total number of calls allowed during the  |
|                               |                |          |               | time interval specified in renewal_period.            |
|                               |                |          |               | Must be non-negative integer.                         |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| **renewal_period**            | string         | Yes      | N/A           | The length in seconds of the sliding window during    |
|                               |                |          |               | which the number of allowed requests should not exceed|
|                               |                |          |               | the value specified in calls. Allowed option: minute. |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+

The following example demonstrates how to set a per user limit and per route limit on an existing route and how to get existing limits of a route:

.. code-block:: python
    
    from mlflow.gateway import set_limits, get_limits

    set_limits(
        route="my-route",
        limits=[
            // You can define multiple limits on a route
            {
                // 5 calls per user per minute
                "key": "user",
                "calls": 5,
                "renewal_period": "minute"
            },
            {
                // 50 calls per minute for all users
                "calls": 50,
                "renewal_period": "minute"
            }
        ]
    )

    get_limits(
        route="my-route"
    )

For more details on how to set limits and get limits via APIs, please see :ref:`gateway_fluent_api`, :ref:`gateway_client_api` and :ref:`gateway_rest_api`.

.. _gateway_query:

Querying the AI Gateway
=======================

Once the Databricks AI Gateway server has been configured and started, it is ready to receive traffic from users.

.. _standard_query_parameters:

Standard Query Parameters
-------------------------

The Databricks AI Gateway defines standard parameters for chat, completions, and embeddings that can be
used when querying any route regardless of its provider. Each parameter has a standard range and
default value. When querying a route with a particular provider, the Databricks AI Gateway automatically
scales parameter values according to the provider's value ranges for that parameter.

.. important::

  When querying an AI Gateway Route with the ``databricks-model-serving`` provider, some of the
  the standard query parameters may be ignored depending on whether or not the Databricks Model
  Serving endpoint supports them. All of the parameters marked **required** are guaranteed to
  be supported. For more information, see :ref:`config_databricks_model_serving`.

Completions
~~~~~~~~~~~

The standard parameters for completions routes with type ``llm/v1/completions`` are:

+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| Query Parameter               | Type           | Required | Default       | Description                                           |
+===============================+================+==========+===============+=======================================================+
| **prompt**                    | string         | Yes      | N/A           | The prompt for which to generate completions.         |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| **candidate_count**           | integer        | No       | 1             | The number of completions to generate for the         |
|                               |                |          |               | specified prompt, between 1 and 5.                    |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| **temperature**               | float          | No       | 0.0           | The sampling temperature to use, between 0 and 1.     |
|                               |                |          |               | Higher values will make the output more random, and   |
|                               |                |          |               | lower values will make the output more deterministic. |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| **max_tokens**                | integer        | No       | infinity      | The maximum completion length, between 1 and infinity |
|                               |                |          |               | (unlimited).                                          |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| **stop**                      | array[string]  | No       | []            | Sequences where the model should stop generating      |
|                               |                |          |               | tokens and return the completion.                     |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+

Chat
~~~~

The standard parameters for completions routes with type ``llm/v1/chat`` are:

+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| Query Parameter               | Type           | Required | Default       | Description                                           |
+===============================+================+==========+===============+=======================================================+
| **messages**                  | array[message] | Yes      | N/A           | A list of messages in a conversation from which to    |
|                               |                |          |               | a new message (chat completion). For information      |
|                               |                |          |               | about the message structure, see                      |
|                               |                |          |               | :ref:`chat_message_structure`.                        |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| **candidate_count**           | integer        | No       | 1             | The number of chat completions to generate for the    |
|                               |                |          |               | specified prompt, between 1 and 5.                    |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| **temperature**               | float          | No       | 0.0           | The sampling temperature to use, between 0 and 1.     |
|                               |                |          |               | Higher values will make the output more random, and   |
|                               |                |          |               | lower values will make the output more deterministic. |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| **max_tokens**                | integer        | No       | infinity      | The maximum completion length, between 1 and infinity |
|                               |                |          |               | (unlimited).                                          |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| **stop**                      | array[string]  | No       | []            | Sequences where the model should stop generating      |
|                               |                |          |               | tokens and return the chat completion.                |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+

.. _chat_message_structure:

Messages
^^^^^^^^

Each chat message is a string dictionary containing the following fields:

+-------------------------------+----------+--------------------------+-------------------------------------------------------+
| Field Name                    | Required | Default                  | Description                                           |
+===============================+==========+==========================+=======================================================+
| **role**                      | Yes      | N/A                      | The role of the conversation participant who sent the |
|                               |          |                          | message. Must be one of: ``"system"``, ``"user"``, or |
|                               |          |                          | ``"assistant"``.                                      |
+-------------------------------+----------+--------------------------+-------------------------------------------------------+
| **content**                   | Yes      | N/A                      | The message content.                                  |
+-------------------------------+----------+--------------------------+-------------------------------------------------------+

Embeddings
~~~~~~~~~~

The standard parameters for completions routes with type ``llm/v1/embeddings`` are:

+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| Query Parameter               | Type           | Required | Default       | Description                                           |
+===============================+================+==========+===============+=======================================================+
| **text**                      | string         | Yes      | N/A           | A string or list of strings for which to generate     |
|                               | or             |          |               | embeddings.                                           |
|                               | array[string]  |          |               |                                                       |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+

Additional Query Parameters
---------------------------
In addition to the :ref:`standard_query_parameters`, you can pass any additional parameters supported by the route's provider as part of your query. For example:

- ``logit_bias`` (supported by OpenAI, Cohere)
- ``top_k`` (supported by MosaicML, Anthropic, Cohere)
- ``frequency_penalty`` (supported by OpenAI, Cohere)
- ``presence_penalty`` (supported by OpenAI, Cohere)

The following parameters are not allowed:

- ``stream`` is not supported. Setting this parameter on any provider will not work currently.

Below is an example of submitting a query request to an Databricks AI Gateway route using additional parameters:

.. code-block:: python

    data = {
        "prompt": (
            "What would happen if an asteroid the size of "
            "a basketball encountered the Earth traveling at 0.5c?"
        ),
        "temperature": 0.5,
        "max_tokens": 1000,
        "candidate_count": 1,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.2,
    }

    query(route="completions-gpt4", data=data)

The results of the query are:

.. code-block:: json

       {
         "candidates": [
           {
             "text": "If an asteroid the size of a basketball (roughly 24 cm in
             diameter) were to hit the Earth at 0.5 times the speed of light
             (approximately 150,000 kilometers per second), the energy released
             on impact would be enormous. The kinetic energy of an object moving
             at relativistic speeds is given by the formula: KE = (\\gamma - 1)
             mc^2 where \\gamma is the Lorentz factor given by...",
             "metadata": {
               "finish_reason": "stop"
             }
           }
         ],
         "metadata": {
           "input_tokens": 40,
           "output_tokens": 622,
           "total_tokens": 662,
           "model": "gpt-4-0613",
           "route_type": "llm/v1/completions"
         }
       }

MLflow Python Client APIs
-------------------------
:class:`MlflowGatewayClient <mlflow.gateway.client.MlflowGatewayClient>` is the user-facing client API that is used to interact with the Databricks AI Gateway.
It abstracts the HTTP requests to the Gateway via a simple, easy-to-use Python API.

The fluent API is a higher-level interface that supports setting the Gateway URI once and using simple functions to interact with the AI Gateway.

.. _gateway_fluent_api:

Fluent API
~~~~~~~~~~
For the ``fluent`` API, here are some examples:

1. Set the Gateway URI:

   Before using the Fluent API, the gateway URI must be set via :func:`set_gateway_uri() <mlflow.gateway.set_gateway_uri>`.

   If you are using the AI Gateway in a Databricks Notebook or Databricks Job, you can set the gateway URI as follows:

   .. code-block:: python

       from mlflow.gateway import set_gateway_uri

       set_gateway_uri(gateway_uri="databricks")

   If you are using the AI Gateway outside of a Databricks Notebook or Databricks Job, you will need to configure 
   your Databricks host name and Databricks access token in your current environment before making requests to
   the Gateway. You can do this using the ``DATABRICKS_HOST`` and ``DATABRICKS_TOKEN`` environment variables.
   For example:

   .. code-block:: python

       import os
       from mlflow.gateway import set_gateway_uri

       os.environ["DATABRICKS_HOST"] = "http://your.workspace.databricks.com"
       os.environ["DATABRICKS_TOKEN"] = "<your_databricks_access_token>"

       set_gateway_uri(gateway_uri="databricks")

   Finally, you can also set the gateway URI using the ``MLFLOW_GATEWAY_URI`` environment variable, as an alternative
   to calling :func:`set_gateway_uri() <mlflow.gateway.set_gateway_uri>`.

2. Query a route:

   The :func:`query() <mlflow.gateway.query>` function queries the specified route and returns the response from the provider
   in a standardized format. The data structure you send in the query depends on the route.

   .. code-block:: python

       from mlflow.gateway import query

       response = query(
           "embeddings", {"text": ["It was the best of times", "It was the worst of times"]}
       )
       print(response)

3. Set rate limtis on a route:

   The :func:`set_limits() <mlflow.gateway.set_limits>` function set rate limits on a route.
   The data structure you send in the query is an array of limits, see :ref:`rate_limits`.

   .. code-block:: python

       from mlflow.gateway import set_limits

       response = set_limits(
           route = "my-route",
           limits = [
            {
                "key": "user",
                "calls": 5,
                "renewal_period": "minute"
            },
            {
                "calls": 50,
                "renewal_period": "minute"
            }
           ]
       )
       print(response)

4. Get rate limtis of a route:

   The :func:`get_limits() <mlflow.gateway.get_limits>` function set rate limits on a route.
   The data structure returned is an array of limits, see :ref:`rate_limits`.

   .. code-block:: python

       from mlflow.gateway import get_limits

       response = get_limits(
           route = "my-route"
       )
       print(response)

.. _gateway_client_api:

Client API
~~~~~~~~~~

To use the ``MlflowGatewayClient`` API, see the below examples for the available API methods:

1. Create an ``MlflowGatewayClient``

   If you are using the AI Gateway in a Databricks Notebook or Databricks Job, you can initialize
   the ``MlflowGatewayClient`` as follows:

   .. code-block:: python

       from mlflow.gateway import MlflowGatewayClient

       gateway_client = MlflowGatewayClient("databricks")

   If you are using the AI Gateway outside of a Databricks Notebook or Databricks Job, you will need to configure
   your Databricks host name and Databricks access token in your current environment before making requests to
   the Gateway. You can do this using the ``DATABRICKS_HOST`` and ``DATABRICKS_TOKEN`` environment variables.
   For example:

   .. code-block:: python

       import os
       from mlflow.gateway import MlflowGatewayClient


       os.environ["DATABRICKS_HOST"] = "http://your.workspace.databricks.com"
       os.environ["DATABRICKS_TOKEN"] = "<your_databricks_access_token>"

       gateway_client = MlflowGatewayClient("databricks")

2. List all routes:

   The :meth:`search_routes() <mlflow.gateway.client.MlflowGatewayClient.search_routes>` method returns a list of all routes.

   .. code-block:: python

       routes = gateway_client.search_routes()
       for route in routes:
           print(route)

3. Query a route:

   The :meth:`query() <mlflow.gateway.client.MlflowGatewayClient.query>` method submits a query to a configured provider route.
   The data structure you send in the query depends on the route.

   .. code-block:: python

       response = gateway_client.query(
           "chat", {"messages": [{"role": "user", "content": "Tell me a joke about rabbits"}]}
       )
       print(response)

4. Set rate limits on a route:
   
   The :meth:`set_limits() <mlflow.gateway.client.MlflowGatewayClient.set_limits>` method set rate limits on a route.
   The data structure you send is an array of limits, see :ref:`rate_limits`.

   .. code-block:: python

       response = gateway_client.set_limits(
           route = "my-route",
           limits = [
            {
                "key": "user",
                "calls": 5,
                "renewal_period": "minute"
            },
            {
                "calls": 50,
                "renewal_period": "minute"
            }
           ]
       )
       print(response)

5. Get rate limits of a route:
   
   The :meth:`get_limits() <mlflow.gateway.client.MlflowGatewayClient.get_limits>` method returns all rate limits of a route.
   The data structure returned is an array of limits, see :ref:`rate_limits`.

   .. code-block:: python

       response = gateway_client.get_limits(
           route = "my-route",
       )
       print(response)

Further route types will be added in the future.

.. _gateway_mlflow_models:

MLflow Models
~~~~~~~~~~~~~
You can also build and deploy MLflow Models that call the Databricks AI Gateway.
The example below demonstrates how to use an AI Gateway server from within a custom ``pyfunc`` model.

.. code-block:: python

    import os
    import pandas as pd
    import mlflow


    def predict(data):
        from mlflow.gateway import MlflowGatewayClient

        client = MlflowGatewayClient("databricks")
        prompt = "Translate the following input text from English to French: {input_text}"

        payload = data.to_dict(orient="records")
        return [
            client.query(
                route="completions-claude",
                data={
                    "prompt": prompt.format(input_text=input_text)
                }
            )["candidates"][0]["text"]
            for input_text in payload
        ]


    input_example = pd.DataFrame.from_dict(
        {"prompt": ["What is an LLM?", "AI is cool!"]}
    )
    signature = mlflow.models.infer_signature(
        input_example, ["Qu'est-ce qu'un LLM?", "L'IA est propre!"]
    )

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=predict,
            registered_model_name="anthropic_french_translator",
            artifact_path="anthropic_french_translator",
            input_example=input_example,
            signature=signature,
        )

    df = pd.DataFrame.from_dict(
        {
            "prompt": ["I like machine learning", "MLflow is awesome!"],
            "temperature": 0.6,
            "max_records": 500,
        }
    )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    print(loaded_model.predict(df))

This custom MLflow model can be used in the same way as any other MLflow model. It can be used within a ``spark_udf``, used with :func:`mlflow.evaluate`, or `deploy <https://mlflow.org/docs/latest/models.html#built-in-deployment-tools>`_ like any other model.

LangChain Integration
~~~~~~~~~~~~~~~~~~~~~
LangChain has `an integration for MLflow AI Gateway <https://python.langchain.com/docs/ecosystem/integrations/mlflow_ai_gateway>`_.
This integration enable users to use prompt engineering, retrieval augmented generation, and other
techniques with LLMs in the gateway.

.. code-block:: python

    import mlflow
    from langchain import LLMChain, PromptTemplate
    from langchain.llms import MlflowAIGateway

    gateway = MlflowAIGateway(
        gateway_uri="databricks",
        route="completions",
        params={
            "temperature": 0.0,
            "top_p": 0.1,
        },
    )

    llm_chain = LLMChain(
        llm=gateway,
        prompt=PromptTemplate(
            input_variables=["adjective"],
            template="Tell me a {adjective} joke",
        ),
    )
    result = llm_chain.run(adjective="funny")
    print(result)

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(chain, "model")

    model = mlflow.pyfunc.load_model(model_info.model_uri)
    print(model.predict([{"adjective": "funny"}]))


.. _gateway_query_serving_endpoint:

Querying the AI Gateway from Databricks Model Serving Endpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once you have defined an :ref:`MLflow Model <gateway_mlflow_models>` that queries the AI Gateway,
you can deploy it to Databricks Model Serving. This enables you to deploy custom application
logic that depends on one or more LLMs in the AI Gateway, such as constructing an
application-specific prompt in response to user input and using it to query an LLM.

Building on the previous :ref:`gateway_mlflow_models` example, run the following steps
to deploy an MLflow Model that uses the AI Gateway to Databricks Model Serving:

1. Log and register your model with the following additional dependencies - ``pydantic<2`` and
   ``psutil``:

   .. code-block:: python

       with mlflow.start_run():
           model_info = mlflow.pyfunc.log_model(
               python_model=predict,
               registered_model_name="anthropic_french_translator",
               artifact_path="anthropic_french_translator",
               input_example=input_example,
               signature=signature,
               extra_pip_requirements=[
                   "pydantic<2",
                   "psutil"
               ]
           )

2. Follow the guide at
   https://docs.databricks.com/machine-learning/model-serving/store-env-variable-model-serving.html
   to create a Databricks Model Serving endpoint for your MLflow Registered Model with the
   following environment variables set:

   * ``DATABRICKS_HOST``: The URL of the Databricks workspace containing the AI Gateway route
     that your MLflow Model queries.
   * ``DATABRICKS_TOKEN``: A Databricks access token corresponding to a user or service principal
     with permission to query the AI Gateway route referenced by the MLflow Model.

   For example:

   .. code-block:: bash

       PUT /api/2.0/serving-endpoints/{name}/config

       {
           "served_models": [{
               "model_name": "anthropic_french_translator",
               "model_version": "1",
               "workload_size": "Small",
               "scale_to_zero_enabled": true,
               "env_vars": [
                   {
                       "env_var_name": "DATABRICKS_HOST"
                       "secret_scope": "my_secret_scope",
                       "secret_key": "my_databricks_host_secret_key"
                   },
                   {
                       "env_var_name": "DATABRICKS_TOKEN"
                       "secret_scope": "my_secret_scope",
                       "secret_key": "my_databricks_token_secret_key"
                   }
               ]
           }]
        }

.. _gateway_rest_api:

REST API
~~~~~~~~
The REST API allows you to send HTTP requests directly to the Databricks AI Gateway server. This is useful if you're not using Python or if you prefer to interact with the Gateway using HTTP directly.

Here are some examples for how you might use curl to interact with the Gateway:

1. Getting information about a particular route: ``GET /api/2.0/gateway/routes/{name}``

   This endpoint returns a serialized representation of the Route data structure.
   This provides information about the name and type, as well as the model details for the requested route endpoint.

   .. code-block:: bash

       curl \
         -X GET \
         -H "Authorization: Bearer <your_databricks_access_token>" \
         http://your.workspace.databricks.com/api/2.0/gateway/routes/<your_route_name>

   **Note:** Remember to replace ``<your_databricks_access_token>`` with your Databricks access token, ``http://your.workspace.databricks.com/``
   with your Databricks workspace URL, and ``<your_route_name>`` with your route name.

2. List all routes: ``GET /api/2.0/gateway/routes``

   This endpoint returns a list of all routes.

   .. code-block:: bash

       curl \
         -X GET \
         -H "Authorization: Bearer <your_databricks_access_token>" \
         http://your.workspace.databricks.com/api/2.0/gateway/routes

3. Querying a particular route: ``POST /gateway/{route}/invocations``

   This endpoint allows you to submit a query to a specified route. The data structure you send in the query depends on the route. Here are examples for the "completions", "chat", and "embeddings" routes:

   * ``Completions``

     .. code-block:: bash

         curl \
           -X POST \
           -H "Content-Type: application/json" \
           -H "Authorization: Bearer <your_databricks_access_token>" \
           -d '{"prompt": "Describe the probability distribution of the decay chain of U-235"}' \
           http://your.workspace.databricks.com/gateway/<your_completions_route>/invocations

   * ``Chat``

     .. code-block:: bash

         curl \
           -X POST \
           -H "Content-Type: application/json" \
           -H "Authorization: Bearer <your_databricks_access_token>" \
           -d '{"messages": [{"role": "user", "content": "Can you write a limerick about orange flavored popsicles?"}]}' \
           http://your.workspace.databricks.com/gateway/<your_chat_route>/invocations

   * ``Embeddings``

     .. code-block:: bash

         curl \
           -X POST \
           -H "Content-Type: application/json" \
           -H "Authorization: Bearer <your_databricks_access_token>" \
           -d '{"text": ["I'd like to return my shipment of beanie babies, please", "Can I please speak to a human now?"]}' \
           http://your.workspace.databricks.com/gateway/<your_embeddings_route>/invocations

4. Set rate limits on a route: ``POST /api/2.0/gateway/limits``

   This endpoint allows you to set rate limits on an AI Gateway Route.

    .. code-block:: bash

         curl \
           -X POST \
           -H "Content-Type: application/json" \
           -H "Authorization: Bearer <your_databricks_access_token>" \
           -d '{"route": "my-route", "limits": [{"key": "user", "calls": 5, "renewal_period": "minute"}]}' \
           http://your.workspace.databricks.com/api/2.0/gateway/limits

5. Get rate limits of a route: ``GET /api/2.0/gateway/limits/{route}``

   This endpoint allows you to get rate limits of an AI Gateway Route.

    .. code-block:: bash

         curl \
           -X GET \
           -H "Authorization: Bearer <your_databricks_access_token>" \
         http://your.workspace.databricks.com/api/2.0/gateway/limits/<your_route>

Using MosaicML-hosted open source models with the AI Gateway
=================================================================================
AI Gateway also provides access to MosaicML’s open source models as hosted APIs. 
These APIs provide fast and easy access to state-of-the-art open source models for rapid experimentation and 
token-based pricing. MosaicML supports the ``Instructor-XL``, a 1.2B parameter instruction fine-tuned embedding model 
by HKUNLP, and the ``Llama2-70b-Chat``† API which was trained on 2 trillion tokens and fine-tuned for dialogue, safety, and 
helpfulness by Meta.

.. note::

    Llama 2 is licensed under the LLAMA 2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.


To access these models via Gateway, you can create MosaicML routes like for the other providers.
The following example demonstrates how to create a route with MosaicML (for the Llama2-70b-Chat† model):

.. code-block:: python

    create_route(
        name="mosaicml-llama-completions",
        route_type="llm/v1/completions",
        model={
            "name": "llama2-70b-chat",
            "provider": "mosaicml",
            "mosaicml_config": {
                "mosaicml_api_key": "<YOUR_MOSAIC_API_KEY>"
            }
        }
    )

For the ``Instructor-XL`` embeddings model, the route can be created like so:

.. code-block:: python

    create_route(
            name="mosaicml-embeddings",
            route_type="llm/v1/embeddings",
            model={
                "name": "instructor-xl",
                "provider": "mosaicml",
                "mosaicml_config": {
                    "mosaicml_api_key": "<YOUR_MOSAIC_API_KEY>"
                }
            }
    )


To query these routes, you can use the :ref:`gateway_fluent_api`, for instance:

.. code-block:: python

    from mlflow.gateway import query

    response = query(
        route="mosaicml-llama-completions",
        data={
            "prompt": "What is MLflow?",
        }
    )
    print(response)

The :ref:`gateway_rest_api` can also be used.


.. _config_databricks_model_serving:

Using open source models with the AI Gateway (Databricks Model Serving Endpoints)
=================================================================================
The Databricks AI Gateway supports `Databricks Model Serving endpoints <https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html>`_
as providers for the ``llm/v1/completions`` route type. These endpoints must accept the
:ref:`standard_query_parameters` that are marked **required**, and they must produce responses
in the following format:

.. code-block:: json

       {
         "candidates": [
            "Completion 1 text",
            "Completion 2 text",
            "..."
         ]
       }

For a detailed example of creating a Databricks Model Serving endpoint with a compatible
:ref:`MLflow Model Signature <model-signature>` and querying it through the AI Gateway,
see :ref:`gateway_databricks_model_serving_completions_example`.

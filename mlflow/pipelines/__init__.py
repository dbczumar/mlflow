# pylint: disable=line-too-long

"""
MLflow Pipelines is an opinionated framework for structuring MLOps workflows that simplifies and
standardizes machine learning application development and productionization. MLflow Pipelines
makes it easy for data scientists to follow best practices for creating production-ready ML
deliverables, allowing them to focus on developing excellent models. MLflow Pipelines also enables
ML engineers and DevOps teams to seamlessly deploy these models to production and incorporate them
into applications.

MLflow Pipelines provides production-quality :ref:`Pipeline Templates <pipeline-templates>` for
common ML problem types, such as regression & classification, and MLOps tasks, such as batch
scoring. Pipelines are structured as git repositories with YAML-based configuration files and
Python code, offering developers a declarative approach to ML application development that reduces
boilerplate.

MLflow Pipelines also implements a cache-aware executor for pipeline steps, ensuring that steps
are only executed when associated
:py:ref:`code or configurations <pipeline-repositories-key-concept>` have changed. This enables
data scientists, ML engineers, and DevOps teams to iterate very quickly within their domains of
expertise. MLflow offers |run() APIs| for executing pipelines, as well as an
|mlflow pipelines run CLI|.

For more information, see the :ref:`MLflow Pipelines Overview <pipelines>`.

.. |mlflow pipelines run CLI| replace:: :ref:`mlflow pipelines run <cli>` CLI
.. |run() APIs| replace:: :py:func:`run() <mlflow.pipelines.regression.v1.pipeline.RegressionPipeline.run>` APIs
"""

# pylint: enable=line-too-long

from IPython import get_ipython
from IPython.core.magic import register_cell_magic
from pathlib import Path

from mlflow.pipelines.pipeline import Pipeline

__all__ = ["Pipeline"]


def mlp_code(file_path, cell):
    import os
    from mlflow.exceptions import MlflowException
    from mlflow.pipelines.decorators import _set_mlp_code_path
    from mlflow.pipelines.utils import get_pipeline_root_path

    if os.path.isabs(file_path):
        raise MlflowException("Code must be written within the pipeline repository")

    _set_mlp_code_path(file_path)

    pipeline_root_path = get_pipeline_root_path()
    file_path = os.path.join(pipeline_root_path, file_path) 

    get_ipython().run_cell(cell).raise_error()
    
    # Check to see if there's any difference in file content with OG train.py
    # before overwriting so that Make doesn't have problems...

    _set_mlp_code_path(None)
    
    if os.path.exists(file_path):
      with open(file_path, "r") as f:
        content = f.read()
        if content == cell:
          return
  
    with open(file_path, "w") as f:
        f.write(cell)


def load_ipython_extension(ipython):
    """This function is called when the extension is
    loaded. It accepts an IPython InteractiveShell
    instance. We can register the magic with the
    `register_magic_function` method of the shell
    instance."""
    ipython.register_magic_function(mlp_code, 'cell')


# class Conf:
#
#     def __init__(name):
#
#
# def conf(name):


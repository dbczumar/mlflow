import os
from subprocess import Popen, PIPE, STDOUT

import mlflow

from mlflow.utils.logging_utils import eprint
from mlflow.utils.file_utils import _copy_project

_DOCKERFILE_TEMPLATE = """
# Build an image that can serve pyfunc model in SageMaker
FROM ubuntu:16.04

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         curl \
         nginx \
         ca-certificates \
         bzip2 \
         build-essential \
         cmake \
         openjdk-8-jdk \
         git-core \
         maven \
    && rm -rf /var/lib/apt/lists/*

# Download and setup miniconda
RUN curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh >> miniconda.sh
RUN bash ./miniconda.sh -b -p /miniconda; rm ./miniconda.sh;
ENV PATH="/miniconda/bin:${PATH}"
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

RUN conda install -c anaconda gunicorn;\
    conda install -c anaconda gevent;\

%s

# Set up the program in the image
WORKDIR /opt/mlflow

# start mlflow scoring
ENTRYPOINT ["python", "-c", "import sys; from mlflow.sagemaker import container as C; \
C._init(sys.argv[1])"]
"""

def get_template(image_resources_path, mlflow_home=None):
    install_mlflow = "RUN pip install mlflow=={version}".format(
        version=mlflow.version.VERSION)
    if mlflow_home:
        mlflow_dir = _copy_project(
            src_path=mlflow_home, dst_path=image_resources_path)
        install_mlflow = ("COPY {mlflow_dir} /opt/mlflow\n"
                          "RUN cd /opt/mlflow/mlflow/java/scoring &&"
                          " mvn --batch-mode package -DskipTests \n"
                          "RUN pip install /opt/mlflow\n")
        install_mlflow = install_mlflow.format(mlflow_dir=mlflow_dir)
    else:
        eprint("`mlflow_home` was not specified. The image will install"
               " MLflow from pip instead. As a result, the container will"
               " not support the MLeap flavor.")
    return _DOCKERFILE_TEMPLATE % install_mlflow


def build_image(image_name, template_path):
    template_dir = os.path.dirname(template_path)
    template_fname = os.path.basename(template_path)
    os.system('find {cwd}/'.format(cwd=template_dir))
    eprint("building docker image")
    proc = Popen(["docker", "build", "-t", image_name, "-f", template_fname, "."],
                 cwd=template_dir,
                 stdout=PIPE,
                 stderr=STDOUT,
                 universal_newlines=True)
    for x in iter(proc.stdout.readline, ""):
        eprint(x, end='')


def push_image(image_uri):
    proc = Popen(["docker", "push", image_uri])
    proc.wait()

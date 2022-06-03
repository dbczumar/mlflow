import os
from mlflow.pipelines.cards import BaseCard


class TrainCard(BaseCard):
    def __init__(self, pipeline_name: str, step_name: str):
        super().__init__(
            template_root=os.path.join(os.path.dirname(__file__), "../resources"),
            template_name="train.html",
            pipeline_name=pipeline_name,
            step_name=step_name,
        )

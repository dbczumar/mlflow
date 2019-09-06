import six
from abc import abstractmethod

from mlflow import entities
from mlflow.entities import ViewType, RunTag, SourceType, RunStatus, Experiment, Metric, Param
from mlflow.exceptions import MlflowException


class BackendStoreTest:

    def __init__(self):
        pass

    @property
    @abstractmethod
    def store(self):
        pass

    def test_default_experiment(self):
        experiments = self.store.list_experiments()

        self.assertEqual(len(experiments), 1)
        first = experiments[0]
        self.assertEqual(first.experiment_id, "0")
        self.assertEqual(first.name, "Default")

    def test_default_experiment_lifecycle(self):
        default_experiment = self.store.get_experiment(experiment_id="0")
        self.assertEqual(default_experiment.name, Experiment.DEFAULT_EXPERIMENT_NAME)
        self.assertEqual(default_experiment.lifecycle_stage, entities.LifecycleStage.ACTIVE)

        self.store.create_experiment('aNothEr')
        all_experiments = [e.name for e in self.store.list_experiments()]
        six.assertCountEqual(self, set(['aNothEr', 'Default']), set(all_experiments))

        self.store.delete_experiment("0")

        six.assertCountEqual(self, ['aNothEr'], [e.name for e in self.store.list_experiments()])
        another = self.store.get_experiment("1")
        self.assertEqual('aNothEr', another.name)

        default_experiment = self.store.get_experiment(experiment_id="0")
        self.assertEqual(default_experiment.name, Experiment.DEFAULT_EXPERIMENT_NAME)
        self.assertEqual(default_experiment.lifecycle_stage, entities.LifecycleStage.DELETED)

        fetched_store = self.get_store()

        # test that default experiment is not reactivated
        default_experiment = fetched_store.get_experiment(experiment_id="0")
        self.assertEqual(default_experiment.name, Experiment.DEFAULT_EXPERIMENT_NAME)
        self.assertEqual(default_experiment.lifecycle_stage, entities.LifecycleStage.DELETED)

        six.assertCountEqual(self, ['aNothEr'], [e.name for e in fetched_store.list_experiments()])
        all_experiments = [e.name for e in fetched_store.list_experiments(ViewType.ALL)]
        six.assertCountEqual(self, set(['aNothEr', 'Default']), set(all_experiments))

        # ensure that experiment ID dor active experiment is unchanged
        another = fetched_store.get_experiment("1")
        self.assertEqual('aNothEr', another.name)

    def test_raise_duplicate_experiments(self):
        with self.assertRaises(MlflowException):
            self.store.create_experiment("test")
            self.store.create_experiment("test")

    def test_raise_experiment_dont_exist(self):
        with self.assertRaises(MlflowException):
            self.store.get_experiment(experiment_id="100")

    def test_delete_experiment(self):
        experiments = [
            self.store.create_experiment(exp) for exp in 'morty', 'rick', 'rick and morty'
        ]

        all_experiments = self.store.list_experiments()
        self.assertEqual(len(all_experiments), len(experiments) + 1)  # default

        exp_id = experiments[0]
        self.store.delete_experiment(exp_id)

        updated_exp = self.store.get_experiment(exp_id)
        self.assertEqual(updated_exp.lifecycle_stage, entities.LifecycleStage.DELETED)

        self.assertEqual(len(self.store.list_experiments()), len(all_experiments) - 1)

    def test_get_experiment(self):
        name = 'goku'
        experiment_id = self.store.create_experiment(name)
        actual = self.store.get_experiment(experiment_id)
        self.assertEqual(actual.name, name)
        self.assertEqual(actual.experiment_id, experiment_id)

        actual_by_name = self.store.get_experiment_by_name(name)
        self.assertEqual(actual_by_name.name, name)
        self.assertEqual(actual_by_name.experiment_id, experiment_id)

    def test_list_experiments(self):
        testnames = ['blue', 'red', 'green']
        experiments = [
            self.store.create_experiment(exp) for exp in testnames 
        ]

        actual = self.store.list_experiments()
        self.assertEqual(len(experiments) + 1, len(actual))  # default
        assert set(zip(testnames, experiments)).issubset(
            set([(exp.name, exp.experiment_id) for exp in actual]))

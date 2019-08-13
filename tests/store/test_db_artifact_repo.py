import os

import tempfile
import unittest

from mlflow.store.dbmodels import initial_artifact_store_models
from mlflow.store.db_artifact_repo import DBArtifactRepository
from mlflow.utils.file_utils import TempDir
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
import mlflow
import posixpath

DB_URI = 'sqlite:///'


class TestSqlAlchemyStoreSqlite(unittest.TestCase):

    def _get_store(self, db_uri=''):
        return DBArtifactRepository(db_uri)

    def setUp(self):
        self.maxDiff = None  # print all differences on assert failures
        fd, self.temp_dbfile = tempfile.mkstemp()
        # Close handle immediately so that we can remove the file later on in Windows
        os.close(fd)
        self.db_url = "%s%s" % (DB_URI, self.temp_dbfile)
        self.store = self._get_store(self.db_url)

    def tearDown(self):
        self.store.clean()

    def test_log_artifact(self):
        with TempDir() as tmp_dir:
            local_file = tmp_dir.path('model')
            with open(local_file, "w") as f:
                f.write('DB store Test')

            self.store.log_artifact(local_file, 'more_path/some')
            with self.store.ManagedSessionMaker() as session:
                result = session.query(initial_artifact_store_models.SqlArtifact).all()
                self.assertEqual(len(result), 1)

                test_art = session.query(initial_artifact_store_models.SqlArtifact). \
                    filter_by(artifact_name='model').first()
                self.assertEqual(str(test_art.artifact_id), "1")
                self.assertEqual(test_art.artifact_name, 'model')
                self.assertEqual(test_art.group_path, 'more_path/some')
                self.assertEqual(test_art.artifact_content, open(
                    local_file, "rb").read())

    def test_log_artifact_no_artifact_path(self):
        with TempDir() as tmp_dir:
            local_file = tmp_dir.path('model')
            with open(local_file, "w") as f:
                f.write('DB store Test')

            self.store.log_artifact(local_file)
            with self.store.ManagedSessionMaker() as session:
                result = session.query(initial_artifact_store_models.SqlArtifact).all()
                self.assertEqual(len(result), 1)

                test_art = session.query(initial_artifact_store_models.SqlArtifact). \
                    filter_by(artifact_name='model').first()
                self.assertEqual(str(test_art.artifact_id), "1")
                self.assertEqual(test_art.artifact_name, 'model')
                self.assertEqual(test_art.group_path, '')
                self.assertEqual(test_art.artifact_content, open(
                    local_file, "rb").read())

    def test_log_artifacts(self):
        with TempDir() as root_dir:
            with open(root_dir.path("file_one.txt"), "w") as f:
                f.write('DB store Test One')

            os.mkdir(root_dir.path("subdir"))
            with open(root_dir.path("subdir/file_two.txt"), "w") as f:
                f.write('DB store Test Two')

            self.store.log_artifacts(root_dir._path, 'new_path/path')

            with self.store.ManagedSessionMaker() as session:
                result = session.query(initial_artifact_store_models.SqlArtifact).all()
                self.assertEqual(len(result), 2)

                test_art = session.query(initial_artifact_store_models.SqlArtifact). \
                    filter_by(artifact_name='file_one.txt').first()
                self.assertEqual(str(test_art.artifact_id), "1")
                self.assertEqual(test_art.artifact_name, 'file_one.txt')
                self.assertEqual(test_art.group_path, 'new_path/path')
                self.assertEqual(test_art.artifact_content, open(
                    root_dir.path("file_one.txt"), "rb").read())

                test_art = session.query(initial_artifact_store_models.SqlArtifact). \
                    filter_by(artifact_name='file_two.txt').first()
                self.assertEqual(str(test_art.artifact_id), "2")
                self.assertEqual(test_art.artifact_name, 'file_two.txt')
                self.assertEqual(test_art.group_path, 'new_path/path/subdir')
                self.assertEqual(test_art.artifact_content, open(
                    root_dir.path("subdir/file_two.txt"), "rb").read())

    def test_log_artifacts_no_artifact_path(self):
        with TempDir() as root_dir:
            with open(root_dir.path("file_one.txt"), "w") as f:
                f.write('DB store Test One')

            os.mkdir(root_dir.path("subdir"))
            with open(root_dir.path("subdir/file_two.txt"), "w") as f:
                f.write('DB store Test Two')

            self.store.log_artifacts(root_dir._path)

            with self.store.ManagedSessionMaker() as session:
                result = session.query(initial_artifact_store_models.SqlArtifact).all()
                self.assertEqual(len(result), 2)

                test_art = session.query(initial_artifact_store_models.SqlArtifact). \
                    filter_by(artifact_name='file_one.txt').first()
                self.assertEqual(str(test_art.artifact_id), "1")
                self.assertEqual(test_art.artifact_name, 'file_one.txt')
                self.assertEqual(test_art.group_path, '')
                self.assertEqual(test_art.artifact_content, open(
                    root_dir.path("file_one.txt"), "rb").read())

                test_art = session.query(initial_artifact_store_models.SqlArtifact). \
                    filter_by(artifact_name='file_two.txt').first()
                self.assertEqual(str(test_art.artifact_id), "2")
                self.assertEqual(test_art.artifact_name, 'file_two.txt')
                self.assertEqual(test_art.group_path, 'subdir')
                self.assertEqual(test_art.artifact_content, open(
                    root_dir.path("subdir/file_two.txt"), "rb").read())

    def test_list_artifacts(self):
        with TempDir() as root_dir:
            with open(root_dir.path("file_one.txt"), "w") as f:
                f.write('DB store Test One')

            os.mkdir(root_dir.path("subdir"))
            with open(root_dir.path("subdir/file_two.txt"), "w") as f:
                f.write('DB store Test Two')

            with open(root_dir.path("subdir/file_three.txt"), "w") as f:
                f.write('DB store Test Three')

            self.store.log_artifacts(root_dir._path, 'new_path/path')
            self.store.log_artifacts(root_dir._path, 'new_path2/path')

            self.assertEqual(len(self.store.list_artifacts('new_path/path')), 3)
            filenames = [f.path for f in self.store.list_artifacts('new_path/path')]
            self.assertTrue(filenames.__contains__('new_path/path/file_one.txt'))
            self.assertTrue(filenames.__contains__('new_path/path/subdir/file_two.txt'))
            self.assertTrue(filenames.__contains__('new_path/path/subdir/file_three.txt'))

            self.assertEqual(len(self.store.list_artifacts('new_path')), 3)
            filenames = [f.path for f in self.store.list_artifacts('new_path')]
            self.assertTrue(filenames.__contains__('new_path/path/file_one.txt'))
            self.assertTrue(filenames.__contains__('new_path/path/subdir/file_two.txt'))
            self.assertTrue(filenames.__contains__('new_path/path/subdir/file_three.txt'))

            self.assertEqual(len(self.store.list_artifacts('new_path2/path')), 3)
            filenames = [f.path for f in self.store.list_artifacts('new_path2/path')]
            self.assertTrue(filenames.__contains__('new_path2/path/file_one.txt'))
            self.assertTrue(filenames.__contains__('new_path2/path/subdir/file_two.txt'))
            self.assertTrue(filenames.__contains__('new_path2/path/subdir/file_three.txt'))

            self.assertEqual(len(self.store.list_artifacts('new_path2')), 3)
            filenames = [f.path for f in self.store.list_artifacts('new_path2')]
            self.assertTrue(filenames.__contains__('new_path2/path/file_one.txt'))
            self.assertTrue(filenames.__contains__('new_path2/path/subdir/file_two.txt'))
            self.assertTrue(filenames.__contains__('new_path2/path/subdir/file_three.txt'))

    def test_download_file_artifact(self):
        with TempDir() as root_dir:
            with open(root_dir.path("file_one.txt"), "w") as f:
                f.write('DB store Test One')

            self.store.log_artifacts(root_dir._path, 'new_path/path')
            local_path = self.store.download_artifacts(artifact_path='new_path/path/file_one.txt')
            assert open(local_path).read() == 'DB store Test One'

    def test_download_dir_artifact(self):
        with TempDir() as root_dir:
            with open(root_dir.path("file_one.txt"), "w") as f:
                f.write('DB store Test One')

            os.mkdir(root_dir.path("subdir"))
            with open(root_dir.path("subdir/file_two.txt"), "w") as f:
                f.write('DB store Test Two')
            self.store.log_artifacts(root_dir._path, 'new_path/path')
            local_path = self.store.download_artifacts(artifact_path='new_path/path')
            assert open(os.path.join(local_path, "file_one.txt")).read() == 'DB store Test One'
            assert open(os.path.join(local_path, "file_two.txt")).read() == 'DB store Test Two'

    def test_download_dir_artifact(self):
        with TempDir() as root_dir:
            with open(root_dir.path("file_one.txt"), "w") as f:
                f.write('DB store Test One')

            self.store.log_artifacts(root_dir._path, 'artifact')
            local_path = self.store.download_artifacts(artifact_path='artifact')
            assert open(
                os.path.join(local_path, "file_one.txt")).read() == 'DB store Test One'

import os
import tempfile

import pytest

from mlflow.exceptions import ExecutionException
from mlflow.projects.utils import (
    _get_storage_dir, _is_valid_branch_name, _is_zip_uri, fetch_project, _parse_subdirectory)
from tests.projects.utils import (
    assert_dirs_equal, GIT_PROJECT_URI, TEST_PROJECT_DIR, TEST_PROJECT_NAME)


def _build_uri(base_uri, subdirectory):
    if subdirectory != "":
        return "%s#%s" % (base_uri, subdirectory)
    return base_uri


@pytest.fixture
def zipped_repo(tmpdir):
    import zipfile
    zip_name = tmpdir.join('%s.zip' % TEST_PROJECT_NAME).strpath
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(TEST_PROJECT_DIR):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                zip_file.write(file_path, file_path[len(TEST_PROJECT_DIR) + len(os.sep):])
    return zip_name


def test_is_zip_uri():
    assert _is_zip_uri('http://foo.bar/moo.zip')
    assert _is_zip_uri('https://foo.bar/moo.zip')
    assert _is_zip_uri('file:///moo.zip')
    assert _is_zip_uri('file://C:/moo.zip')
    assert _is_zip_uri('/moo.zip')
    assert _is_zip_uri('C:/moo.zip')
    assert not _is_zip_uri('http://foo.bar/moo')
    assert not _is_zip_uri('https://foo.bar/moo')
    assert not _is_zip_uri('file:///moo')
    assert not _is_zip_uri('file://C:/moo')
    assert not _is_zip_uri('/moo')
    assert not _is_zip_uri('C:/moo')


def test_fetch_project(local_git_repo, local_git_repo_uri, zipped_repo, httpserver):
    httpserver.serve_content(open(zipped_repo, 'rb').read())
    # The tests are as follows:
    # 1. Fetching a locally saved project.
    # 2. Fetching a project located in a Git repo root directory.
    # 3. Fetching a project located in a Git repo subdirectory.
    # 4. Passing a subdirectory works for local directories.
    # 5. Fetching a remote ZIP file
    # 6. Using a local ZIP file
    # 7. Using a file:// URL to a local ZIP file
    test_list = [
        (TEST_PROJECT_DIR, '', TEST_PROJECT_DIR),
        (local_git_repo_uri, '', local_git_repo),
        (local_git_repo_uri, 'example_project', os.path.join(local_git_repo, 'example_project')),
        (os.path.dirname(TEST_PROJECT_DIR), os.path.basename(TEST_PROJECT_DIR), TEST_PROJECT_DIR),
        (httpserver.url + '/%s.zip' % TEST_PROJECT_NAME, '', TEST_PROJECT_DIR),
        (zipped_repo, '', TEST_PROJECT_DIR),
        ('file://%s' % zipped_repo, '', TEST_PROJECT_DIR),
    ]
    for base_uri, subdirectory, expected in test_list:
        work_dir = fetch_project(uri=_build_uri(base_uri, subdirectory), force_tempdir=False)
        assert_dirs_equal(expected=expected, actual=work_dir)
    # Test that we correctly determine the dest directory to use when fetching a project.
    for force_tempdir, uri in [(True, TEST_PROJECT_DIR), (False, GIT_PROJECT_URI)]:
        dest_dir = fetch_project(uri=uri, force_tempdir=force_tempdir)
        assert os.path.commonprefix([dest_dir, tempfile.gettempdir()]) == tempfile.gettempdir()
        assert os.path.exists(dest_dir)
    for force_tempdir, uri in [(None, TEST_PROJECT_DIR), (False, TEST_PROJECT_DIR)]:
        assert fetch_project(
            uri=uri, force_tempdir=force_tempdir) == os.path.abspath(TEST_PROJECT_DIR)


def test_fetch_project_validations(local_git_repo_uri):
    # Verify that runs fail if given incorrect subdirectories via the `#` character.
    for base_uri in [TEST_PROJECT_DIR, local_git_repo_uri]:
        with pytest.raises(ExecutionException):
            fetch_project(uri=_build_uri(base_uri, "fake"), force_tempdir=False)

    # Passing `version` raises an exception for local projects
    with pytest.raises(ExecutionException):
        fetch_project(uri=TEST_PROJECT_DIR, force_tempdir=False, version="version")


def test_dont_remove_mlruns(tmpdir):
    # Fetching a directory containing an "mlruns" folder doesn't remove the "mlruns" folder
    src_dir = tmpdir.mkdir("mlruns-src-dir")
    src_dir.mkdir("mlruns").join("some-file.txt").write("hi")
    src_dir.join("MLproject").write("dummy MLproject contents")
    dst_dir = fetch_project(uri=src_dir.strpath, version=None,
                            force_tempdir=False)
    assert_dirs_equal(expected=src_dir.strpath, actual=dst_dir)


def test_parse_subdirectory():
    # Make sure the parsing works as intended.
    test_uri = "uri#subdirectory"
    parsed_uri, parsed_subdirectory = _parse_subdirectory(test_uri)
    assert parsed_uri == "uri"
    assert parsed_subdirectory == "subdirectory"

    # Make sure periods are restricted in Git repo subdirectory paths.
    period_fail_uri = GIT_PROJECT_URI + "#.."
    with pytest.raises(ExecutionException):
        _parse_subdirectory(period_fail_uri)


def test_storage_dir(tmpdir):
    """
    Test that we correctly handle the `storage_dir` argument, which specifies where to download
    distributed artifacts passed to arguments of type `path`.
    """
    assert os.path.dirname(_get_storage_dir(tmpdir.strpath)) == tmpdir.strpath
    assert os.path.dirname(_get_storage_dir(None)) == tempfile.gettempdir()


def test_is_valid_branch_name(local_git_repo):
    assert _is_valid_branch_name(local_git_repo, "master")
    assert not _is_valid_branch_name(local_git_repo, "dev")

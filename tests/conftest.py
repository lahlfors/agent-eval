# agent-eval-framework/tests/conftest.py
import pytest
from unittest.mock import MagicMock
import sys

@pytest.fixture(autouse=True)
def mock_pyserini_for_tests(mocker):
    """
    Automatically mocks the pyserini library for all tests.

    This prevents issues with installing pyserini and its dependencies (like nmslib)
    in the test environment. Any code under test that imports 'pyserini'
    will get a MagicMock object instead.
    """
    # Create a mock object for pyserini
    mock_pyserini = MagicMock()
    mock_pyserini.search = MagicMock()
    mock_pyserini.search.lucene = MagicMock()
    mock_pyserini.search.lucene.LuceneSearcher = MagicMock()


    # Add the mock to sys.modules
    mocker.patch.dict(sys.modules, {
        "pyserini": mock_pyserini,
        "pyserini.search": mock_pyserini.search,
        "pyserini.search.lucene": mock_pyserini.search.lucene,
    })
    mocker.patch('google.auth.default', return_value=(None, 'test-project'))

    print("Mocked sys.modules['pyserini']")

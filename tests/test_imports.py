"""Smoke tests des dépendances principales du projet."""
import pytest


pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:builtin type SwigPyPacked has no __module__ attribute:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:builtin type SwigPyObject has no __module__ attribute:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:builtin type swigvarlink has no __module__ attribute:DeprecationWarning"
    ),
]

def test_imports():
    """Vérifie que les principales dépendances techniques sont importables."""
    import faiss
    import fastapi
    import langchain
    import mistralai
    import pandas
    import requests

    assert faiss is not None
    assert fastapi is not None
    assert langchain is not None
    assert mistralai is not None
    assert pandas is not None
    assert requests is not None

import numpy as np
import pytest
from src.knn import KNNClassifier

@pytest.fixture
def simple_data():
    X = np.array([[0.0], [1.0], [3.0], [4.0]])
    y = np.array([0,   0,   1,   1])
    return X, y

def test_euclidean(simple_data):
    X, _ = simple_data
    clf = KNNClassifier(k=1)
    clf.fit(X, [0,0,1,1])
    pred = clf.predict(np.array([[3.0]]))
    assert pred[0] == 1

def test_majority_vote(simple_data):
    X, y = simple_data
    clf = KNNClassifier(k=3)
    clf.fit(X, y)
    pred = clf.predict(np.array([[2.0]]))
    assert pred[0] == 0

def test_error_before_fit():
    clf = KNNClassifier(k=1)
    with pytest.raises(ValueError):
        clf.predict(np.array([[0.0]]))
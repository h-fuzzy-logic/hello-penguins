import numpy as np
import pytest
import sys
import os

# Add the parent directory to sys.path to allow imports from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from confusion_matrix_combined import (
    calculate_recall, 
    calculate_precision, 
    calculate_f1, 
    calculate_accuracy,
    calculate_class_metrics
)

@pytest.fixture
def simple_confusion_matrix():
    """Simple 3x3 confusion matrix for testing.
    
    The confusion matrix:
    [
        [5, 2, 0],  # Class 0
        [1, 8, 1],  # Class 1
        [0, 3, 10]  # Class 2
    ]
    """
    return np.array([
        [5, 2, 0], 
        [1, 8, 1], 
        [0, 3, 10]
    ])

@pytest.fixture
def perfect_confusion_matrix():
    """A perfect confusion matrix with all predictions correct."""
    return np.array([
        [10, 0, 0],
        [0, 15, 0],
        [0, 0, 20]
    ])

@pytest.fixture
def zero_confusion_matrix():
    """A confusion matrix with zeros in some positions."""
    return np.array([
        [0, 0, 0],
        [0, 5, 0],
        [0, 0, 0]
    ])

def test_calculate_recall(simple_confusion_matrix, perfect_confusion_matrix, zero_confusion_matrix):
    # Test with simple confusion matrix
    assert calculate_recall(simple_confusion_matrix, 0) == 5/7
    assert calculate_recall(simple_confusion_matrix, 1) == 8/10
    assert calculate_recall(simple_confusion_matrix, 2) == 10/13
    
    # Test with perfect confusion matrix
    assert calculate_recall(perfect_confusion_matrix, 0) == 1.0
    assert calculate_recall(perfect_confusion_matrix, 1) == 1.0
    assert calculate_recall(perfect_confusion_matrix, 2) == 1.0
    
    # Test with zeros
    assert calculate_recall(zero_confusion_matrix, 0) == 0.0

def test_calculate_precision(simple_confusion_matrix, perfect_confusion_matrix, zero_confusion_matrix):
    # Test with simple confusion matrix
    assert calculate_precision(simple_confusion_matrix, 0) == 5/6
    assert calculate_precision(simple_confusion_matrix, 1) == 8/13
    assert calculate_precision(simple_confusion_matrix, 2) == 10/11
    
    # Test with perfect confusion matrix
    assert calculate_precision(perfect_confusion_matrix, 0) == 1.0
    assert calculate_precision(perfect_confusion_matrix, 1) == 1.0
    assert calculate_precision(perfect_confusion_matrix, 2) == 1.0
    
    # Test with zeros
    assert calculate_precision(zero_confusion_matrix, 0) == 0.0

def test_calculate_f1():
    # Test with various precision and recall values
    assert calculate_f1(0.8, 0.6) == pytest.approx(0.6857, 0.001)
    assert calculate_f1(1.0, 1.0) == 1.0
    assert calculate_f1(0.0, 0.8) == 0.0
    assert calculate_f1(0.0, 0.0) == 0.0

def test_calculate_accuracy(simple_confusion_matrix, perfect_confusion_matrix, zero_confusion_matrix):
    # Test with simple confusion matrix
    assert calculate_accuracy(simple_confusion_matrix) == 23/30
    
    # Test with perfect confusion matrix
    assert calculate_accuracy(perfect_confusion_matrix) == 1.0
    
    # Test with zeros
    assert calculate_accuracy(zero_confusion_matrix) == 5/5

def test_calculate_class_metrics(simple_confusion_matrix):
    # Test class 0 metrics
    metrics_0 = calculate_class_metrics(simple_confusion_matrix, 0)
    assert metrics_0['TP'] == 5
    assert metrics_0['Support'] == 7
    assert metrics_0['PredictedTotal'] == 6
    assert metrics_0['Recall'] == 5/7
    assert metrics_0['Precision'] == 5/6
    assert metrics_0['F1'] == pytest.approx(2 * (5/6 * 5/7) / (5/6 + 5/7), 0.001)
    
    # Test class 1 metrics
    metrics_1 = calculate_class_metrics(simple_confusion_matrix, 1)
    assert metrics_1['TP'] == 8
    assert metrics_1['Support'] == 10
    assert metrics_1['PredictedTotal'] == 13
    assert metrics_1['Recall'] == 8/10
    assert metrics_1['Precision'] == 8/13
    assert metrics_1['F1'] == pytest.approx(2 * (8/13 * 8/10) / (8/13 + 8/10), 0.001)

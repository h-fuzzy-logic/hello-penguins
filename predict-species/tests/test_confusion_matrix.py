import numpy as np
import pytest
from confusion_matrix_combined import ClassificationMetrics

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

@pytest.fixture
def simple_metrics(simple_confusion_matrix):
    """Create a ClassificationMetrics instance with the simple confusion matrix."""
    return ClassificationMetrics(simple_confusion_matrix, ["Class0", "Class1", "Class2"])

@pytest.fixture
def perfect_metrics(perfect_confusion_matrix):
    """Create a ClassificationMetrics instance with the perfect confusion matrix."""
    return ClassificationMetrics(perfect_confusion_matrix, ["Class0", "Class1", "Class2"])

@pytest.fixture
def zero_metrics(zero_confusion_matrix):
    """Create a ClassificationMetrics instance with the zero confusion matrix."""
    return ClassificationMetrics(zero_confusion_matrix, ["Class0", "Class1", "Class2"])
        
def test_calculate_recall(simple_metrics, perfect_metrics, zero_metrics):
    # Test with simple confusion matrix
    assert simple_metrics.calculate_recall(0) == 5/7
    assert simple_metrics.calculate_recall(1) == 8/10
    assert simple_metrics.calculate_recall(2) == 10/13
    
    # Test with perfect confusion matrix
    assert perfect_metrics.calculate_recall(0) == 1.0
    assert perfect_metrics.calculate_recall(1) == 1.0
    assert perfect_metrics.calculate_recall(2) == 1.0
    
    # Test with zeros
    assert zero_metrics.calculate_recall(0) == 0.0

def test_calculate_precision(simple_metrics, perfect_metrics, zero_metrics):
    # Test with simple confusion matrix
    assert simple_metrics.calculate_precision(0) == 5/6
    assert simple_metrics.calculate_precision(1) == 8/13
    assert simple_metrics.calculate_precision(2) == 10/11
    
    # Test with perfect confusion matrix
    assert perfect_metrics.calculate_precision(0) == 1.0
    assert perfect_metrics.calculate_precision(1) == 1.0
    assert perfect_metrics.calculate_precision(2) == 1.0
    
    # Test with zeros
    assert zero_metrics.calculate_precision(0) == 0.0

def test_calculate_f1():
    # Test the static method
    assert ClassificationMetrics.calculate_f1(0.8, 0.6) == pytest.approx(0.6857, 0.001)
    assert ClassificationMetrics.calculate_f1(1.0, 1.0) == 1.0
    assert ClassificationMetrics.calculate_f1(0.0, 0.8) == 0.0
    assert ClassificationMetrics.calculate_f1(0.0, 0.0) == 0.0

def test_accuracy(simple_metrics, perfect_metrics, zero_metrics):
    # Test with simple confusion matrix
    assert simple_metrics._calculate_accuracy() == 23/30
    
    # Test with perfect confusion matrix
    assert perfect_metrics._calculate_accuracy() == 1.0
    
    # Test with zeros
    assert zero_metrics._calculate_accuracy() == 5/5

def test_class_metrics(simple_metrics):
    # Test metrics for class 0
    metrics_0 = simple_metrics.get_metrics_for_class(0)
    assert metrics_0['Species'] == "Class0"
    assert metrics_0['TP'] == 5
    assert metrics_0['Support'] == 7
    assert metrics_0['PredictedTotal'] == 6
    assert metrics_0['Recall'] == 5/7
    assert metrics_0['Precision'] == 5/6
    assert metrics_0['F1'] == pytest.approx(2 * (5/6 * 5/7) / (5/6 + 5/7), 0.001)
    
    # Test metrics for class 1
    metrics_1 = simple_metrics.get_metrics_for_class(1)
    assert metrics_1['Species'] == "Class1"
    assert metrics_1['TP'] == 8
    assert metrics_1['Support'] == 10
    assert metrics_1['PredictedTotal'] == 13
    assert metrics_1['Recall'] == 8/10
    assert metrics_1['Precision'] == 8/13
    assert metrics_1['F1'] == pytest.approx(2 * (8/13 * 8/10) / (8/13 + 8/10), 0.001)

def test_overall_metrics(simple_metrics, perfect_metrics):
    # Test overall metrics for simple matrix
    overall = simple_metrics.get_overall_metrics()
    assert overall['TotalSamples'] == 30
    assert overall['CorrectPredictions'] == 23
    assert overall['Accuracy'] == 23/30
    
    # Test overall metrics for perfect matrix
    overall_perfect = perfect_metrics.get_overall_metrics()
    assert overall_perfect['TotalSamples'] == 45
    assert overall_perfect['CorrectPredictions'] == 45
    assert overall_perfect['Accuracy'] == 1.0

def test_formatted_metrics(simple_metrics):
    # Test formatted metrics for class
    formatted = simple_metrics.get_formatted_metrics_for_class(0)
    assert formatted['Recall'] == "71.4%"
    assert formatted['Precision'] == "83.3%"
    
    # Test formatted overall metrics
    formatted_overall = simple_metrics.get_formatted_overall_metrics()
    assert formatted_overall['Accuracy'] == "76.7%"
    assert isinstance(formatted_overall['F1'], str)
    assert "%" in formatted_overall['F1']

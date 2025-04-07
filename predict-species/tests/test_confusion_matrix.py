import numpy as np
import pytest
from metrics import ClassificationMetrics  # Updated import path

@pytest.fixture
def simple_true_pred():
    """Create y_true and y_pred arrays that would generate the simple confusion matrix."""
    # The simple confusion matrix is:
    # [
    #     [5, 2, 0],  # Class 0
    #     [1, 8, 1],  # Class 1
    #     [0, 3, 10]  # Class 2
    # ]
    y_true = []
    y_pred = []
    
    # Class 0: 5 correct, 2 predicted as class 1, 0 predicted as class 2
    y_true.extend([0] * 7)
    y_pred.extend([0] * 5 + [1] * 2)
    
    # Class 1: 1 predicted as class 0, 8 correct, 1 predicted as class 2
    y_true.extend([1] * 10)
    y_pred.extend([0] * 1 + [1] * 8 + [2] * 1)
    
    # Class 2: 0 predicted as class 0, 3 predicted as class 1, 10 correct
    y_true.extend([2] * 13)
    y_pred.extend([1] * 3 + [2] * 10)
    
    return np.array(y_true), np.array(y_pred)

@pytest.fixture
def perfect_true_pred():
    """Create y_true and y_pred arrays that would generate a perfect confusion matrix."""
    # The perfect confusion matrix is:
    # [
    #     [10, 0, 0],
    #     [0, 15, 0],
    #     [0, 0, 20]
    # ]
    y_true = []
    y_pred = []
    
    # All predictions are correct
    y_true.extend([0] * 10 + [1] * 15 + [2] * 20)
    y_pred.extend([0] * 10 + [1] * 15 + [2] * 20)
    
    return np.array(y_true), np.array(y_pred)

@pytest.fixture
def zero_class_true_pred():
    """Create y_true and y_pred arrays that would generate a matrix with some zeros."""
    # The zero confusion matrix is:
    # [
    #     [0, 0, 0],
    #     [0, 5, 0],
    #     [0, 0, 0]
    # ]
    y_true = []
    y_pred = []
    
    # Only class 1 has predictions, all correct
    y_true.extend([1] * 5)
    y_pred.extend([1] * 5)
    
    return np.array(y_true), np.array(y_pred)

@pytest.fixture
def simple_metrics(simple_true_pred):
    """Create a ClassificationMetrics instance with data that generates the simple confusion matrix."""
    y_true, y_pred = simple_true_pred
    return ClassificationMetrics(y_true, y_pred, class_labels=["Class0", "Class1", "Class2"])

@pytest.fixture
def perfect_metrics(perfect_true_pred):
    """Create a ClassificationMetrics instance with data that generates the perfect confusion matrix."""
    y_true, y_pred = perfect_true_pred
    return ClassificationMetrics(y_true, y_pred, class_labels=["Class0", "Class1", "Class2"])

@pytest.fixture
def zero_metrics(zero_class_true_pred):
    """Create a ClassificationMetrics instance with data that generates the matrix with zeros."""
    y_true, y_pred = zero_class_true_pred
    return ClassificationMetrics(y_true, y_pred, class_labels=["Class0", "Class1", "Class2"])
        
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

# Fix the failing test_f1_score_calculation
def test_f1_score_calculation():
    """Test that F1 scores are calculated correctly from predictions."""
    # Create simple true and predicted labels
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 1, 0, 1, 1, 1, 2, 2])
    
    # Create metrics object
    metrics = ClassificationMetrics(y_true, y_pred)
    
    # Get the actual F1 scores from the metrics object
    class_0_metrics = metrics.get_metrics_for_class(0)
    class_1_metrics = metrics.get_metrics_for_class(1)
    class_2_metrics = metrics.get_metrics_for_class(2)
    
    # Recalculate using scikit-learn's calculation method
    # Class 0: precision=2/3, recall=2/3
    assert class_0_metrics['Precision'] == pytest.approx(2/3, abs=0.01)
    assert class_0_metrics['Recall'] == pytest.approx(2/3, abs=0.01)
    assert class_0_metrics['F1'] == pytest.approx(2*(2/3)*(2/3)/((2/3)+(2/3)), abs=0.01)
    
    # Class 1: precision=2/4, recall=2/3
    assert class_1_metrics['Precision'] == pytest.approx(2/4, abs=0.01)
    assert class_1_metrics['Recall'] == pytest.approx(2/3, abs=0.01)
    assert class_1_metrics['F1'] == pytest.approx(2*(2/4)*(2/3)/((2/4)+(2/3)), abs=0.01)
    
    # Class 2: precision=2/2, recall=2/3
    assert class_2_metrics['Precision'] == pytest.approx(2/2, abs=0.01)
    assert class_2_metrics['Recall'] == pytest.approx(2/3, abs=0.01)
    assert class_2_metrics['F1'] == pytest.approx(2*(2/2)*(2/3)/((2/2)+(2/3)), abs=0.01)
    
    # Test that overall F1 is correctly calculated as the macro-average
    overall = metrics.get_overall_metrics()
    expected_macro_f1 = np.mean([
        2*(2/3)*(2/3)/((2/3)+(2/3)),
        2*(2/4)*(2/3)/((2/4)+(2/3)),
        2*(2/2)*(2/3)/((2/2)+(2/3))
    ])
    assert overall['F1'] == pytest.approx(expected_macro_f1, abs=0.01)

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

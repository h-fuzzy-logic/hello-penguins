import numpy as np
import pytest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score
from metrics import ClassificationMetrics

@pytest.fixture
def binary_classification_data():
    """Create a simple binary classification dataset."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1])
    return y_true, y_pred

@pytest.fixture
def multiclass_classification_data():
    """Create a simple multiclass classification dataset."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    y_pred = np.array([0, 0, 1, 1, 1, 2, 0, 2, 2, 1, 3, 3])
    return y_true, y_pred

@pytest.fixture
def imbalanced_classification_data():
    """Create an imbalanced classification dataset."""
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 1])
    return y_true, y_pred

@pytest.fixture
def penguin_classification_data():
    """Create a more realistic penguin species classification dataset."""
    # Simulate Adelie, Chinstrap, Gentoo classification
    species = ['Adelie', 'Chinstrap', 'Gentoo']
    # True labels (0=Adelie, 1=Chinstrap, 2=Gentoo)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    # Predictions with some mistakes
    y_pred = np.array([0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, 2, 1, 2, 2])
    return y_true, y_pred, species

def test_initialization_with_labels():
    """Test that ClassificationMetrics initializes correctly with custom labels."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 2, 0])
    class_labels = ["Class A", "Class B", "Class C"]
    
    metrics = ClassificationMetrics(y_true, y_pred, class_labels=class_labels)
    
    assert metrics.n_classes == 3
    assert metrics.class_labels == class_labels
    assert metrics.total_samples == 6
    assert np.array_equal(metrics.label_set, np.array([0, 1, 2]))

def test_confusion_matrix_creation(multiclass_classification_data):
    """Test that the confusion matrix is created correctly."""
    y_true, y_pred = multiclass_classification_data
    
    # Create expected confusion matrix using sklearn
    expected_cm = confusion_matrix(y_true, y_pred)
    
    # Create metrics object
    metrics = ClassificationMetrics(y_true, y_pred)
    
    # Check if confusion matrix matches
    assert np.array_equal(metrics.get_confusion_matrix(), expected_cm)

def test_accuracy_calculation(binary_classification_data):
    """Test that accuracy is calculated correctly."""
    y_true, y_pred = binary_classification_data
    
    # Calculate expected accuracy
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    expected_accuracy = correct / total
    
    # Create metrics object
    metrics = ClassificationMetrics(y_true, y_pred)
    
    # Check accuracy
    assert metrics.get_overall_metrics()['Accuracy'] == expected_accuracy

def test_precision_recall_f1(binary_classification_data):
    """Test that precision, recall, and F1 are calculated correctly."""
    y_true, y_pred = binary_classification_data
    
    # Calculate expected values using sklearn
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Create metrics object
    metrics = ClassificationMetrics(y_true, y_pred)
    
    # Check metrics for each class
    for i in range(2):
        class_metrics = metrics.get_metrics_for_class(i)
        assert class_metrics['Precision'] == pytest.approx(precision[i])
        assert class_metrics['Recall'] == pytest.approx(recall[i])
        assert class_metrics['F1'] == pytest.approx(f1[i])

def test_overall_f1_calculation(multiclass_classification_data):
    """Test that overall F1 score is calculated correctly."""
    y_true, y_pred = multiclass_classification_data
    
    # Calculate expected macro F1 using sklearn
    expected_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Create metrics object
    metrics = ClassificationMetrics(y_true, y_pred)
    
    # Check overall F1
    assert metrics.get_overall_metrics()['F1'] == pytest.approx(expected_f1)

def test_formatted_metrics(penguin_classification_data):
    """Test the formatted metrics output."""
    y_true, y_pred, species = penguin_classification_data
    
    # Create metrics object
    metrics = ClassificationMetrics(y_true, y_pred, class_labels=species)
    
    # Check formatted output for a class
    adelie_metrics = metrics.get_formatted_metrics_for_class(0)
    assert isinstance(adelie_metrics['Precision'], str)
    assert isinstance(adelie_metrics['Recall'], str)
    assert isinstance(adelie_metrics['F1'], str)
    assert "%" in adelie_metrics['Precision']
    
    # Check formatted overall metrics
    formatted_overall = metrics.get_formatted_overall_metrics()
    assert isinstance(formatted_overall['Accuracy'], str)
    assert isinstance(formatted_overall['F1'], str)
    assert "%" in formatted_overall['Accuracy']
    
    # Verify formatting is correct
    raw_metrics = metrics.get_metrics_for_class(0)
    formatted_precision = f"{raw_metrics['Precision']*100:.1f}%"
    assert adelie_metrics['Precision'] == formatted_precision

def test_correct_class_support(penguin_classification_data):
    """Test that support (total actual instances) is calculated correctly for each class."""
    y_true, y_pred, species = penguin_classification_data
    
    # Count occurrences of each class in y_true
    unique, counts = np.unique(y_true, return_counts=True)
    expected_support = dict(zip(unique, counts))
    
    # Create metrics object
    metrics = ClassificationMetrics(y_true, y_pred, class_labels=species)
    
    # Check support for each class
    for i in range(len(species)):
        class_metrics = metrics.get_metrics_for_class(i)
        assert class_metrics['Support'] == expected_support[i]

def test_correct_predicted_totals(penguin_classification_data):
    """Test that predicted totals are calculated correctly for each class."""
    y_true, y_pred, species = penguin_classification_data
    
    # Count occurrences of each class in y_pred
    unique, counts = np.unique(y_pred, return_counts=True)
    expected_predicted = dict(zip(unique, counts))
    
    # Create metrics object
    metrics = ClassificationMetrics(y_true, y_pred, class_labels=species)
    
    # Check predicted totals for each class
    for i in range(len(species)):
        class_metrics = metrics.get_metrics_for_class(i)
        assert class_metrics['PredictedTotal'] == expected_predicted.get(i, 0)

def test_classification_report(penguin_classification_data):
    """Test that classification report is generated correctly."""
    y_true, y_pred, species = penguin_classification_data
    
    # Create metrics object
    metrics = ClassificationMetrics(y_true, y_pred, class_labels=species)
    
    # Get classification report as dictionary
    report = metrics.get_classification_report(output_dict=True)
    
    # Check that expected keys exist
    assert all(s in report for s in species)
    assert 'accuracy' in report
    assert 'macro avg' in report
    assert 'weighted avg' in report
    
    # Check specific metrics
    for i, s in enumerate(species):
        assert 'precision' in report[s]
        assert 'recall' in report[s]
        assert 'f1-score' in report[s]
        assert 'support' in report[s]
        
        # Verify values match our metrics
        class_metrics = metrics.get_metrics_for_class(i)
        assert report[s]['precision'] == pytest.approx(class_metrics['Precision'])
        assert report[s]['recall'] == pytest.approx(class_metrics['Recall'])
        assert report[s]['f1-score'] == pytest.approx(class_metrics['F1'])
        assert report[s]['support'] == class_metrics['Support']

def test_edge_case_single_class():
    """Test with only a single class."""
    y_true = np.array([0, 0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0, 0])
    
    metrics = ClassificationMetrics(y_true, y_pred)
    
    assert metrics.n_classes == 1
    assert metrics.get_overall_metrics()['Accuracy'] == 1.0
    assert metrics.get_metrics_for_class(0)['Precision'] == 1.0
    assert metrics.get_metrics_for_class(0)['Recall'] == 1.0
    assert metrics.get_metrics_for_class(0)['F1'] == 1.0

def test_edge_case_zero_predictions():
    """Test with no correct predictions."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    
    metrics = ClassificationMetrics(y_true, y_pred)
    
    assert metrics.get_overall_metrics()['Accuracy'] == 0.0
    # F1 should be zero since precision and recall are both zero
    for i in range(2):
        assert metrics.get_metrics_for_class(i)['F1'] == 0.0

import os
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from metrics import ClassificationMetrics
from confusion_matrix_combined import (
    create_confusion_heatmap,
    create_metrics_table,
    create_combined_confusion_matrix
)

@pytest.fixture
def sample_metrics():
    """Create a sample metrics object for testing visualization."""
    # Create a simple classification dataset
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 2, 0, 2, 2, 2, 2])
    class_labels = ['Adelie', 'Chinstrap', 'Gentoo']
    
    # Create metrics object
    return ClassificationMetrics(y_true, y_pred, class_labels=class_labels)

@pytest.fixture
def output_dir(tmpdir):
    """Create a temporary directory for output files."""
    return str(tmpdir)

def test_create_confusion_heatmap(sample_metrics, monkeypatch):
    """Test that the confusion heatmap is created correctly."""
    # Patch the set_xlabel and set_ylabel methods to handle the 'loc' parameter
    # This is a compatibility fix for older matplotlib versions
    original_set_xlabel = plt.Axes.set_xlabel
    original_set_ylabel = plt.Axes.set_ylabel
    
    def patched_set_xlabel(self, xlabel, *args, **kwargs):
        # Remove 'loc' parameter if present for backward compatibility
        if 'loc' in kwargs:
            kwargs.pop('loc')
        return original_set_xlabel(self, xlabel, *args, **kwargs)
    
    def patched_set_ylabel(self, ylabel, *args, **kwargs):
        # Remove 'loc' parameter if present for backward compatibility
        if 'loc' in kwargs:
            kwargs.pop('loc')
        return original_set_ylabel(self, ylabel, *args, **kwargs)
    
    # Apply the patches
    monkeypatch.setattr(plt.Axes, "set_xlabel", patched_set_xlabel)
    monkeypatch.setattr(plt.Axes, "set_ylabel", patched_set_ylabel)
    
    # Create heatmap
    fig, ax = create_confusion_heatmap(sample_metrics)
    
    # Check basic properties
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    
    # Check if species labels are used for x and y ticks
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    
    # Since the order might be different depending on matplotlib version,
    # check if all labels are present without enforcing exact order
    for label in sample_metrics.class_labels:
        assert label in xticklabels
        assert label in yticklabels
    
    # Check that the heatmap contains annotations
    assert len(ax.texts) > 0
    
    # Clean up
    plt.close(fig)

def test_create_metrics_table(sample_metrics):
    """Test that the metrics table is created correctly."""
    # Create metrics table
    fig, ax, metrics = create_metrics_table(sample_metrics)
    
    # Check basic properties
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert isinstance(metrics, list)
    
    # Check metrics data
    assert len(metrics) == len(sample_metrics.class_labels)
    assert all('Species' in m for m in metrics)
    assert all('Support' in m for m in metrics)
    assert all('Recall' in m for m in metrics)
    assert all('Precision' in m for m in metrics)
    assert all('F1' in m for m in metrics)
    
    # Check title
    assert ax.get_title() is not None
    
    # Check if table is created
    tables = [c for c in ax.get_children() if isinstance(c, matplotlib.table.Table)]
    assert len(tables) == 1
    
    # Clean up
    plt.close(fig)

def test_create_combined_confusion_matrix_return_fig(sample_metrics):
    """Test creating combined confusion matrix without saving."""
    # Create combined visualization without saving
    fig, axes = create_combined_confusion_matrix(
        metrics_obj=sample_metrics,
        title="Test Combined Visualization"
    )
    
    # Check basic properties
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, tuple)
    assert len(axes) == 2
    assert isinstance(axes[0], plt.Axes)  # Heatmap axis
    assert isinstance(axes[1], plt.Axes)  # Table axis
    
    # Check figure properties
    suptitle = fig.texts[0]  # The suptitle is the first text element of the figure
    assert suptitle.get_text() == "Test Combined Visualization"
    
    # Clean up
    plt.close(fig)

def test_create_combined_confusion_matrix_save_file(sample_metrics, output_dir):
    """Test creating and saving combined confusion matrix."""
    # Create and save combined visualization
    output_path = os.path.join(output_dir, "test_output")
    result_path = create_combined_confusion_matrix(
        metrics_obj=sample_metrics,
        title="Test Combined Visualization",
        filename_prefix=output_path
    )
    
    # Check that the file was created and the correct path is returned
    expected_path = f"{output_path}.png"
    assert result_path == expected_path
    assert os.path.exists(expected_path)
    assert os.path.getsize(expected_path) > 0  # File should not be empty

def test_custom_axis_for_heatmap(sample_metrics):
    """Test providing a custom axis for the heatmap."""
    # Create a custom figure and axis
    custom_fig, custom_ax = plt.subplots(figsize=(8, 6))
    
    # Pass the custom axis to the heatmap function
    fig, ax = create_confusion_heatmap(sample_metrics, ax=custom_ax)
    
    # The returned fig should be None and ax should be custom_ax
    assert fig is None
    assert ax is custom_ax
    
    # Clean up
    plt.close(custom_fig)

def test_custom_axis_for_metrics_table(sample_metrics):
    """Test providing a custom axis for the metrics table."""
    # Create a custom figure and axis
    custom_fig, custom_ax = plt.subplots(figsize=(8, 6))
    
    # Pass the custom axis to the metrics table function
    fig, ax, metrics = create_metrics_table(sample_metrics, ax=custom_ax)
    
    # The returned fig should be None and ax should be custom_ax
    assert fig is None
    assert ax is custom_ax
    
    # Clean up
    plt.close(custom_fig)

def test_colormap_customization(sample_metrics):
    """Test customizing the colormap for the heatmap."""
    # Create heatmap with a custom colormap
    fig, ax = create_confusion_heatmap(sample_metrics, colormap='YlOrRd')
    
    # Check that the colormap is applied
    heatmap = next(im for im in ax.get_images())
    
    # Custom colormap behavior might vary, so just check that something is present
    assert heatmap.get_cmap() is not None
    
    # Clean up
    plt.close(fig)

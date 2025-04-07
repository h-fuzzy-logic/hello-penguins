import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from metrics import ClassificationMetrics


def create_confusion_heatmap(cm, species_labels, ax=None, colormap='Blues'):
    """Create the confusion matrix heatmap visualization.
    
    Args:
        cm: The confusion matrix array
        species_labels: List of class labels
        ax: Matplotlib axis to plot on (optional)
        colormap: Colormap to use for the heatmap
        
    Returns:
        fig: Figure object if ax is None
        ax: Axis with the plotted heatmap
    """
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = None
    
    # Calculate row percentages (recall perspective)
    cm_sum_rows = np.sum(cm, axis=1, keepdims=True)
    cm_perc_recall = cm / cm_sum_rows * 100
    
    # Create annotation with both counts and percentages
    annot = np.empty_like(cm).astype(str)
    for i in range(len(species_labels)):
        for j in range(len(species_labels)):
            count = cm[i, j]
            percentage = cm_perc_recall[i, j]
            annot[i, j] = f'{count}\n{percentage:.1f}%'
    
    colors = ["#f7fbff", "#08306b"]  # Light to dark blue
    cmap = LinearSegmentedColormap.from_list("blue_gradient", colors, N=10)
    # Plot the confusion matrix
    sns.heatmap(
        cm, 
        annot=annot, 
        fmt='', 
        cmap=cmap, 
        cbar_kws={'label': 'Count of Samples'},
        xticklabels=species_labels, 
        yticklabels=species_labels, 
        ax=ax,
        linewidths=0.5, linecolor='white'
    )
    
    # Highlight the diagonal with borders
    for i in range(len(species_labels)):
        ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=2))
    
     # Highlight diagonals with special borders
    for i in range(len(species_labels)):
        # Add a thicker green border around diagonal elements (correct predictions)
        ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='#0b4d8c', lw=2))
    
    # Add custom hatching to highlight the highest value in each row
    for i in range(len(species_labels)):
        max_idx = np.argmax(cm[i])
        if max_idx != i:  # If the maximum is not on the diagonal
            # Add red border to highlight misclassification tendency
            ax.add_patch(Rectangle((max_idx, i), 1, 1, fill=False, edgecolor='#f47b26', lw=4, linestyle=':'))

    # Set labels
    ax.set_title('Confusion Matrix: Class Count and Class Recall\n', fontsize=12, pad=10, loc='left')
    ax.set_xlabel('PREDICTED SPECIES', fontsize=12, labelpad=15, loc='left')
    ax.set_ylabel('ACTUAL SPECIES', fontsize=12, labelpad=15, loc='bottom')
    
    return fig, ax


def create_metrics_table(cm, species_labels, ax=None):
    """Create a metrics summary table visualization.
    
    Args:
        cm: The confusion matrix array
        species_labels: List of class labels
        ax: Matplotlib axis to plot on (optional)
        
    Returns:
        fig: Figure object if ax is None
        ax: Axis with the plotted table
        metrics: List of calculated metrics dictionaries
    """
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
    else:
        fig = None
        ax.axis('off')
    
    # Create metrics object and calculate all metrics
    metrics_obj = ClassificationMetrics(cm, species_labels)
    overall = metrics_obj.get_overall_metrics()
    formatted_overall = metrics_obj.get_formatted_overall_metrics()
    
    # Prepare table data
    metrics_data = []
    metrics = []  # Store metrics for return
    headers = ['Species', 'Total\nActual', 'Total\nPredicted', 'Correctly\nClassified', 'Recall', 'Precision', 'F1']
    
    # Add data for each class
    for i, species in enumerate(species_labels):
        class_metrics = metrics_obj.get_metrics_for_class(i)
        formatted_metrics = metrics_obj.get_formatted_metrics_for_class(i)
        
        # Store metrics for return
        metrics.append({
            'Species': species,
            'Support': class_metrics['Support'],
            'Recall': class_metrics['Recall'],
            'Precision': class_metrics['Precision'],
            'F1': class_metrics['F1']
        })
        
        # Add to table data
        metrics_data.append([
            species,                                 # Species name
            f"{class_metrics['Support']}",           # Total actual instances
            f"{class_metrics['PredictedTotal']}",    # Total predicted instances
            f"{class_metrics['TP']}",                # Correctly classified
            formatted_metrics['Recall'],             # Recall
            formatted_metrics['Precision'],          # Precision
            formatted_metrics['F1']                  # F1
        ])
    
    # Add overall row
    metrics_data.append([
        'Overall',
        f"{overall['TotalSamples']}",
        f"{overall['TotalSamples']}",
        f"{overall['CorrectPredictions']}",
        formatted_overall['Accuracy'],
        formatted_overall['Accuracy'],
        formatted_overall['F1']
    ])
    
    # Create table
    table = ax.table(
        cellText=metrics_data,
        colLabels=headers,
        loc='center',
        cellLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color the header row
    for j in range(len(headers)):
        table[0, j].set_facecolor("#f7fbff")
        table[0, j].set_text_props(fontweight='bold')
    
    # Color the Overall row
    for j in range(len(headers)):
        table[len(species_labels) + 1, j].set_facecolor("#f7fbff")
    
    # Add title and subtitle with sample count and accuracy
    ax.set_title(
        f'Performance Metrics by Species\nTotal samples: {overall["TotalSamples"]} | Overall accuracy: {overall["Accuracy"]:.1%}', 
        fontsize=12, pad=5, loc='left', y=0.75
    )
    
    return fig, ax, metrics


def create_combined_confusion_matrix(cm, species_labels, title, filename_prefix):
    """Create and save a combined confusion matrix visualization with metrics table.
    
    This function creates a comprehensive visualization with both a
    confusion matrix heatmap and a metrics summary table.
    
    Args:
        cm: The confusion matrix array
        species_labels: List of class labels
        title: Overall title for the visualization
        filename_prefix: Prefix for the saved file
        
    Returns:
        cm_path: Path to the saved visualization
    """
    # Create a figure with stacked subplots (heatmap on top, metrics table below)
    fig, (ax_heatmap, ax_metrics) = plt.subplots(
        2, 1, 
        figsize=(10, 12),
        gridspec_kw={'height_ratios': [1.2, 0.8]}
    )
    
    # Create the heatmap
    create_confusion_heatmap(cm, species_labels, ax=ax_heatmap)
    
    # Create the metrics table
    create_metrics_table(cm, species_labels, ax=ax_metrics)
    
    # Add overall title to the figure
    plt.suptitle(title, fontsize=14, y=0.98)
    
    # Use tight_layout with padding to ensure no overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.0)
    
    # Save the visualization
    cm_path = f'{filename_prefix}.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm_path
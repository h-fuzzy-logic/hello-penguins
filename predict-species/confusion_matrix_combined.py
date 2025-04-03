import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap


class ClassificationMetrics:
    """Class to calculate and store classification metrics based on a confusion matrix."""
    
    def __init__(self, confusion_matrix, class_labels=None):
        """Initialize with a confusion matrix.
        
        Args:
            confusion_matrix: The confusion matrix as a numpy array
            class_labels: Optional list of class labels
        """
        self.cm = confusion_matrix
        self.n_classes = confusion_matrix.shape[0]
        self.class_labels = class_labels if class_labels is not None else [f"Class {i}" for i in range(self.n_classes)]
        
        # Calculate and store all metrics
        self.metrics_by_class = self._calculate_all_metrics()
        self.accuracy = self._calculate_accuracy()
        self.overall_f1 = np.mean([m['F1'] for m in self.metrics_by_class])
        self.total_samples = np.sum(self.cm)
        self.correct_predictions = np.trace(self.cm)
    
    def _calculate_all_metrics(self):
        """Calculate metrics for all classes.
        
        Returns:
            List of dictionaries containing metrics for each class
        """
        metrics = []
        for i in range(self.n_classes):
            metrics.append(self._calculate_class_metrics(i))
        return metrics
    
    def _calculate_class_metrics(self, class_idx):
        """Calculate metrics for a specific class.
        
        Args:
            class_idx: Index of the class
            
        Returns:
            dict: Dictionary containing all metrics for the class
        """
        tp = self.cm[class_idx, class_idx]
        actual_count = np.sum(self.cm[class_idx])
        predicted_count = np.sum(self.cm[:, class_idx])
        
        recall = self.calculate_recall(class_idx)
        precision = self.calculate_precision(class_idx)
        f1 = self.calculate_f1(precision, recall)
        
        return {
            'Species': self.class_labels[class_idx],
            'TP': int(tp),
            'Support': int(actual_count),
            'PredictedTotal': int(predicted_count),
            'Recall': recall,
            'Precision': precision,
            'F1': f1
        }
    
    def calculate_recall(self, class_idx):
        """Calculate recall for a specific class.
        
        Args:
            class_idx: Index of the class
            
        Returns:
            float: Recall value (0-1)
        """
        row_sum = np.sum(self.cm[class_idx])
        if row_sum > 0:
            return self.cm[class_idx, class_idx] / row_sum
        return 0
    
    def calculate_precision(self, class_idx):
        """Calculate precision for a specific class.
        
        Args:
            class_idx: Index of the class
            
        Returns:
            float: Precision value (0-1)
        """
        col_sum = np.sum(self.cm[:, class_idx])
        if col_sum > 0:
            return self.cm[class_idx, class_idx] / col_sum
        return 0
    
    @staticmethod
    def calculate_f1(precision, recall):
        """Calculate F1 score given precision and recall.
        
        Args:
            precision: Precision value (0-1)
            recall: Recall value (0-1)
            
        Returns:
            float: F1 score (0-1)
        """
        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0
    
    def _calculate_accuracy(self):
        """Calculate overall accuracy from confusion matrix.
        
        Returns:
            float: Accuracy value (0-1)
        """
        return np.trace(self.cm) / np.sum(self.cm) if np.sum(self.cm) > 0 else 0
    
    def get_metrics_for_class(self, class_idx):
        """Get metrics for a specific class.
        
        Args:
            class_idx: Index of the class
            
        Returns:
            dict: Dictionary containing metrics for the class
        """
        return self.metrics_by_class[class_idx]
    
    def get_formatted_metrics_for_class(self, class_idx):
        """Get metrics with formatted percentage values for a specific class.
        
        Args:
            class_idx: Index of the class
            
        Returns:
            dict: Dictionary containing formatted metrics for the class
        """
        metrics = self.metrics_by_class[class_idx].copy()
        metrics['Recall'] = f"{metrics['Recall']*100:.1f}%"
        metrics['Precision'] = f"{metrics['Precision']*100:.1f}%"
        metrics['F1'] = f"{metrics['F1']*100:.1f}%"
        return metrics
    
    def get_overall_metrics(self):
        """Get overall metrics for the confusion matrix.
        
        Returns:
            dict: Dictionary containing overall metrics
        """
        return {
            'Accuracy': self.accuracy,
            'F1': self.overall_f1,
            'TotalSamples': self.total_samples,
            'CorrectPredictions': self.correct_predictions
        }
    
    def get_formatted_overall_metrics(self):
        """Get overall metrics with formatted percentage values.
        
        Returns:
            dict: Dictionary containing formatted overall metrics
        """
        metrics = self.get_overall_metrics()
        return {
            'Accuracy': f"{metrics['Accuracy']*100:.1f}%",
            'F1': f"{metrics['F1']*100:.1f}%",
            'TotalSamples': metrics['TotalSamples'],
            'CorrectPredictions': metrics['CorrectPredictions']
        }


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
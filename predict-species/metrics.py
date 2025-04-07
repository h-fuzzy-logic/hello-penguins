import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix as sk_confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

class ClassificationMetrics:
    """Class to calculate and store classification metrics based on true and predicted labels."""
    
    def __init__(self, y_true, y_pred, labels=None, class_labels=None):
        """Initialize with prediction data.
        
        Args:
            y_true: Array of true class labels
            y_pred: Array of predicted class labels
            labels: List of possible class labels (optional)
            class_labels: Optional list of descriptive class labels for display
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        
        # Determine labels from the data if not provided
        if labels is not None:
            self.label_set = np.array(labels)
        else:
            self.label_set = np.unique(np.concatenate([self.y_true, self.y_pred]))
        
        # Create the confusion matrix
        self.cm = sk_confusion_matrix(self.y_true, self.y_pred, labels=self.label_set)
        self.n_classes = len(self.label_set)
        
        # Set display labels
        if class_labels is not None:
            self.class_labels = class_labels
        else:
            self.class_labels = [str(label) for label in self.label_set]
            
        # Calculate and store all metrics
        self.metrics_by_class = self._calculate_all_metrics()
        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        self.overall_f1 = f1_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        self.total_samples = len(self.y_true)
        self.correct_predictions = np.sum(self.y_true == self.y_pred)
    
    def _calculate_all_metrics(self):
        """Calculate metrics for all classes using scikit-learn.
        
        Returns:
            List of dictionaries containing metrics for each class
        """
        # Calculate precision, recall, f1, and support for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, 
            labels=self.label_set, 
            average=None,
            zero_division=0
        )
        
        metrics = []
        for i, label in enumerate(self.label_set):
            # Get the index in the confusion matrix corresponding to this label
            cm_idx = i  # Since we ordered the confusion matrix using labels
            
            # Calculate true positives for each class
            tp = self.cm[cm_idx, cm_idx]
            predicted_total = np.sum(self.cm[:, cm_idx])
            
            metrics.append({
                'Species': self.class_labels[i],
                'Label': label,
                'TP': int(tp),
                'Support': int(support[i]),
                'PredictedTotal': int(predicted_total),
                'Recall': recall[i],
                'Precision': precision[i],
                'F1': f1[i]
            })
        
        return metrics
    
    def get_metrics_for_class(self, class_idx):
        """Get metrics for a specific class.
        
        Args:
            class_idx: Index of the class in the label_set
            
        Returns:
            dict: Dictionary containing metrics for the class
        """
        return self.metrics_by_class[class_idx]
    
    def get_formatted_metrics_for_class(self, class_idx):
        """Get metrics with formatted percentage values for a specific class.
        
        Args:
            class_idx: Index of the class in the label_set
            
        Returns:
            dict: Dictionary containing formatted metrics for the class
        """
        metrics = self.metrics_by_class[class_idx].copy()
        metrics['Recall'] = f"{metrics['Recall']*100:.1f}%"
        metrics['Precision'] = f"{metrics['Precision']*100:.1f}%"
        metrics['F1'] = f"{metrics['F1']*100:.1f}%"
        return metrics
    
    def get_overall_metrics(self):
        """Get overall metrics for the predictions.
        
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
    
    def get_classification_report(self, output_dict=False):
        """Get a classification report from scikit-learn.
        
        Args:
            output_dict: Whether to return the report as a dict (True) or string (False)
            
        Returns:
            dict or str: Classification report
        """
        return classification_report(
            self.y_true, 
            self.y_pred, 
            labels=self.label_set,
            target_names=self.class_labels,
            output_dict=output_dict,
            zero_division=0
        )
    
    def get_confusion_matrix(self):
        """Get the confusion matrix.
        
        Returns:
            numpy.ndarray: The confusion matrix
        """
        return self.cm

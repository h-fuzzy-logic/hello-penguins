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
        
        # Create the true and predicted labels arrays from the confusion matrix
        self.y_true, self.y_pred = self._confusion_matrix_to_labels()
        
        # Calculate and store all metrics
        self.metrics_by_class = self._calculate_all_metrics()
        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        self.overall_f1 = f1_score(self.y_true, self.y_pred, average='macro')
        self.total_samples = len(self.y_true)
        self.correct_predictions = np.sum(self.y_true == self.y_pred)
    
    def _confusion_matrix_to_labels(self):
        """Convert confusion matrix to true and predicted label arrays.
        
        Returns:
            tuple: (y_true, y_pred) arrays
        """
        y_true = []
        y_pred = []
        
        # Iterate through the confusion matrix and create label arrays
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                count = int(self.cm[i, j])
                y_true.extend([i] * count)
                y_pred.extend([j] * count)
        
        return np.array(y_true), np.array(y_pred)
    
    def _calculate_all_metrics(self):
        """Calculate metrics for all classes using scikit-learn.
        
        Returns:
            List of dictionaries containing metrics for each class
        """
        # Calculate precision, recall, f1, and support for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, labels=range(self.n_classes), average=None
        )
        
        metrics = []
        for i in range(self.n_classes):
            # Calculate true positives for each class
            tp = self.cm[i, i]
            predicted_total = np.sum(self.cm[:, i])
            
            metrics.append({
                'Species': self.class_labels[i],
                'TP': int(tp),
                'Support': int(support[i]),
                'PredictedTotal': int(predicted_total),
                'Recall': recall[i],
                'Precision': precision[i],
                'F1': f1[i]
            })
        
        return metrics
    
    def calculate_recall(self, class_idx):
        """Calculate recall for a specific class using scikit-learn.
        
        Args:
            class_idx: Index of the class
            
        Returns:
            float: Recall value (0-1)
        """
        class_mask = (self.y_true == class_idx)
        if np.sum(class_mask) == 0:
            return 0
        return recall_score(
            self.y_true[class_mask], 
            self.y_pred[class_mask], 
            labels=[class_idx], 
            average='micro',
            zero_division=0
        )
    
    def calculate_precision(self, class_idx):
        """Calculate precision for a specific class using scikit-learn.
        
        Args:
            class_idx: Index of the class
            
        Returns:
            float: Precision value (0-1)
        """
        pred_mask = (self.y_pred == class_idx)
        if np.sum(pred_mask) == 0:
            return 0
        return precision_score(
            self.y_true[pred_mask], 
            self.y_pred[pred_mask], 
            labels=[class_idx], 
            average='micro',
            zero_division=0
        )
    
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
            labels=range(self.n_classes),
            target_names=self.class_labels,
            output_dict=output_dict
        )

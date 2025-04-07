import logging
import os
from pathlib import Path
#from confusion_matrix import create_combined_visualization
from metrics import ClassificationMetrics
from confusion_matrix_combined import create_combined_confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle




from common import (
    PYTHON,
    DatasetMixin,
    build_features_transformer,
    build_model,
    build_target_transformer,
    configure_logging,
    packages,
)
from metaflow import (
    FlowSpec,
    Parameter,
    card,
    conda_base,
    current,
    environment,
    project,
    resources,
    step,
)

configure_logging()

class AdelieClassifier:
    """A simple classifier that always predicts Adelie penguins."""
    
    def __init__(self):
        self.classes_ = ['Adelie', 'Chinstrap', 'Gentoo']
        
    def fit(self, X, y=None):
        """No-op fit method to match sklearn interface."""
        return self
        
    def predict(self, X):
        """Always predict Adelie."""
        return np.array(['Adelie'] * len(X))
    
    def predict_proba(self, X):
        """Return probability 1.0 for Adelie, 0 for others."""
        probas = np.zeros((len(X), 3))
        probas[:, 0] = 1.0  # Adelie is first class
        return probas

    def save(self, path):
        """Save model to disk."""
        import joblib
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path):
        """Load model from disk."""
        import joblib
        return joblib.load(path)

@project(name="penguins")
@conda_base(
    python=PYTHON,
    packages=packages(
        "scikit-learn",
        "pandas",
        "numpy",
        "keras",
        "jax[cpu]",
        "boto3",
        "mlflow",
        "seaborn"
    ),
)
class BaselineModel(FlowSpec, DatasetMixin):
    """Training pipeline.

    This pipeline trains, evaluates, and registers a model to predict the species of
    penguins.
    """

    mlflow_tracking_uri = Parameter(
        "mlflow-tracking-uri",
        help="Location of the MLflow tracking server.",
        default=os.getenv("MLFLOW_TRACKING_URI", "https://127.0.0.1:5000"),
    )

    accuracy_threshold = Parameter(
        "accuracy-threshold",
        help="Minimum accuracy threshold required to register the model.",
        default=0.7,
    )

    @card
    @step
    def start(self):
        """Start and prepare the Training pipeline."""
        import mlflow

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        logging.info("MLflow tracking server: %s", self.mlflow_tracking_uri)

        self.mode = "production" if current.is_production else "development"
        logging.info("Running flow in %s mode.", self.mode)

        self.data = self.load_dataset()

        try:
            # Let's start a new MLflow run to track the execution of this flow. We want
            # to set the name of the MLflow run to the Metaflow run ID so we can easily
            # recognize how they relate to each other.
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_uri}."
            raise RuntimeError(message) from e


        self.next(self.transform)

    @card
    @step
    def transform(self):
        """Apply the transformation pipeline to the entire dataset.

        We'll use the entire dataset to build the final model, so we need to transform
        the dataset before training.

        We want to store the transformers as artifacts so we can later use them
        to transform the input data during inference.
        """
        # Let's build the SciKit-Learn pipeline and transform the dataset features.
        self.features_transformer = build_features_transformer()
        self.x = self.features_transformer.fit_transform(self.data)

        # Let's build the SciKit-Learn pipeline and transform the target column.
        self.target_transformer = build_target_transformer()
        self.y = self.target_transformer.fit_transform(self.data)

        # Now that we have transformed the data, we can train the final model.
        self.next(self.train)

    @card
    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "jax"),
        },
    )
    @resources(memory=4096)
    @step
    def train(self):
        """Train the final model using the entire dataset."""
        import mlflow
        import numpy as np
        from sklearn.metrics import confusion_matrix, accuracy_score

         # Create and store baseline model
        self.model = AdelieClassifier()
        
        # Generate predictions
        y_pred = self.model.predict(self.x)
        y_pred_proba = self.model.predict_proba(self.x)
        
        # Calculate accuracy
        self.test_accuracy = accuracy_score(self.data['species'], y_pred)
      
        logging.info(f"Baseline accuracy: {self.test_accuracy:.4f}")
            

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Let's log the training process under the current MLflow run.
        # Set the run name for the existing run
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id, nested=True):
            mlflow.set_tag("mlflow.runName", "Baseline")
        with mlflow.start_run(run_id=self.mlflow_run_id):
            # We want to log the model manually, so let's disable automatic logging.
            mlflow.autolog(log_models=False)

            mlflow.log_metric("test_accuracy", self.test_accuracy)

            metrics = ClassificationMetrics(y_true=self.data['species'],
                    y_pred=y_pred,
                    labels=self.model.classes_
                    )
            
            combined_path = create_combined_confusion_matrix(metrics, "Baseline Model Performance", "baseline_confusion_matrix")

            # Log to MLflow
            if self.mlflow_run_id:
                with mlflow.start_run(run_id=self.mlflow_run_id, nested=True):
                    mlflow.log_artifact(combined_path, "confusion_matrix")
                    mlflow.log_metrics(
                        {
                            "accuracy": metrics.accuracy,
                        },
                    )
            else:
                mlflow.log_artifact(combined_path, "confusion_matrix")
    
        # Store predictions for model registration
        self.model = y_pred

        # After we finish training the model, we want to register it.
        self.next(self.register)

    def _get_model_artifacts(self, directory: str):
        """Save model artifacts."""
        import joblib

        # Save the simple baseline model
        model_path = (Path(directory) / "model.joblib").as_posix()
        self.model.save(model_path)

        # Save the transformers
        features_transformer_path = (Path(directory) / "features.joblib").as_posix()
        target_transformer_path = (Path(directory) / "target.joblib").as_posix()
        joblib.dump(self.features_transformer, features_transformer_path)
        joblib.dump(self.target_transformer, target_transformer_path)

        return {
            "model": model_path,
            "features_transformer": features_transformer_path,
            "target_transformer": target_transformer_path,
        }

    
    @resources(memory=4096)
    @step
    def register(self):
        """Register the model in the model registry.

        This function will prepare and register the final model in the model registry
        if its accuracy is above a predefined threshold.
        """
        import tempfile

        import mlflow

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Since this is a join step, we need to merge the artifacts from the incoming
        # branches to make them available here.
        #self.merge_artifacts(inputs)

        # We only want to register the model if its accuracy is above the
        # `accuracy_threshold` parameter.
        if self.test_accuracy >= self.accuracy_threshold:
            self.registered = True
            logging.info("Registering model...")

            # We'll register the model under the current MLflow run. We also need to
            # create a temporary directory to store the model artifacts.
            with (
                mlflow.start_run(run_id=self.mlflow_run_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                self.artifacts = self._get_model_artifacts(directory)
                self.pip_requirements = self._get_model_pip_requirements()

                root = Path(__file__).parent
                self.code_paths = [(root / "inference" / "backend.py").as_posix()]

                # We can now register the model in the model registry. This will
                # automatically create a new version of the model.
                mlflow.pyfunc.log_model(
                    python_model=Path(__file__).parent / "inference" / "model.py",
                    registered_model_name="penguins",
                    artifact_path="model",
                    code_paths=self.code_paths,
                    artifacts=self.artifacts,
                    pip_requirements=self.pip_requirements,
                    # Our model expects a Python dictionary, so we want to save the
                    # input example directly as it is by setting`example_no_conversion`
                    # to `True`.
                    example_no_conversion=True,
                )
        else:
            self.registered = False
            logging.info(
                "The accuracy of the model (%.2f) is lower than the accuracy threshold "
                "(%.2f). Skipping model registration.",
                self.test_accuracy,
                self.accuracy_threshold,
            )

        # Let's now move to the final step of the pipeline.
        self.next(self.end)

    @step
    def end(self):
        """End the Training pipeline."""
        logging.info("The pipeline finished successfully.")

    def _get_model_artifacts(self, directory: str):
        """Return the list of artifacts that will be included with model.

        The model must preprocess the raw input data before making a prediction, so we
        need to include the Scikit-Learn transformers as part of the model package.
        """
        import joblib

        # Let's start by saving the model inside the supplied directory.
        model_path = (Path(directory) / "model.keras").as_posix()
        self.model.save(model_path)

        # We also want to save the Scikit-Learn transformers so we can package them
        # with the model and use them during inference.
        features_transformer_path = (Path(directory) / "features.joblib").as_posix()
        target_transformer_path = (Path(directory) / "target.joblib").as_posix()
        joblib.dump(self.features_transformer, features_transformer_path)
        joblib.dump(self.target_transformer, target_transformer_path)

        return {
            "model": model_path,
            "features_transformer": features_transformer_path,
            "target_transformer": target_transformer_path,
        }

    def _get_model_pip_requirements(self):
        """Return the list of required packages to run the model in production."""
        return [
            f"{package}=={version}" if version else package
            for package, version in packages(
                "scikit-learn",
                "pandas",
                "numpy",
                "keras",
                "jax[cpu]",
            ).items()
        ]

    def _create_metrics_summary(self, cm, species_labels, filename, run_id=None):
        """Create a separate metrics summary visualization."""
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        
        # Calculate metrics
        row_sums = np.sum(cm, axis=1)
        col_sums = np.sum(cm, axis=0)
        total = np.sum(cm)
        
        # Per-class metrics
        metrics = []
        for i, species in enumerate(species_labels):
            tp = cm[i, i]  # True positives
            fn = row_sums[i] - tp  # False negatives
            fp = col_sums[i] - tp  # False positives
            
            recall = tp / row_sums[i] if row_sums[i] > 0 else 0
            precision = tp / col_sums[i] if col_sums[i] > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics.append({
                'Species': species,
                'Support': row_sums[i],
                'Recall': recall * 100,
                'Precision': precision * 100,
                'F1 Score': f1
            })
        
        # Create DataFrame
        df = pd.DataFrame(metrics)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')
        
        # Add metrics table
        tbl = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            loc='center',
            cellLoc='center'
        )
        
        # Style the table
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)
        
        # Color the header
        for i, col in enumerate(df.columns):
            tbl[0, i].set_facecolor('#d8e6f3')
            tbl[0, i].set_text_props(fontweight='bold')
        
        # Add title
        plt.title('Performance Metrics by Species', fontsize=14, pad=20)
        
        # Add explanatory note
        note = (
            "Recall: Percentage of actual class correctly predicted\n"
            "Precision: Percentage of predicted class that was correct\n"
            "F1 Score: Harmonic mean of precision and recall (0-1)"
        )
        plt.figtext(0.5, 0.05, note, ha='center', fontsize=9, 
                    bbox=dict(facecolor='#f0f0f0', alpha=0.5, boxstyle='round,pad=0.5'))
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        
        # Save figure
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log metrics to MLflow
        import mlflow
        if run_id:
            with mlflow.start_run(run_id=run_id, nested=True):
                for m in metrics:
                    mlflow.log_metrics({
                        f"{m['Species']}_recall": m['Recall']/100,
                        f"{m['Species']}_precision": m['Precision']/100,
                        f"{m['Species']}_f1": m['F1 Score'],
                        f"{m['Species']}_support": m['Support']
                    })
        else:
            for m in metrics:
                mlflow.log_metrics({
                    f"{m['Species']}_recall": m['Recall']/100,
                    f"{m['Species']}_precision": m['Precision']/100,
                    f"{m['Species']}_f1": m['F1 Score'],
                    f"{m['Species']}_support": m['Support']
                })
    def create_and_log_confusion_matrix(self, cm, species_labels, title, filename_prefix, run_id=None):
        """Create, save, and log a confusion matrix to MLflow.
        
        Args:
            cm: The confusion matrix array
            species_labels: List of class labels
            title: Title for the confusion matrix plot
            filename_prefix: Prefix for saved files
            run_id: MLflow run ID to log to (optional)
        """
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        import mlflow
        import numpy as np
        
        # Create the confusion matrix visualization
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=species_labels
        )
        disp.plot(cmap='Blues', values_format='d')
        plt.title(title)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()
        
        # Save and log the confusion matrix
        cm_path = f'{filename_prefix}.png'
        plt.savefig(cm_path)
        plt.close()
        
        if run_id:
            mlflow.log_artifact(cm_path, "confusion_matrix", run_id=run_id)
        else:
            mlflow.log_artifact(cm_path, "confusion_matrix")
        
        # Also create normalized version (percentages)
        plt.figure(figsize=(10, 8))
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
        
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_norm,
            display_labels=species_labels
        )
        disp.plot(cmap='Blues', values_format='.2%')
        plt.title(f'Normalized {title}')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()
        
        # Save and log the normalized matrix
        cm_norm_path = f'{filename_prefix}_normalized.png'
        plt.savefig(cm_norm_path)
        plt.close()
        
        if run_id:
            mlflow.log_artifact(cm_norm_path, "confusion_matrix", run_id=run_id)
        else:
            mlflow.log_artifact(cm_norm_path, "confusion_matrix")

    def create_and_log_confusion_matrix_improved3(self, cm, species_labels, title, filename_prefix, run_id=None):
        """Create a clearly labeled, comprehensive confusion matrix with key metrics.
        
        Args:
            cm: The confusion matrix array
            species_labels: List of class labels 
            title: Title for the confusion matrix plot
            filename_prefix: Prefix for saved files
            run_id: MLflow run ID to log to (optional)
        """
        import mlflow
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        # Create a figure with stacked subplots - heatmap on top, metrics table below
        # Increase the figure height to allow more space between elements
        fig, (ax_cm, ax_metrics) = plt.subplots(2, 1, figsize=(10, 12),
                                               gridspec_kw={'height_ratios': [1.2, 0.8]})
        
        # 1. Create the main confusion matrix in the first subplot
        
        # Calculate raw counts and percentages (normalized by row = recall)
        cm_sum_rows = np.sum(cm, axis=1, keepdims=True)
        cm_perc_recall = cm / cm_sum_rows * 100
        
        # Create annotation with both counts and percentages
        annot = np.empty_like(cm).astype(str)
        for i in range(len(species_labels)):
            for j in range(len(species_labels)):
                count = cm[i, j]
                percentage = cm_perc_recall[i, j]
                annot[i, j] = f'{count}\n{percentage:.1f}%'
        
        # Plot the confusion matrix
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False,
                    xticklabels=species_labels, yticklabels=species_labels, ax=ax_cm)
        
        # Highlight the diagonal with borders
        for i in range(len(species_labels)):
            ax_cm.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=2))
        
        ax_cm.set_title('Confusion Matrix: Count and Row %\n(% = Class Recall)', fontsize=12, pad=10)
        ax_cm.set_xlabel('Predicted Species', fontsize=12, labelpad=15)  # Increase labelpad
        ax_cm.set_ylabel('True Species', fontsize=12)
        
        # Calculate total counts for each class (row and column sums)
        row_sums = np.sum(cm, axis=1)
        col_sums = np.sum(cm, axis=0)
        total_samples = np.sum(cm)
        
        # Calculate overall accuracy
        accuracy = np.trace(cm) / total_samples
        
        # Add the note about percentages ABOVE the metrics section rather than below the heatmap
        # This avoids overlap with x-axis labels
        ax_metrics.text(0.5, 0.95, 'Note: Percentages are normalized by row (recall)',
                     ha='center', fontsize=10, bbox=dict(facecolor='#f0f0f0', alpha=0.5))
        
        # 2. Create a metrics summary table
        
        # Create a table for key metrics
        metrics_data = []
        headers = ['Species', 'Total\nActual', 'Total\nPredicted', 'Correctly\nClassified', 'Recall', 'Precision']
        
        for i, species in enumerate(species_labels):
            # Calculate metrics
            recall = cm[i, i] / row_sums[i] * 100 if row_sums[i] > 0 else 0
            precision = cm[i, i] / col_sums[i] * 100 if col_sums[i] > 0 else 0
            
            # Add to table data
            metrics_data.append([
                species,                  # Species name
                f"{row_sums[i]}",         # Total actual instances
                f"{col_sums[i]}",         # Total predicted instances
                f"{cm[i, i]}",            # Correctly classified
                f"{recall:.1f}%",         # Recall
                f"{precision:.1f}%"       # Precision
            ])
        
        # Add overall row
        metrics_data.append([
            'Overall',
            f"{total_samples}",
            f"{total_samples}",
            f"{np.trace(cm)}",
            f"{accuracy * 100:.1f}%",
            f"{accuracy * 100:.1f}%"
        ])
        
        # Create table
        table = ax_metrics.table(
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
            table[0, j].set_facecolor('#e6e6ff')
            table[0, j].set_text_props(fontweight='bold')
        
        # Color the Overall row
        for j in range(len(headers)):
            table[len(species_labels) + 1, j].set_facecolor('#e6e6ff')
        
        # Hide the axes for the metrics subplot
        ax_metrics.axis('off')
        
        # Add overall title to the figure
        plt.suptitle(title, fontsize=14, y=0.98)
        
        # Use tight_layout with padding to ensure no overlap
        plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.0)
        
        # Save and log to MLflow
        cm_path = f'{filename_prefix}.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to MLflow
        if run_id:
            mlflow.log_artifact(cm_path, "confusion_matrix", run_id=run_id)
        else:
            mlflow.log_artifact(cm_path, "confusion_matrix")
        
        # Also log metrics
        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.log_metric("overall_accuracy", accuracy)
            # Log per-class metrics
            for i, species in enumerate(species_labels):
                recall = cm[i, i] / row_sums[i] if row_sums[i] > 0 else 0
                precision = cm[i, i] / col_sums[i] if col_sums[i] > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                mlflow.log_metrics({
                    f"{species}_precision": precision,
                    f"{species}_recall": recall,
                    f"{species}_f1": f1,
                    f"{species}_support": int(row_sums[i])
                })
    
    def create_and_log_confusion_matrix_improved4(self, cm, species_labels, title, filename_prefix, run_id=None):
        """Create a clearly labeled, comprehensive confusion matrix with key metrics.
        
        Args:
            cm: The confusion matrix array
            species_labels: List of class labels 
            title: Title for the confusion matrix plot
            filename_prefix: Prefix for saved files
            run_id: MLflow run ID to log to (optional)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd
        
        # Create a figure with two subplots side by side
        fig, (ax_cm, ax_metrics) = plt.subplots(1, 2, figsize=(18, 8), 
                                              gridspec_kw={'width_ratios': [1.2, 0.8]})
        
        # 1. Create the main confusion matrix in the first subplot
        
        # Calculate raw counts and percentages (normalized by row = recall)
        cm_sum_rows = np.sum(cm, axis=1, keepdims=True)
        cm_perc_recall = cm / cm_sum_rows * 100
        
        # Create annotation with both counts and percentages
        annot = np.empty_like(cm).astype(str)
        for i in range(len(species_labels)):
            for j in range(len(species_labels)):
                count = cm[i, j]
                percentage = cm_perc_recall[i, j]
                annot[i, j] = f'{count}\n{percentage:.1f}%'
        
        # Plot the confusion matrix
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False,
                    xticklabels=species_labels, yticklabels=species_labels, ax=ax_cm)
        
        ax_cm.set_title('Confusion Matrix: Count and Row %\n(% = Class Recall)', fontsize=12, pad=10)
        ax_cm.set_xlabel('Predicted Species', fontsize=12)
        ax_cm.set_ylabel('True Species', fontsize=12)
        
        # Add a note about the percentages
        ax_cm.annotate('Note: Percentages are normalized by row (recall)',
                     xy=(0.5, -0.15), xycoords='axes fraction', ha='center', fontsize=10)
        
        # 2. Create a metrics summary table in the second subplot
        
        # Calculate key metrics
        accuracy = np.trace(cm) / np.sum(cm)
        
        # Per-class metrics
        metrics_data = []
        for i, species in enumerate(species_labels):
            # True positives, false positives, etc.
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - tp - fp - fn
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store metrics
            metrics_data.append([
                species,
                int(np.sum(cm[i, :])),  # Total actual
                int(np.sum(cm[:, i])),  # Total predicted
                tp,                     # True positives
                f"{precision:.1%}",     # Precision
                f"{recall:.1%}",        # Recall
                f"{f1:.2f}"             # F1 score
            ])
        
        # Turn off the axes for the metrics subplot
        ax_metrics.axis('off')
        
        # Create the metrics table
        metrics_table = ax_metrics.table(
            cellText=metrics_data,
            colLabels=['Class', 'Total\nActual', 'Total\nPredicted', 'True\nPositives', 'Precision', 'Recall', 'F1\nScore'],
            loc='center',
            cellLoc='center',
            colWidths=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
        )
        
        # Style the metrics table
        metrics_table.auto_set_font_size(False)
        metrics_table.set_fontsize(9)
        metrics_table.scale(1, 1.5)
        
        # Header style
        for i in range(len(metrics_table.get_celld())):
            if i < 7:  # Only style the header row
                cell = metrics_table[0, i]
                cell.set_height(0.15)
                cell.set_text_props(fontweight='bold')
                cell.set_facecolor('#e6e6ff')
        
        # Add overall accuracy at the top of metrics panel
        ax_metrics.text(0.5, 0.9, f"Overall Accuracy: {accuracy:.2%}", 
                      fontsize=14, fontweight='bold', ha='center')
        
        # Add explanatory text for the metrics
        metrics_explanation = (
            "Precision = TP / (TP + FP) - How many predicted as X are actually X\n"
            "Recall = TP / (TP + FN) - How many actual X were correctly predicted\n"
            "F1 Score = Harmonic mean of precision and recall"
        )
        ax_metrics.text(0.5, 0.05, metrics_explanation, fontsize=9, ha='center', 
                      va='center', bbox=dict(facecolor='#f0f0f0', alpha=0.5))
        
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save and log to MLflow
        cm_path = f'{filename_prefix}_enhanced.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to MLflow
        import mlflow
        if run_id:
            mlflow.log_artifact(cm_path, "confusion_matrix", run_id=run_id)
        else:
            mlflow.log_artifact(cm_path, "confusion_matrix")
        
        # Also log metrics
        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.log_metric("overall_accuracy", accuracy)
            # Log per-class metrics
            for i, species in enumerate(species_labels):
                tp = cm[i, i]
                fp = np.sum(cm[:, i]) - tp
                fn = np.sum(cm[i, :]) - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                mlflow.log_metrics({
                    f"{species}_precision": precision,
                    f"{species}_recall": recall,
                    f"{species}_f1": f1,
                    f"{species}_support": int(np.sum(cm[i, :]))
                })

    def create_and_log_confusion_matrix_improved2(self, cm, species_labels, title, filename_prefix, run_id=None):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import seaborn as sns
        import numpy as np

        total = np.sum(cm)
        labels = [[f"{val:0.0f}\n{val / total:.2%}" for val in row] for row in cm]
        states = species_labels

        ax = sns.heatmap(cm, annot=labels, cmap='Reds', fmt='',
                        xticklabels=states, yticklabels=states, cbar=False)
        ax.set_title('Predicted Species', fontweight='bold')
        ax.tick_params(labeltop=True, labelbottom=False, length=0)
        ax.set_ylabel('Actual Species', fontweight='bold')

        # matrix for the extra column and row
        f_mat = np.zeros((cm.shape[0] + 1, cm.shape[1] + 1))
        f_mat[:-1, -1] = np.diag(cm) / np.sum(cm, axis=1)  # fill recall column
        f_mat[-1, :-1] = np.diag(cm) / np.sum(cm, axis=0)  # fill precision row
        f_mat[-1, -1] = np.trace(cm) / np.sum(cm)  # accuracy

        f_mask = np.ones_like(f_mat)  # puts 1 for masked elements
        f_mask[:, -1] = 0  # last column will be unmasked
        f_mask[-1, :] = 0  # last row will be unmasked

        # matrix for coloring the heatmap
        # only last row and column will be used due to masking
        f_color = np.ones_like(f_mat)
        f_color[-1, -1] = 0  # lower right gets different color

        # matrix of annotations, only last row and column will be used
        f_annot = [[f"{val:0.2%}" for val in row] for row in f_mat]
        f_annot[-1][-1] = "Acc.:\n" + f_annot[-1][-1]

        sns.heatmap(f_color, mask=f_mask, annot=f_annot, fmt='',
                    xticklabels=states + ["Recall"],
                    yticklabels=states + ["Precision"],
                    cmap=ListedColormap(['skyblue', 'lightgrey']), cbar=False, ax=ax)
        """ plt.show()

        plt.text(1.45, 0.6, summary_text,
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8)) """
        
        # Save and log the matrix
        plt.tight_layout()
        cm_path = f'{filename_prefix}_enhanced.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

         # Log to MLflow
        import mlflow
        if run_id:
            mlflow.log_artifact(cm_path, "confusion_matrix", run_id=run_id)
        else:
            mlflow.log_artifact(cm_path, "confusion_matrix")

    def create_and_log_confusion_matrix_improved(self, cm, species_labels, title, filename_prefix, run_id=None):
        """Create enhanced confusion matrix with summary statistics and percentages.
    
        Args:
            cm: The confusion matrix array
            species_labels: List of class labels
            title: Title for the confusion matrix plot
            filename_prefix: Prefix for saved files
            run_id: MLflow run ID to log to (optional)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd
        from matplotlib.ticker import PercentFormatter
        
        # Set up the style
        sns.set(font_scale=1.3)
        
        # Calculate percentages
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        
        # Create annotation text
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                count = cm[i, j]
                percentage = cm_perc[i, j]
                if i == j:
                    total = cm_sum[i][0]
                    annot[i, j] = f'{percentage:.1f}%\n{count}/{total}'
                else:
                    annot[i, j] = f'{percentage:.1f}%\n{count}'
        
        # Create DataFrame for seaborn
        cm_df = pd.DataFrame(cm_perc, 
                            index=species_labels, 
                            columns=species_labels)
        cm_df.index.name = 'True Species'
        cm_df.columns.name = 'Predicted Species'
        
        # Create main confusion matrix plot
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        
        # Plot heatmap with both counts and percentages
        sns.heatmap(cm_df, 
                    annot=annot, 
                    fmt='', 
                    ax=ax,
                    cmap='Blues',
                    cbar_kws={'format': PercentFormatter()})
        
        # Customize the plot
        plt.title(f'{title}\nTotal Samples: {np.sum(cm):,}')
        
        # Add summary statistics as text
        accuracy = np.trace(cm) / np.sum(cm) * 100
        class_accuracy = np.diag(cm) / np.sum(cm, axis=1) * 100
        
        summary_text = (
            f"Overall Accuracy: {accuracy:.1f}%\n\n"
            f"Per-Class Accuracy:\n"
            f"Adelie: {class_accuracy[0]:.1f}%\n"
            f"Chinstrap: {class_accuracy[1]:.1f}%\n"
            f"Gentoo: {class_accuracy[2]:.1f}%"
        )
        
        plt.text(1.45, 0.6, summary_text,
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Save and log the matrix
        plt.tight_layout()
        cm_path = f'{filename_prefix}_enhanced.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Log to MLflow
        import mlflow
        if run_id:
            mlflow.log_artifact(cm_path, "confusion_matrix", run_id=run_id)
        else:
            mlflow.log_artifact(cm_path, "confusion_matrix")

if __name__ == "__main__":
    BaselineModel()

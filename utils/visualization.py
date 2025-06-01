# sparkforge/utils/visualization.py

"""
Visualization utilities for SparkForge framework.

This module provides comprehensive visualization tools for model evaluation,
feature analysis, and data exploration using matplotlib and seaborn.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pyspark.sql import DataFrame
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path


class SparkForgeVisualizer:
    """
    Comprehensive visualization utilities for SparkForge framework.

    Provides methods for creating various plots and charts for model evaluation,
    feature analysis, and data exploration.
    """

    def __init__(
        self,
        style: str = "whitegrid",
        palette: str = "husl",
        figure_size: Tuple[int, int] = (12, 8),
    ):
        """
        Initialize visualizer with style settings.

        Args:
            style: Seaborn style
            palette: Color palette
            figure_size: Default figure size
        """

        self.logger = logging.getLogger(__name__)

        # Set style
        plt.style.use("default")
        sns.set_style(style)
        sns.set_palette(palette)

        self.figure_size = figure_size
        self.colors = sns.color_palette(palette)
        self.output_dir = Path("plots")
        self.output_dir.mkdir(exist_ok=True)

    def plot_model_comparison(
        self,
        model_results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        title: str = "Model Performance Comparison",
        save_path: str = None,
    ) -> plt.Figure:
        """
        Create a comprehensive model comparison plot.

        Args:
            model_results (Dict[str, Dict[str, float]]): Dictionary of model results
            metrics (list): List of metrics to plot
            title (str): Plot title
            save_path (str): Path to save the plot

        Returns:
            Matplotlib figure
        """

        if metrics is None:
            all_metrics = set()
            for results in model_results.values():
                all_metrics.update(results.keys())

            metrics = [
                m for m in all_metrics if not m.startswith(("confusion", "class_"))
            ]

        plot_data = []
        for model_name, results in model_results.items():
            for metric in metrics:
                if metric in results:
                    plot_data.append(
                        {
                            "Model": model_name,
                            "Metric": metric,
                            "Score": results[metric],
                        }
                    )

        df = pd.DataFrame(plot_data)

        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(self.figure_size[0], self.figure_size[1] * n_rows / 2),
        )

        if n_metrics == 1:
            axes = [axes]

        elif n_rows == 1:
            axes = axes.flatten()

        else:
            axes = axes.flatten()

        for i, metric in enumerate(metrics):
            metric_data = df[df["Metric"] == metric]

            if len(metric_data) > 0:
                sns.barplot(data=metric_data, x="Model", y="Score", ax=axes[i])
                axes[i].set_title(f"{metric.capitalize()}")
                axes[i].set_xlabel("")
                axes[i].tick_params(axis="x", rotation=45)

                for container in axes[i].containers:
                    axes[i].bar_label(container, fmt="%.3f")

        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            self._save_plot(fig, save_path)

        return fig

    def plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        top_k: int = 20,
        title: str = "Feature Importance",
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plot feature importance rankings.

        Args:
            importance_dict (Dict[str, float]): Dictionary of feature importance scores
            top_k (int): Number of top features to show
            title (str): Plot title
            save_path (str): Path to save the plot

        Returns:
            Matplotlib figure
        """

        sorted_features = sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        features, importances = zip(*sorted_features)
        fig, ax = plt.subplots(figsize=(self.figure_size[0], max(8, top_k * 0.4)))

        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importances, color=self.colors[0])

        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")
        ax.set_title(title)

        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax.text(
                bar.get_width() + max(importances) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{importance:.3f}",
                va="center",
                ha="left",
            )

        plt.tight_layout()
        if save_path:
            self._save_plot(fig, save_path)

        return fig

    def plot_confusion_matrix(
        self,
        confusion_matrix: List[List[int]],
        class_labels: List[str] = None,
        title: str = "Confusion Matrix",
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plot confusion matrix heatmap.

        Args:
            confusion_matrix (List[List[int]]): Confusion matrix as 2D list
            class_labels (List[str]): Class labels
            title (str): Plot title
            save_path (str): Path to save the plot

        Returns:
            Matplotlib figure
        """

        # Convert to numpy array
        cm = np.array(confusion_matrix)

        # Create labels if not provided
        if class_labels is None:
            class_labels = [f"Class {i}" for i in range(cm.shape[0])]

        # Create plot
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
        )

        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        plt.tight_layout()
        if save_path:
            self._save_plot(fig, save_path)

        return fig

    def plot_learning_curves(
        self,
        train_scores: List[float],
        val_scores: List[float],
        train_sizes: List[int] = None,
        metric_name: str = "Score",
        title: str = "Learning Curves",
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plot learning curves showing training and validation performance.

        Args:
            train_scores (List[float]): Training scores over iterations/sizes
            val_scores (List[float]): Validation scores over iterations/sizes
            train_sizes (List[int]): Training set sizes (if None, uses indices)
            metric_name (str): Name of the metric being plotted
            title (str): Plot title
            save_path (str): Path to save the plot

        Returns:
            Matplotlib figure
        """

        if train_sizes is None:
            train_sizes = list(range(1, len(train_scores) + 1))

        fig, ax = plt.subplots(figsize=self.figure_size)
        ax.plot(
            train_sizes,
            train_scores,
            "o-",
            color=self.colors[0],
            label="Training Score",
            linewidth=2,
            markersize=6,
        )
        ax.plot(
            train_sizes,
            val_scores,
            "o-",
            color=self.colors[1],
            label="Validation Score",
            linewidth=2,
            markersize=6,
        )

        ax.set_xlabel("Training Set Size" if len(set(train_sizes)) > 2 else "Iteration")
        ax.set_ylabel(metric_name)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            self._save_plot(fig, save_path)

        return fig

    def plot_residuals(
        self,
        predictions_df: DataFrame,
        label_col: str = "label",
        prediction_col: str = "prediction",
        sample_size: int = 10000,
        title: str = "Residual Analysis",
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plot residual analysis for regression models.

        Args:
            predictions_df (DataFrame): DataFrame with predictions and labels
            label_col (str): Label column name
            prediction_col (str): Prediction column name
            sample_size (int): Sample size for plotting
            title (str): Plot title
            save_path (str): Path to save the plot

        Returns:
            Matplotlib figure
        """

        sample_df = predictions_df.sample(
            False, min(1.0, sample_size / predictions_df.count())
        )

        pandas_df = sample_df.toPandas()
        pandas_df["residuals"] = pandas_df[label_col] - pandas_df[prediction_col]

        fig, axes = plt.subplots(
            2, 2, figsize=(self.figure_size[0], self.figure_size[1])
        )

        axes[0, 0].scatter(
            pandas_df[prediction_col],
            pandas_df["residuals"],
            alpha=0.6,
            color=self.colors[0],
        )
        axes[0, 0].axhline(y=0, color="red", linestyle="--")
        axes[0, 0].set_xlabel("Predicted Values")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].set_title("Residuals vs Predicted")

        axes[0, 1].scatter(
            pandas_df[label_col],
            pandas_df["residuals"],
            alpha=0.6,
            color=self.colors[1],
        )
        axes[0, 1].axhline(y=0, color="red", linestyle="--")
        axes[0, 1].set_xlabel("Actual Values")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].set_title("Residuals vs Actual")

        axes[1, 0].hist(
            pandas_df["residuals"], bins=50, alpha=0.7, color=self.colors[2]
        )
        axes[1, 0].set_xlabel("Residuals")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Distribution of Residuals")

        from scipy import stats

        stats.probplot(pandas_df["residuals"], dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("Q-Q Plot of Residuals")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            self._save_plot(fig, save_path)

        return fig

    def plot_roc_curve(
        self,
        predictions_df: DataFrame,
        label_col: str = "label",
        probability_col: str = "probability",
        title: str = "ROC Curve",
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plot ROC curve for binary classification.

        Args:
            predictions_df (DataFrame): DataFrame with predictions and probabilities
            label_col (str): Label column name
            probability_col (str): Probability column name
            title (str): Plot title
            save_path (str): Path to save the plot

        Returns:
            Matplotlib figure
        """

        from sklearn.metrics import roc_curve, auc

        pandas_df = predictions_df.toPandas()
        if isinstance(pandas_df[probability_col].iloc[0], (list, np.ndarray)):
            y_prob = [
                prob[1] if len(prob) > 1 else prob[0]
                for prob in pandas_df[probability_col]
            ]

        else:
            y_prob = pandas_df[probability_col]

        y_true = pandas_df[label_col]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=self.figure_size)
        ax.plot(
            fpr,
            tpr,
            color=self.colors[0],
            linewidth=2,
            label=f"ROC Curve (AUC = {roc_auc:.3f})",
        )
        ax.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random Classifier")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            self._save_plot(fig, save_path)

        return fig

    def plot_precision_recall_curve(
        self,
        predictions_df: DataFrame,
        label_col: str = "label",
        probability_col: str = "probability",
        title: str = "Precision-Recall Curve",
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plot Precision-Recall curve for binary classification.

        Args:
            predictions_df (DataFrame): DataFrame with predictions and probabilities
            label_col (str): Label column name
            probability_col (str): Probability column name
            title (str): Plot title
            save_path (str): Path to save the plot

        Returns:
            Matplotlib figure
        """

        from sklearn.metrics import precision_recall_curve, average_precision_score

        pandas_df = predictions_df.toPandas()
        if isinstance(pandas_df[probability_col].iloc[0], (list, np.ndarray)):
            y_prob = [
                prob[1] if len(prob) > 1 else prob[0]
                for prob in pandas_df[probability_col]
            ]

        else:
            y_prob = pandas_df[probability_col]

        y_true = pandas_df[label_col]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=self.figure_size)
        ax.plot(
            recall,
            precision,
            color=self.colors[0],
            linewidth=2,
            label=f"PR Curve (AP = {avg_precision:.3f})",
        )

        positive_ratio = sum(y_true) / len(y_true)
        ax.axhline(
            y=positive_ratio,
            color="red",
            linestyle="--",
            label=f"Random Classifier (AP = {positive_ratio:.3f})",
        )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            self._save_plot(fig, save_path)

        return fig

    def plot_data_distribution(
        self,
        df: DataFrame,
        columns: List[str],
        sample_size: int = 10000,
        title: str = "Data Distribution",
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plot distribution of selected columns.

        Args:
            df (DataFrame): Spark DataFrame
            columns (list): List of columns to plot
            sample_size (float): Sample size for plotting
            title (str): Plot title
            save_path (str): Path to save the plot

        Returns:
            Matplotlib figure
        """

        sample_df = df.select(columns).sample(False, min(1.0, sample_size / df.count()))
        pandas_df = sample_df.toPandas()

        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(self.figure_size[0], self.figure_size[1] * n_rows / 2),
        )

        if len(columns) == 1:
            axes = [axes]

        elif n_rows == 1:
            axes = axes.flatten()

        else:
            axes = axes.flatten()

        for i, col in enumerate(columns):
            if col in pandas_df.columns:
                if pandas_df[col].dtype in ["int64", "float64"]:
                    axes[i].hist(
                        pandas_df[col].dropna(),
                        bins=50,
                        alpha=0.7,
                        color=self.colors[i % len(self.colors)],
                    )

                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel("Frequency")

                else:
                    value_counts = pandas_df[col].value_counts().head(10)
                    axes[i].bar(
                        range(len(value_counts)),
                        value_counts.values,
                        color=self.colors[i % len(self.colors)],
                    )

                    axes[i].set_xticks(range(len(value_counts)))
                    axes[i].set_xticklabels(value_counts.index, rotation=45, ha="right")
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel("Count")

                axes[i].set_title(f"Distribution of {col}")
                axes[i].grid(True, alpha=0.3)

        for i in range(len(columns), len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            self._save_plot(fig, save_path)

        return fig

    def plot_correlation_matrix(
        self,
        df: DataFrame,
        columns: List[str] = None,
        sample_size: int = 10000,
        title: str = "Correlation Matrix",
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plot correlation matrix heatmap.

        Args:
            df (DataFrame): Spark DataFrame
            columns (list): List of columns to include (None for all numeric)
            sample_size (int): Sample size for correlation calculation
            title (str): Plot title
            save_path (str): Path to save the plot

        Returns:
            Matplotlib figure
        """

        if columns is None:
            sample_df = df.limit(100).toPandas()
            columns = list(sample_df.select_dtypes(include=[np.number]).columns)

        sample_df = df.select(columns).sample(False, min(1.0, sample_size / df.count()))
        pandas_df = sample_df.toPandas()

        corr_matrix = pandas_df.corr()
        fig, ax = plt.subplots(
            figsize=(max(8, len(columns)), max(6, len(columns) * 0.8))
        )

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )

        ax.set_title(title)
        plt.tight_layout()

        if save_path:
            self._save_plot(fig, save_path)

        return fig

    def plot_feature_group_importance(
        self,
        group_analysis: Dict[str, Dict[str, Any]],
        title: str = "Feature Group Importance",
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plot feature importance by groups.

        Args:
            group_analysis (Dict[str, Dict[str, Any]]): Results from feature group analysis
            title (str): Plot title
            save_path (str): Path to save the plot

        Returns:
            Matplotlib figure
        """

        groups = list(group_analysis.keys())
        total_importance = [
            group_analysis[group]["total_importance"] for group in groups
        ]

        avg_importance = [group_analysis[group]["avg_importance"] for group in groups]
        feature_counts = [group_analysis[group]["feature_count"] for group in groups]

        fig, axes = plt.subplots(
            1, 3, figsize=(self.figure_size[0] * 1.5, self.figure_size[1] * 0.7)
        )

        bars1 = axes[0].bar(groups, total_importance, color=self.colors[: len(groups)])
        axes[0].set_title("Total Importance by Group")
        axes[0].set_ylabel("Total Importance")
        axes[0].tick_params(axis="x", rotation=45)

        for bar in bars1:
            height = bar.get_height()
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

        bars2 = axes[1].bar(groups, avg_importance, color=self.colors[: len(groups)])
        axes[1].set_title("Average Importance by Group")
        axes[1].set_ylabel("Average Importance")
        axes[1].tick_params(axis="x", rotation=45)

        for bar in bars2:
            height = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

        bars3 = axes[2].bar(groups, feature_counts, color=self.colors[: len(groups)])
        axes[2].set_title("Feature Count by Group")
        axes[2].set_ylabel("Number of Features")
        axes[2].tick_params(axis="x", rotation=45)

        for bar in bars3:
            height = bar.get_height()
            axes[2].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            self._save_plot(fig, save_path)

        return fig

    def plot_hyperparameter_tuning_results(
        self,
        tuning_results: Dict[str, Any],
        param_name: str,
        metric_name: str = "score",
        title: str = "Hyperparameter Tuning Results",
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plot hyperparameter tuning results.

        Args:
            tuning_results (Dict[str, Any]): Results from hyperparameter tuning
            param_name (str): Parameter name to plot
            metric_name (str): Metric name to plot
            title (str): Plot title
            save_path (str): Path to save the plot

        Returns:
            Matplotlib figure
        """

        param_values = tuning_results.get("param_values", [])
        scores = tuning_results.get("scores", [])

        if not param_values or not scores:
            self.logger.warning("No tuning results data available for plotting")
            return None

        fig, ax = plt.subplots(figsize=self.figure_size)
        if isinstance(param_values[0], (int, float)):
            ax.plot(
                param_values,
                scores,
                "o-",
                color=self.colors[0],
                linewidth=2,
                markersize=8,
            )

            ax.set_xlabel(param_name)

        else:
            ax.bar(range(len(param_values)), scores, color=self.colors[0])
            ax.set_xticks(range(len(param_values)))
            ax.set_xticklabels(param_values, rotation=45, ha="right")
            ax.set_xlabel(param_name)

        ax.set_ylabel(metric_name)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        best_idx = np.argmax(scores)
        ax.annotate(
            f"Best: {param_values[best_idx]}\nScore: {scores[best_idx]:.3f}",
            xy=(
                (
                    best_idx
                    if not isinstance(param_values[0], (int, float))
                    else param_values[best_idx]
                ),
                scores[best_idx],
            ),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

        plt.tight_layout()
        if save_path:
            self._save_plot(fig, save_path)

        return fig

    def plot_model_complexity_curve(
        self,
        complexity_params: List[Any],
        train_scores: List[float],
        val_scores: List[float],
        param_name: str = "Model Complexity",
        title: str = "Model Complexity Curve",
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plot model complexity vs performance curve.

        Args:
            complexity_params (List[Any]): List of complexity parameter values
            train_scores (List[float]): Training scores
            val_scores (List[float]): Validation scores
            param_name (str): Name of the complexity parameter
            title (str): Plot title
            save_path (str): Path to save the plot

        Returns:
            Matplotlib figure
        """

        fig, ax = plt.subplots(figsize=self.figure_size)
        ax.plot(
            complexity_params,
            train_scores,
            "o-",
            color=self.colors[0],
            label="Training Score",
            linewidth=2,
            markersize=6,
        )

        ax.plot(
            complexity_params,
            val_scores,
            "o-",
            color=self.colors[1],
            label="Validation Score",
            linewidth=2,
            markersize=6,
        )

        ax.set_xlabel(param_name)
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        best_idx = np.argmax(val_scores)
        ax.axvline(
            x=complexity_params[best_idx],
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Optimal: {complexity_params[best_idx]}",
        )

        ax.legend()
        plt.tight_layout()

        if save_path:
            self._save_plot(fig, save_path)

        return fig

    def create_model_dashboard(
        self,
        model_results: Dict[str, Any],
        feature_importance: Dict[str, float] = None,
        predictions_df: DataFrame = None,
        save_path: str = None,
    ) -> plt.Figure:
        """
        Create a comprehensive model performance dashboard.

        Args:
            model_results: Model evaluation results
            feature_importance: Feature importance scores
            predictions_df: DataFrame with predictions
            save_path: Path to save the dashboard

        Returns:
            Matplotlib figure
        """

        fig = plt.figure(figsize=(20, 16))
        ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2)
        metrics = {
            k: v
            for k, v in model_results.items()
            if isinstance(v, (int, float)) and not k.startswith("confusion")
        }

        if metrics:
            metric_names = list(metrics.keys())[:6]
            metric_values = [metrics[name] for name in metric_names]

            bars = ax1.bar(
                metric_names, metric_values, color=self.colors[: len(metric_names)]
            )

            ax1.set_title("Key Performance Metrics", fontsize=14, fontweight="bold")
            ax1.tick_params(axis="x", rotation=45)

            for bar in bars:
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                )

        if feature_importance:
            ax2 = plt.subplot2grid((4, 4), (0, 2), colspan=2)
            top_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:10]
            features, importances = zip(*top_features)

            y_pos = np.arange(len(features))
            ax2.barh(y_pos, importances, color=self.colors[2])
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(features)
            ax2.invert_yaxis()
            ax2.set_title("Top 10 Feature Importance", fontsize=14, fontweight="bold")

        if "confusion_matrix" in model_results:
            ax3 = plt.subplot2grid((4, 4), (1, 0), colspan=2, rowspan=2)
            cm = np.array(model_results["confusion_matrix"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
            ax3.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
            ax3.set_xlabel("Predicted")
            ax3.set_ylabel("Actual")

        if predictions_df is not None:
            ax4 = plt.subplot2grid((4, 4), (1, 2), colspan=2, rowspan=2)

            try:
                sample_df = predictions_df.limit(1000).toPandas()
                if "probability" in sample_df.columns:
                    from sklearn.metrics import roc_curve, auc

                    y_true = sample_df["label"]
                    y_prob = [
                        (
                            prob[1]
                            if isinstance(prob, (list, np.ndarray)) and len(prob) > 1
                            else prob
                        )
                        for prob in sample_df["probability"]
                    ]

                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    roc_auc = auc(fpr, tpr)

                    ax4.plot(
                        fpr,
                        tpr,
                        color=self.colors[1],
                        linewidth=2,
                        label=f"ROC Curve (AUC = {roc_auc:.3f})",
                    )
                    ax4.plot([0, 1], [0, 1], "r--", label="Random")
                    ax4.set_xlabel("False Positive Rate")
                    ax4.set_ylabel("True Positive Rate")
                    ax4.set_title("ROC Curve", fontsize=14, fontweight="bold")
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)

            except:
                try:
                    sample_df = predictions_df.limit(1000).toPandas()
                    residuals = sample_df["label"] - sample_df["prediction"]

                    ax4.scatter(
                        sample_df["prediction"],
                        residuals,
                        alpha=0.6,
                        color=self.colors[3],
                    )
                    ax4.axhline(y=0, color="red", linestyle="--")
                    ax4.set_xlabel("Predicted Values")
                    ax4.set_ylabel("Residuals")
                    ax4.set_title("Residual Plot", fontsize=14, fontweight="bold")

                except:
                    ax4.text(
                        0.5,
                        0.5,
                        "No prediction data\navailable for plotting",
                        ha="center",
                        va="center",
                        transform=ax4.transAxes,
                    )
                    ax4.set_title(
                        "Predictions Analysis", fontsize=14, fontweight="bold"
                    )

        ax5 = plt.subplot2grid((4, 4), (3, 0), colspan=4)
        ax5.axis("off")

        info_text = f"""
        Model Performance Summary
        
        Best Metric: {max(metrics.values()):.4f} | Worst Metric: {min(metrics.values()):.4f}
        Total Features: {len(feature_importance) if feature_importance else 'N/A'}
        """

        ax5.text(
            0.1,
            0.5,
            info_text,
            transform=ax5.transAxes,
            fontsize=12,
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5),
        )

        plt.suptitle(
            "Model Performance Dashboard", fontsize=20, fontweight="bold", y=0.98
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)

        if save_path:
            self._save_plot(fig, save_path)

        return fig

    def _save_plot(self, fig: plt.Figure, save_path: str):
        """
        Save plot to file
        """

        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )

            self.logger.info(f"Plot saved to: {save_path}")

        except Exception as e:
            self.logger.error(f"Error saving plot to {save_path}: {e}")

    def set_style(self, style: str = "whitegrid", palette: str = "husl"):
        """
        Update visualization style
        """

        sns.set_style(style)
        sns.set_palette(palette)
        self.colors = sns.color_palette(palette)
        self.logger.info(f"Visualization style updated: {style}, {palette}")


def quick_model_comparison(
    model_results: Dict[str, Dict[str, float]], save_path: str = None
) -> plt.Figure:
    """
    Quick model comparison plot
    """

    visualizer = SparkForgeVisualizer()
    return visualizer.plot_model_comparison(model_results, save_path=save_path)


def quick_feature_importance(
    importance_dict: Dict[str, float], top_k: int = 20, save_path: str = None
) -> plt.Figure:
    """
    Quick feature importance plot
    """

    visualizer = SparkForgeVisualizer()
    return visualizer.plot_feature_importance(
        importance_dict, top_k=top_k, save_path=save_path
    )


def quick_confusion_matrix(
    confusion_matrix: List[List[int]],
    class_labels: List[str] = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Quick confusion matrix plot
    """

    visualizer = SparkForgeVisualizer()
    return visualizer.plot_confusion_matrix(
        confusion_matrix, class_labels=class_labels, save_path=save_path
    )

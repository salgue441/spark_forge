# sparkforge/utils/evaluation.py

"""
Evaluation utilities for SparkForge framework.

This module provides comprehensive model evaluation metrics, comparison tools,
and statistical testing utilities for machine learning models.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.ml.evaluation import *
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Comprehensive model evaluation utilities for classification and regression.

    Provides methods for calculating various metrics, comparing models,
    and performing statistical significance tests.
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize model evaluator.

        Args:
            spark: SparkSession instance
        """

        self.spark = spark
        self.logger = logging.getLogger(__name__)

    def evaluate_classification(
        self,
        predictions_df: DataFrame,
        label_col: str = "label",
        prediction_col: str = "prediction",
        probability_col: str = "probability",
        metrics: List[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate classification model performance.

        Args:
            predictions_df (DataFrame): DataFrame with predictions
            label_col (str): Label column name
            prediction_col (str): Prediction column name
            probability_col (str): Probability column name
            metrics (list): List of metrics to compute

        Returns:
            Dictionary with computed metrics
        """

        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1", "auc", "log_loss"]

        results = {}

        try:
            if "accuracy" in metrics:
                accuracy_evaluator = MulticlassClassificationEvaluator(
                    labelCol=label_col,
                    predictionCol=prediction_col,
                    metricName="accuracy",
                )

                results["accuracy"] = accuracy_evaluator.evaluate(predictions_df)

            if "precision" in metrics:
                precision_evaluator = MulticlassClassificationEvaluator(
                    labelCol=label_col,
                    predictionCol=prediction_col,
                    metricName="weightedPrecision",
                )

                results["precision"] = precision_evaluator.evaluate(predictions_df)

            if "recall" in metrics:
                recall_evaluator = MulticlassClassificationEvaluator(
                    labelCol=label_col,
                    predictionCol=prediction_col,
                    metricName="weightedRecall",
                )

                results["recall"] = recall_evaluator.evaluate(predictions_df)

            if "f1" in metrics:
                f1_evaluator = MulticlassClassificationEvaluator(
                    labelCol=label_col, predictionCol=prediction_col, metricName="f1"
                )

                results["f1"] = f1_evaluator.evaluate(predictions_df)

            if self._is_binary_classification(predictions_df, label_col):
                if "auc" in metrics:
                    auc_evaluator = BinaryClassificationEvaluator(
                        labelCol=label_col,
                        rawPredictionCol=probability_col,
                        metricName="areaUnderROC",
                    )

                    results["auc"] = auc_evaluator.evaluate(predictions_df)

                if "log_loss" in metrics and probability_col in predictions_df.columns:
                    results["log_loss"] = self._calculate_log_loss(
                        predictions_df, label_col, probability_col
                    )

                class_metrics = self._calculate_per_class_metrics(
                    predictions_df, label_col, prediction_col
                )

                results.update(class_metrics)

            confusion_matrix = self._calculate_confusion_matrix(
                predictions_df, label_col, prediction_col
            )

            results["confusion_matrix"] = confusion_matrix
            self.logger.info("Classification evaluation completed")

        except Exception as e:
            self.logger.error(f"Error in classification evaluation: {e}")
            raise

        return results

    def evaluate_regression(
        self,
        predictions_df: DataFrame,
        label_col: str = "label",
        prediction_col: str = "prediction",
        metrics: List[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate regression model performance.

        Args:
            predictions_df (DataFrame): DataFrame with predictions
            label_col (str): Label column name
            prediction_col (str): Prediction column name
            metrics (list): List of metrics to compute

        Returns:
            Dictionary with computed metrics
        """

        if metrics is None:
            metrics = ["rmse", "mae", "r2", "explained_variance"]

        results = {}
        try:
            if "rmse" in metrics:
                rmse_evaluator = RegressionEvaluator(
                    labelCol=label_col, predictionCol=prediction_col, metricName="rmse"
                )

                results["rmse"] = rmse_evaluator.evaluate(predictions_df)

            if "mae" in metrics:
                mae_evaluator = RegressionEvaluator(
                    labelCol=label_col, predictionCol=prediction_col, metricName="mae"
                )

                results["mae"] = mae_evaluator.evaluate(predictions_df)

            if "r2" in metrics:
                r2_evaluator = RegressionEvaluator(
                    labelCol=label_col, predictionCol=prediction_col, metricName="r2"
                )

                results["r2"] = r2_evaluator.evaluate(predictions_df)

            if "explained_variance" in metrics:
                results["explained_variance"] = self._calculate_explained_variance(
                    predictions_df, label_col, prediction_col
                )

            if "mape" in metrics:
                results["mape"] = self._calculate_mape(
                    predictions_df, label_col, prediction_col
                )

            residual_stats = self._calculate_residual_stats(
                predictions_df, label_col, prediction_col
            )

            results.update(residual_stats)
            self.logger.info("Regression evaluation completed")

        except Exception as e:
            self.logger.error(f"Error in regression evaluation: {e}")
            raise

        return results

    def compare_models(
        self,
        model_results: Dict[str, Dict[str, float]],
        primary_metric: str,
        statistical_test: str = "paired_ttest",
    ) -> Dict[str, Any]:
        """
        Compare multiple models using statistical tests.

        Args:
            model_results (Dict[str, Dict[str, float]]): Dictionary of model results
            primary_metric (str): Primary metric for comparison
            statistical_test (str): Statistical test to use

        Returns:
            Comparison results
        """

        comparison_results = {
            "best_model": None,
            "ranking": [],
            "statistical_tests": {},
            "summary": {},
        }

        try:
            metric_values = {}
            for model_name, results in model_results.items():
                if primary_metric in results:
                    metric_values[model_name] = results[primary_metric]

            if not metric_values:
                raise ValueError(
                    f"Primary metric '{primary_metric}' not found in results"
                )

            sorted_models = sorted(
                metric_values.items(),
                key=lambda x: x[1],
                reverse=self._is_higher_better(primary_metric),
            )

            comparison_results["best_model"] = sorted_models[0][0]
            comparison_results["ranking"] = [
                (name, score) for name, score in sorted_models
            ]

            if statistical_test == "paired_ttest":
                comparison_results["statistical_tests"] = self._perform_paired_ttests(
                    model_results, primary_metric
                )

            comparison_results["summary"] = {
                "num_models": len(metric_values),
                "metric_range": max(metric_values.values())
                - min(metric_values.values()),
                "best_score": sorted_models[0][1],
                "worst_score": sorted_models[-1][1],
            }

            self.logger.info(
                f"Model comparison completed. Best model: {comparison_results['best_model']}"
            )

        except Exception as e:
            self.logger.error(f"Error in model comparison: {e}")
            raise

        return comparison_results

    def _is_binary_classification(self, df: DataFrame, label_col: str) -> bool:
        """
        Check if this is a binary classification problem
        """

        distinct_labels = df.select(label_col).distinct().count()
        return distinct_labels == 2

    def _calculate_log_loss(
        self, df: DataFrame, label_col: str, probability_col: str
    ) -> float:
        """
        Calculate log loss for binary classification
        """

        df_with_prob = df.withColumn("prob_positive", col(probability_col).getItem(1))

        log_loss_df = df_with_prob.withColumn(
            "log_loss_component",
            -(
                col(label_col) * log(col("prob_positive"))
                + (1 - col(label_col)) * log(1 - col("prob_positive"))
            ),
        )

        avg_log_loss = log_loss_df.agg(avg("log_loss_component")).collect()[0][0]
        return float(avg_log_loss)

    def _calculate_per_class_metrics(
        self, df: DataFrame, label_col: str, prediction_col: str
    ) -> Dict[str, Any]:
        """
        Calculate per-class precision, recall, and F1 scores
        """

        prediction_and_labels = df.select(prediction_col, label_col).rdd.map(
            lambda row: (float(row[0]), float(row[1]))
        )

        metrics = MulticlassMetrics(prediction_and_labels)
        labels = df.select(label_col).distinct().rdd.map(lambda row: row[0]).collect()
        labels = sorted([float(label) for label in labels])

        per_class_metrics = {}
        for label in labels:
            per_class_metrics[f"precision_class_{int(label)}"] = metrics.precision(
                label
            )

            per_class_metrics[f"recall_class_{int(label)}"] = metrics.recall(label)
            per_class_metrics[f"f1_class_{int(label)}"] = metrics.fMeasure(label)

        return per_class_metrics

    def _calculate_confusion_matrix(
        self, df: DataFrame, label_col: str, prediction_col: str
    ) -> List[List[int]]:
        """
        Calculate confusion matrix
        """

        confusion_df = df.crosstab(label_col, prediction_col)
        confusion_matrix = []
        pandas_df = confusion_df.toPandas()

        for _, row in pandas_df.iterrows():
            confusion_matrix.append(row[1:].tolist())

        return confusion_matrix

    def _calculate_explained_variance(
        self, df: DataFrame, label_col: str, prediction_col: str
    ) -> float:
        """
        Calculate explained variance score
        """

        label_mean = df.agg(avg(label_col)).collect()[0][0]
        df_with_vars = df.withColumn(
            "label_var", pow(col(label_col) - label_mean, 2)
        ).withColumn("residual_var", pow(col(label_col) - col(prediction_col), 2))

        total_var = df_with_vars.agg(avg("label_var")).collect()[0][0]
        residual_var = df_with_vars.agg(avg("residual_var")).collect()[0][0]

        explained_variance = 1 - (residual_var / total_var) if total_var > 0 else 0
        return float(explained_variance)

    def _calculate_mape(
        self, df: DataFrame, label_col: str, prediction_col: str
    ) -> float:
        """
        Calculate Mean Absolute Percentage Error
        """

        mape_df = df.withColumn(
            "ape",
            when(
                col(label_col) != 0,
                abs(col(label_col) - col(prediction_col)) / abs(col(label_col)) * 100,
            ).otherwise(0),
        )

        mape = mape_df.agg(avg("ape")).collect()[0][0]
        return float(mape)

    def _calculate_residual_stats(
        self, df: DataFrame, label_col: str, prediction_col: str
    ) -> Dict[str, float]:
        """
        Calculate residual statistics
        """

        residual_df = df.withColumn("residual", col(label_col) - col(prediction_col))

        stats_df = residual_df.agg(
            avg("residual").alias("mean_residual"),
            stddev("residual").alias("std_residual"),
            min("residual").alias("min_residual"),
            max("residual").alias("max_residual"),
        ).collect()[0]

        return {
            "mean_residual": float(stats_df["mean_residual"]),
            "std_residual": float(stats_df["std_residual"]),
            "min_residual": float(stats_df["min_residual"]),
            "max_residual": float(stats_df["max_residual"]),
        }

    def _is_higher_better(self, metric: str) -> bool:
        """
        Determine if higher values are better for a metric
        """

        higher_better_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "r2",
            "explained_variance",
        ]

        lower_better_metrics = ["rmse", "mae", "log_loss", "mape"]
        if metric in higher_better_metrics:
            return True

        elif metric in lower_better_metrics:
            return False

        else:
            return True

    def _perform_paired_ttests(
        self, model_results: Dict[str, Dict[str, float]], metric: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform paired t-tests between models
        """

        test_results = {}
        model_names = list(model_results.keys())

        for i, model1 in enumerate(model_names):
            test_results[model1] = {}
            for model2 in model_names[i + 1 :]:
                scores1 = model_results[model1].get("cv_scores", [])
                scores2 = model_results[model2].get("cv_scores", [])

                if len(scores1) > 0 and len(scores2) > 0:
                    t_stat, p_value = stats.ttest_rel(scores1, scores2)
                    test_results[model1][model2] = {
                        "statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05,
                        "mean_diff": np.mean(scores1) - np.mean(scores2),
                    }

                else:
                    score1 = model_results[model1].get(metric, 0)
                    score2 = model_results[model2].get(metric, 0)
                    test_results[model1][model2] = {
                        "statistic": abs(score1 - score2),
                        "p_value": None,
                        "significant": False,
                        "mean_diff": score1 - score2,
                    }

        return test_results

    def generate_evaluation_report(
        self,
        model_results: Dict[str, Dict[str, Any]],
        problem_type: str,
        output_path: str = None,
    ) -> str:
        """
        Generate comprehensive evaluation report.

        Args:
            model_results: Dictionary of model evaluation results
            problem_type: 'classification' or 'regression'
            output_path: Path to save report

        Returns:
            Report as string
        """

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SPARKFORGE MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Problem Type: {problem_type.upper()}")
        report_lines.append(f"Number of Models: {len(model_results)}")
        report_lines.append("")

        if problem_type == "classification":
            primary_metric = "f1"

        else:
            primary_metric = "rmse"

        comparison = self.compare_models(model_results, primary_metric)
        report_lines.append("MODEL RANKING")
        report_lines.append("-" * 40)

        for rank, (model_name, score) in enumerate(comparison["ranking"], 1):
            report_lines.append(f"{rank}. {model_name}: {score:.4f}")

        report_lines.append("")
        report_lines.append("DETAILED RESULTS")
        report_lines.append("-" * 40)

        for model_name, results in model_results.items():
            report_lines.append(f"\n{model_name.upper()}")
            report_lines.append("=" * len(model_name))

            main_metrics = (
                ["accuracy", "f1", "precision", "recall"]
                if problem_type == "classification"
                else ["rmse", "mae", "r2"]
            )

            for metric in main_metrics:
                if metric in results:
                    report_lines.append(f"{metric.capitalize()}: {results[metric]:.4f}")

            other_metrics = [
                k
                for k in results.keys()
                if k not in main_metrics and not k.startswith("confusion")
            ]

            if other_metrics:
                report_lines.append("\nAdditional Metrics:")
                for metric in sorted(other_metrics):
                    if isinstance(results[metric], (int, float)):
                        report_lines.append(f"  {metric}: {results[metric]:.4f}")

        report_lines.append("\n" + "=" * 40)
        report_lines.append("BEST MODEL SUMMARY")
        report_lines.append("=" * 40)
        best_model = comparison["best_model"]
        report_lines.append(f"Best Model: {best_model}")
        report_lines.append(
            f"Best {primary_metric}: {comparison['summary']['best_score']:.4f}"
        )

        report = "\n".join(report_lines)
        if output_path:
            try:
                with open(output_path, "w") as f:
                    f.write(report)

                self.logger.info(f"Evaluation report saved to: {output_path}")

            except Exception as e:
                self.logger.error(f"Error saving report: {e}")

        return report


class CrossValidationEvaluator:
    """
    Cross-validation evaluation utilities for robust model assessment.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = logging.getLogger(__name__)

    def cross_validate_model(
        self,
        estimator,
        dataset: DataFrame,
        evaluator,
        param_grid=None,
        num_folds: int = 5,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Perform cross-validation for a model.

        Args:
            estimator: ML estimator to evaluate
            dataset: Dataset for cross-validation
            evaluator: Evaluator for the model
            param_grid: Parameter grid for hyperparameter tuning
            num_folds: Number of cross-validation folds
            seed: Random seed

        Returns:
            Cross-validation results
        """

        from pyspark.ml.tuning import CrossValidator

        try:
            cv = CrossValidator(
                estimator=estimator,
                estimatorParamMaps=param_grid or [{}],
                evaluator=evaluator,
                numFolds=num_folds,
                seed=seed,
            )

            cv_model = cv.fit(dataset)
            results = {
                "best_model": cv_model.bestModel,
                "best_params": cv_model.getEstimatorParamMaps()[
                    np.argmax(cv_model.avgMetrics)
                ],
                "avg_metrics": cv_model.avgMetrics,
                "std_metrics": (
                    cv_model.stdMetrics if hasattr(cv_model, "stdMetrics") else None
                ),
                "best_metric": max(cv_model.avgMetrics),
                "worst_metric": min(cv_model.avgMetrics),
            }

            self.logger.info(
                f"Cross-validation completed. Best metric: {results['best_metric']:.4f}"
            )

            return results

        except Exception as e:
            self.logger.error(f"Error in cross-validation: {e}")
            raise


class FeatureImportanceAnalyzer:
    """
    Analyze and visualize feature importance from trained models.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = logging.getLogger(__name__)

    def extract_feature_importance(
        self, model, feature_names: List[str] = None
    ) -> Dict[str, float]:
        """
        Extract feature importance from trained model.

        Args:
            model: Trained ML model
            feature_names: Names of features

        Returns:
            Dictionary of feature importance scores
        """

        try:
            importance_dict = {}

            if hasattr(model, "featureImportances"):
                importances = model.featureImportances.toArray()

                if feature_names and len(feature_names) == len(importances):
                    importance_dict = dict(zip(feature_names, importances))

                else:
                    importance_dict = {
                        f"feature_{i}": imp for i, imp in enumerate(importances)
                    }

            elif hasattr(model, "coefficients"):
                coefficients = model.coefficients.toArray()
                if feature_names and len(feature_names) == len(coefficients):
                    importance_dict = dict(zip(feature_names, np.abs(coefficients)))

                else:
                    importance_dict = {
                        f"feature_{i}": abs(coef) for i, coef in enumerate(coefficients)
                    }

            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )

            self.logger.info(
                f"Extracted importance for {len(importance_dict)} features"
            )

            return importance_dict

        except Exception as e:
            self.logger.error(f"Error extracting feature importance: {e}")
            return {}

    def get_top_features(
        self, importance_dict: Dict[str, float], top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Get top k most important features.

        Args:
            importance_dict: Feature importance dictionary
            top_k: Number of top features to return

        Returns:
            List of (feature_name, importance_score) tuples
        """

        sorted_features = sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_features[:top_k]

    def analyze_feature_groups(
        self, importance_dict: Dict[str, float], group_patterns: Dict[str, str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze feature importance by groups (e.g., by feature type).

        Args:
            importance_dict: Feature importance dictionary
            group_patterns: Dictionary mapping group names to regex patterns

        Returns:
            Analysis by feature groups
        """

        import re

        if group_patterns is None:
            group_patterns = {
                "text": r"text_.*",
                "time_series": r"(rolling_|lag_|trend_|seasonal_).*",
                "categorical": r".*_encoded.*",
                "numerical": r".*",
            }

        group_analysis = {}
        for group_name, pattern in group_patterns.items():
            group_features = []
            group_importance = 0

            for feature_name, importance in importance_dict.items():
                if re.match(pattern, feature_name):
                    group_features.append((feature_name, importance))
                    group_importance += importance

            if group_features:
                group_analysis[group_name] = {
                    "total_importance": group_importance,
                    "avg_importance": group_importance / len(group_features),
                    "feature_count": len(group_features),
                    "top_features": sorted(
                        group_features, key=lambda x: x[1], reverse=True
                    )[:5],
                }

        return group_analysis


class ModelDriftDetector:
    """
    Detect model drift and data drift over time.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = logging.getLogger(__name__)

    def detect_prediction_drift(
        self,
        reference_predictions: DataFrame,
        current_predictions: DataFrame,
        prediction_col: str = "prediction",
        threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Detect drift in prediction distributions.

        Args:
            reference_predictions (DataFrame): Reference prediction dataset
            current_predictions (DataFrame): Current prediction dataset
            prediction_col (str): Prediction column name
            threshold (float): Drift detection threshold

        Returns:
          Drift detection results

        Raises:
          Exception: If there's an error during drift detection
        """

        try:
            ref_stats = reference_predictions.agg(
                avg(prediction_col).alias("mean"),
                stddev(prediction_col).alias("std"),
                min(prediction_col).alias("min"),
                max(prediction_col).alias("max"),
            ).collect()[0]

            curr_stats = current_predictions.agg(
                avg(prediction_col).alias("mean"),
                stddev(prediction_col).alias("std"),
                min(prediction_col).alias("min"),
                max(prediction_col).alias("max"),
            ).collect()[0]

            mean_drift = (
                abs(curr_stats["mean"] - ref_stats["mean"]) / ref_stats["std"]
                if ref_stats["std"] > 0
                else 0
            )

            std_drift = (
                abs(curr_stats["std"] - ref_stats["std"]) / ref_stats["std"]
                if ref_stats["std"] > 0
                else 0
            )

            drift_detected = mean_drift > threshold or std_drift > threshold
            results = {
                "drift_detected": drift_detected,
                "mean_drift": mean_drift,
                "std_drift": std_drift,
                "threshold": threshold,
                "reference_stats": dict(ref_stats.asDict()),
                "current_stats": dict(curr_stats.asDict()),
            }

            self.logger.info(
                f"Prediction drift analysis completed. Drift detected: {drift_detected}"
            )

            return results

        except Exception as e:
            self.logger.error(f"Error in drift detection: {e}")
            raise

    def detect_feature_drift(
        self,
        reference_data: DataFrame,
        current_data: DataFrame,
        feature_cols: List[str],
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Detect drift in feature distributions.

        Args:
          reference_data (DataFrame): Reference dataset
          current_data (DataFrame): Current dataset
          feature_cols (list): List of feature columns to check
          threshold (float): Drift detection threshold

        Returns:
          Feature drift results

        Raises:
          Exception: If there's an error during drift detection
        """

        drift_results = {}

        try:
            for feature in feature_cols:
                ref_values = (
                    reference_data.select(feature).rdd.map(lambda x: x[0]).collect()
                )
                curr_values = (
                    current_data.select(feature).rdd.map(lambda x: x[0]).collect()
                )

                ref_mean = np.mean(ref_values)
                curr_mean = np.mean(curr_values)
                ref_std = np.std(ref_values)

                if ref_std > 0:
                    drift_score = abs(curr_mean - ref_mean) / ref_std
                    drift_detected = drift_score > threshold

                else:
                    drift_score = 0
                    drift_detected = False

                drift_results[feature] = {
                    "drift_detected": drift_detected,
                    "drift_score": drift_score,
                    "reference_mean": ref_mean,
                    "current_mean": curr_mean,
                }

            overall_drift = any(
                result["drift_detected"] for result in drift_results.values()
            )

            return {
                "overall_drift_detected": overall_drift,
                "feature_results": drift_results,
                "features_with_drift": [
                    f for f, r in drift_results.items() if r["drift_detected"]
                ],
            }

        except Exception as e:
            self.logger.error(f"Error in feature drift detection: {e}")
            raise

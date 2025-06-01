# sparkforge/utils/__init__.py

"""
SparkForge Utilities Module

This module provides comprehensive utilities for data processing, evaluation,
and visualization within the SparkForge framework.
"""

from .data_utils import SparkDataUtils, DataValidationError

from .evaluation import (
    ModelEvaluator,
    CrossValidationEvaluator,
    FeatureImportanceAnalyzer,
    ModelDriftDetector,
)

from .visualization import (
    SparkForgeVisualizer,
    quick_model_comparison,
    quick_feature_importance,
    quick_confusion_matrix,
)

import logging
from pyspark.sql import SparkSession
from typing import Optional, Dict, Any


class SparkForgeUtils:
    """
    Unified utilities class providing easy access to all SparkForge utility functions.

    This class serves as a central hub for accessing data processing, evaluation,
    and visualization utilities with consistent configuration and logging.
    """

    def __init__(self, spark: SparkSession, config: Dict[str, Any] = None):
        """
        Initialize SparkForge utilities.

        Args:
            spark: SparkSession instance
            config: Configuration dictionary
        """
        self.spark = spark
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        self.data = SparkDataUtils(spark)
        self.evaluator = ModelEvaluator(spark)
        self.cv_evaluator = CrossValidationEvaluator(spark)
        self.feature_analyzer = FeatureImportanceAnalyzer(spark)
        self.drift_detector = ModelDriftDetector(spark)
        self.visualizer = SparkForgeVisualizer()

        self.logger.info("SparkForge utilities initialized")

    def validate_and_prepare_data(self, df, validation_config: Dict[str, Any] = None):
        """
        Comprehensive data validation and preparation.

        Args:
            df: DataFrame to validate and prepare
            validation_config: Validation configuration

        Returns:
            Tuple of (validated_df, validation_results)
        """

        validation_config = validation_config or self.config.get("data", {}).get(
            "validation", {}
        )

        validation_results = self.data.validate_data(df, validation_config)
        if not validation_results["is_valid"]:
            raise DataValidationError(
                f"Data validation failed: {validation_results['errors']}"
            )

        clean_df = self.data.clean_column_names(df)
        self.logger.info(
            f"Data validation completed. Shape: {validation_results['statistics']['row_count']} x {validation_results['statistics']['column_count']}"
        )

        if validation_results["warnings"]:
            self.logger.warning(
                f"Data validation warnings: {validation_results['warnings']}"
            )

        return clean_df, validation_results

    def comprehensive_model_evaluation(
        self,
        predictions_df,
        problem_type: str,
        model_name: str = "model",
        generate_plots: bool = True,
        output_dir: str = "results",
    ) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation with automatic plotting.

        Args:
            predictions_df: DataFrame with model predictions
            problem_type: 'classification' or 'regression'
            model_name: Name of the model
            generate_plots: Whether to generate visualization plots
            output_dir: Output directory for plots and reports

        Returns:
            Comprehensive evaluation results
        """

        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        if problem_type == "classification":
            results = self.evaluator.evaluate_classification(predictions_df)

        else:
            results = self.evaluator.evaluate_regression(predictions_df)

        if generate_plots:
            plot_dir = output_path / "plots"
            plot_dir.mkdir(exist_ok=True)

            try:
                if problem_type == "classification":
                    if "confusion_matrix" in results:
                        self.visualizer.plot_confusion_matrix(
                            results["confusion_matrix"],
                            title=f"{model_name} - Confusion Matrix",
                            save_path=str(
                                plot_dir / f"{model_name}_confusion_matrix.png"
                            ),
                        )

                    if "auc" in results:
                        self.visualizer.plot_roc_curve(
                            predictions_df,
                            title=f"{model_name} - ROC Curve",
                            save_path=str(plot_dir / f"{model_name}_roc_curve.png"),
                        )

                        self.visualizer.plot_precision_recall_curve(
                            predictions_df,
                            title=f"{model_name} - Precision-Recall Curve",
                            save_path=str(plot_dir / f"{model_name}_pr_curve.png"),
                        )

                else:  
                    self.visualizer.plot_residuals(
                        predictions_df,
                        title=f"{model_name} - Residual Analysis",
                        save_path=str(plot_dir / f"{model_name}_residuals.png"),
                    )

            except Exception as e:
                self.logger.warning(f"Error generating plots: {e}")

        report = self.evaluator.generate_evaluation_report(
            {model_name: results},
            problem_type,
            output_path=str(output_path / f"{model_name}_evaluation_report.txt"),
        )

        results["evaluation_report"] = report
        results["output_directory"] = str(output_path)

        return results

    def analyze_feature_importance_comprehensive(
        self,
        model,
        feature_names: list,
        generate_plots: bool = True,
        output_dir: str = "results",
    ) -> Dict[str, Any]:
        """
        Comprehensive feature importance analysis with visualizations.

        Args:
            model: Trained model
            feature_names: List of feature names
            generate_plots: Whether to generate plots
            output_dir: Output directory

        Returns:
            Feature importance analysis results
        """

        from pathlib import Path

        importance_dict = self.feature_analyzer.extract_feature_importance(
            model, feature_names
        )

        if not importance_dict:
            self.logger.warning(
                "No feature importance could be extracted from the model"
            )
            return {}

        top_features = self.feature_analyzer.get_top_features(importance_dict, top_k=20)

        group_analysis = self.feature_analyzer.analyze_feature_groups(importance_dict)
        results = {
            "feature_importance": importance_dict,
            "top_features": top_features,
            "group_analysis": group_analysis,
            "total_features": len(importance_dict),
        }

        if generate_plots:
            output_path = Path(output_dir)
            plot_dir = output_path / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)

            try:
                self.visualizer.plot_feature_importance(
                    importance_dict,
                    top_k=20,
                    title="Top 20 Feature Importance",
                    save_path=str(plot_dir / "feature_importance.png"),
                )

                if group_analysis:
                    self.visualizer.plot_feature_group_importance(
                        group_analysis,
                        title="Feature Importance by Groups",
                        save_path=str(plot_dir / "feature_group_importance.png"),
                    )

            except Exception as e:
                self.logger.warning(f"Error generating feature importance plots: {e}")

        return results

    def monitor_model_drift(
        self,
        reference_data,
        current_data,
        model=None,
        feature_cols: list = None,
        prediction_col: str = "prediction",
    ) -> Dict[str, Any]:
        """
        Comprehensive model and data drift monitoring.

        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            model: Trained model (optional)
            feature_cols: Feature columns to monitor
            prediction_col: Prediction column name

        Returns:
            Drift monitoring results
        """

        drift_results = {}

        try:
            if feature_cols:
                feature_drift = self.drift_detector.detect_feature_drift(
                    reference_data, current_data, feature_cols
                )
                drift_results["feature_drift"] = feature_drift

            if model is not None:
                ref_predictions = model.transform(reference_data)
                curr_predictions = model.transform(current_data)

                prediction_drift = self.drift_detector.detect_prediction_drift(
                    ref_predictions, curr_predictions, prediction_col
                )
                drift_results["prediction_drift"] = prediction_drift

            overall_drift = any(
                [
                    drift_results.get("feature_drift", {}).get(
                        "overall_drift_detected", False
                    ),
                    drift_results.get("prediction_drift", {}).get(
                        "drift_detected", False
                    ),
                ]
            )

            drift_results["overall_drift_detected"] = overall_drift

            self.logger.info(
                f"Drift monitoring completed. Overall drift detected: {overall_drift}"
            )

        except Exception as e:
            self.logger.error(f"Error in drift monitoring: {e}")
            drift_results["error"] = str(e)

        return drift_results

    def create_model_dashboard(
        self,
        model_results: Dict[str, Any],
        feature_importance: Dict[str, float] = None,
        predictions_df=None,
        model_name: str = "Model",
        save_path: str = None,
    ):
        """
        Create a comprehensive model performance dashboard.

        Args:
            model_results: Model evaluation results
            feature_importance: Feature importance scores
            predictions_df: Predictions DataFrame
            model_name: Name of the model
            save_path: Path to save dashboard

        Returns:
            Dashboard figure
        """

        try:
            dashboard = self.visualizer.create_model_dashboard(
                model_results=model_results,
                feature_importance=feature_importance,
                predictions_df=predictions_df,
                save_path=save_path,
            )

            self.logger.info(f"Model dashboard created for {model_name}")

            if save_path:
                self.logger.info(f"Dashboard saved to: {save_path}")

            return dashboard

        except Exception as e:
            self.logger.error(f"Error creating model dashboard: {e}")
            return None

    def quick_data_profile(
        self,
        df,
        sample_size: int = 10000,
        generate_plots: bool = True,
        output_dir: str = "data_profile",
    ) -> Dict[str, Any]:
        """
        Quick data profiling with summary statistics and visualizations.

        Args:
            df: DataFrame to profile
            sample_size: Sample size for analysis
            generate_plots: Whether to generate plots
            output_dir: Output directory

        Returns:
            Data profile results
        """

        from pathlib import Path

        try:
            summary = self.data.get_data_summary(df)
            missing_analysis = self.data.analyze_missing_values(df)
            numeric_cols = summary["column_info"]["numeric_columns"]
            if numeric_cols:
                outlier_df = self.data.detect_outliers(
                    df, numeric_cols[:5]
                )

                outlier_summary = {}
                for col in numeric_cols[:5]:
                    outlier_col = f"{col}_outlier"

                    if outlier_col in outlier_df.columns:
                        outlier_count = outlier_df.filter(col(outlier_col) == 1).count()
                        outlier_summary[col] = {
                            "outlier_count": outlier_count,
                            "outlier_percentage": (outlier_count / df.count()) * 100,
                        }

                summary["outlier_analysis"] = outlier_summary

            if generate_plots:
                output_path = Path(output_dir)
                plot_dir = output_path / "plots"
                plot_dir.mkdir(parents=True, exist_ok=True)

                all_cols = numeric_cols[:6] if len(numeric_cols) > 6 else numeric_cols
                if all_cols:
                    self.visualizer.plot_data_distribution(
                        df,
                        all_cols,
                        sample_size=sample_size,
                        title="Data Distributions",
                        save_path=str(plot_dir / "data_distributions.png"),
                    )

                if len(numeric_cols) > 1:
                    self.visualizer.plot_correlation_matrix(
                        df,
                        numeric_cols[:10],
                        sample_size=sample_size,
                        title="Feature Correlation Matrix",
                        save_path=str(plot_dir / "correlation_matrix.png"),
                    )

            self.logger.info("Data profiling completed successfully")

            return summary

        except Exception as e:
            self.logger.error(f"Error in data profiling: {e}")
            return {"error": str(e)}

    def get_spark_optimization_recommendations(self, df) -> Dict[str, Any]:
        """
        Analyze DataFrame and provide Spark optimization recommendations.

        Args:
            df: DataFrame to analyze

        Returns:
            Optimization recommendations
        """

        recommendations = {
            "caching": [],
            "partitioning": [],
            "broadcasting": [],
            "general": [],
        }

        try:
            row_count = df.count()
            col_count = len(df.columns)

            if row_count > 100000:
                recommendations["caching"].append(
                    "Consider caching this DataFrame as it's large and likely to be reused"
                )

            current_partitions = df.rdd.getNumPartitions()
            optimal_partitions = max(row_count // 50000, 1)

            if current_partitions < optimal_partitions:
                recommendations["partitioning"].append(
                    f"Consider increasing partitions from {current_partitions} to {optimal_partitions} "
                    f"for better parallelism"
                )
            elif current_partitions > optimal_partitions * 2:
                recommendations["partitioning"].append(
                    f"Consider reducing partitions from {current_partitions} to {optimal_partitions} "
                    f"to reduce overhead"
                )

            if row_count < 10000:
                recommendations["broadcasting"].append(
                    "This DataFrame is small enough to consider for broadcasting in joins"
                )

            if col_count > 100:
                recommendations["general"].append(
                    "DataFrame has many columns - consider selecting only needed columns early"
                )

            if row_count > 1000000:
                recommendations["general"].append(
                    "Large DataFrame detected - ensure proper file format (Parquet) and compression"
                )

            self.logger.info("Spark optimization analysis completed")

        except Exception as e:
            self.logger.error(f"Error in optimization analysis: {e}")
            recommendations["error"] = str(e)

        return recommendations


# Convenience functions for common workflows
def quick_setup(spark: SparkSession, config_path: str = None) -> SparkForgeUtils:
    """
    Quick setup of SparkForge utilities with default configuration.

    Args:
        spark: SparkSession instance
        config_path: Path to configuration file

    Returns:
        Configured SparkForgeUtils instance
    """

    from sparkforge.config import get_config

    config = get_config(config_path) if config_path else get_config()
    return SparkForgeUtils(spark, config.to_dict())


def setup_logging(level: str = "INFO", format_str: str = None):
    """
    Setup logging for SparkForge utilities.

    Args:
        level: Logging level
        format_str: Log format string
    """

    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("pyspark").setLevel(logging.WARNING)


__all__ = [
    "SparkForgeUtils",
    "SparkDataUtils",
    "ModelEvaluator",
    "CrossValidationEvaluator",
    "FeatureImportanceAnalyzer",
    "ModelDriftDetector",
    "SparkForgeVisualizer",
    "DataValidationError",
    "quick_model_comparison",
    "quick_feature_importance",
    "quick_confusion_matrix",
    "quick_setup",
    "setup_logging",
]

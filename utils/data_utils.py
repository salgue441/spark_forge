# sparkforge/utils/data_utils.py

"""
Data utilities for SparkForge framework.

This module provides utilities for data loading, preprocessing, validation,
and manipulation specific to Spark DataFrames.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import numpy as np


class DataValidationError(Exception):
    """
    Custom exception for data validation errors
    """

    pass


class SparkDataUtils:
    """
    Comprehensive data utilities for Spark DataFrames.

    Provides methods for data loading, validation, preprocessing,
    and quality assessment.
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize data utilities

        Args:
          spark (SparkSession): SparkSession Instance
        """

        self.spark = spark
        self.logger = logging.getLogger(__name__)

    def load_data(
        self,
        path: str,
        format: str = "parquet",
        options: Dict[str, Any] = None,
        schema: StructType = None,
    ) -> DataFrame:
        """
        Load data from various formats.

        Args:
          path (str): Path to data file/directory
          format (str): Data format (parquet, csv, json, delta, etc)
          options (dict): Additional options for data loading
          schema (optional, StructType): Optional schema for the data

        Returns:
          Loaded DataFrame

        Raises:
          Exception: If there's an I/O when loading the data
        """

        options = options or {}
        try:
            reader = self.spark.read.format(format)

            if schema:
                reader = reader.schema(schema)

            for key, value in options.items():
                reader = reader.option(key, value)

            df = reader.load(path)

            self.logger.info(f"Successfully loaded data from {path}")
            self.logger.info(
                f"Data shape: {df.count()} rows, {len(df.columns)} columns"
            )

            return df

        except Exception as e:
            self.logger.error(f"Error loading data from {path}: {e}")
            raise

    def validate_data(
        self, df: DataFrame, config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Validate DataFrame for quality and consistency

        Args:
          df (DataFrame): DataFrame to validate
          config (Dict[str, Any]): Validation configuration

        Returns:
          Dictionary with validation results
        """

        config = config = {}
        max_missing_ratio = config.get("max_missing_ratio", 0.3)
        min_rows = config.get("min_rows", 100)
        check_duplicates = config.get("check_duplicates", True)

        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "statistics": [],
        }

        try:
            row_count = df.count()
            col_count = len(df.columns)

            validation_results["statistics"]["row_count"] = row_count
            validation_results["statistics"]["column_count"] = col_count

            if row_count < min_rows:
                validation_results["errors"].append(
                    f"Insufficient data: {row_count} rows < {min_rows} minimum"
                )

                validation_results["is_valid"] = False

            missing_stats = self.analyze_missing_values(df)
            validation_results["statistics"]["missing_values"] = missing_stats
            high_missing_cols = [
                col
                for col, ratio in missing_stats["missing_ratios"].items()
                if ratio > max_missing_ratio
            ]

            if high_missing_cols:
                validation_results["warnings"].append(
                    f"High missing values in columns: {high_missing_cols}"
                )

            if check_duplicates:
                duplicate_count = row_count - df.distinct().count()
                validation_results["statistics"]["duplicate_count"] = duplicate_count

                if duplicate_count > 0:
                    validation_results["warnings"].append(
                        f"Found {duplicate_count} duplicate rows"
                    )

            type_stats = self.analyze_data_types(df)
            validation_results["statistics"]["data_types"] = type_stats

            schema_issues = self.validate_schema(df)
            if schema_issues:
                validation_results["warnings"].extend(schema_issues)

            self.logger.info("Data validation completed")

        except Exception as e:
            validation_results["errors"].append(f"Validation error: {e}")
            validation_results["is_valid"] = False

            self.logger.error(f"Error during data validation: {e}")

        return validation_results

    def analyze_missing_values(self, df: DataFrame) -> Dict[str, Any]:
        """
        Analyze missing values in DataFrame.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with missing value statistics
        """

        total_rows = df.count()
        missing_counts = {}
        missing_ratios = {}

        for col_name in df.columns:
            missing_count = df.filter(col(col_name).isNull()).count()
            missing_counts[col_name] = missing_count
            missing_ratios[col_name] = (
                missing_count / total_rows if total_rows > 0 else 0
            )

        total_missing = sum(missing_counts.values())
        total_cells = total_rows * len(df.columns)
        overall_missing_ratio = total_missing / total_cells if total_cells > 0 else 0

        return {
            "missing_counts": missing_counts,
            "missing_ratios": missing_ratios,
            "total_missing": total_missing,
            "overall_missing_ratio": overall_missing_ratio,
            "columns_with_missing": [
                col for col, count in missing_counts.items() if count > 0
            ],
        }

    def analyze_data_types(self, df: DataFrame) -> Dict[str, Any]:
        """
        Analyze data types in DataFrame.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with data type statistics
        """

        type_counts = {}
        columns_by_type = {}

        for field in df.schema.fields:
            type_name = str(field.dataType)

            if type_name not in type_counts:
                type_counts[type_name] = 0
                columns_by_type[type_name] = []

            type_counts[type_name] += 1
            columns_by_type[type_name].append(field.name)

        numeric_types = [
            "IntegerType",
            "LongType",
            "FloatType",
            "DoubleType",
            "DecimalType",
        ]

        string_types = ["StringType"]
        datetime_types = ["TimestampType", "DateType"]
        numeric_columns = []
        string_columns = []
        datetime_columns = []
        other_columns = []

        for field in df.schema.fields:
            type_name = field.dataType.typeName()

            if any(nt in str(field.dataType) for nt in numeric_types):
                numeric_columns.append(field.name)

            elif any(st in str(field.dataType) for st in string_types):
                string_columns.append(field.name)

            elif any(dt in str(field.dataType) for dt in datetime_types):
                datetime_columns.append(field.name)

            else:
                other_columns.append(field.name)

        return {
            "type_counts": type_counts,
            "columns_by_type": columns_by_type,
            "numeric_columns": numeric_columns,
            "string_columns": string_columns,
            "datetime_columns": datetime_columns,
            "other_columns": other_columns,
        }

    def validate_schema(self, df: DataFrame) -> List[str]:
        """
        Validate DataFrame schema for common issues.

        Args:
            df: DataFrame to validate

        Returns:
            List of schema issues
        """

        issues = []
        col_names = df.columns
        if len(col_names) != len(set(col_names)):
            issues.append("Duplicate column names found")

        problematic_chars = [" ", ".", "-", "/", "\\", "(", ")", "[", "]"]
        problematic_cols = []

        for col_name in col_names:
            if any(char in col_name for char in problematic_chars):
                problematic_cols.append(col_name)

        if problematic_cols:
            issues.append(f"Problematic column names: {problematic_cols}")

        empty_cols = [col for col in col_names if not col or col.strip() == ""]
        if empty_cols:
            issues.append("Empty column names found")

        return issues

    def clean_column_names(self, df: DataFrame) -> DataFrame:
        """
        Clean column names to be Spark-friendly.

        Args:
            df: DataFrame with potentially problematic column names

        Returns:
            DataFrame with cleaned column names
        """

        cleaned_columns = {}
        for col_name in df.columns:
            clean_name = col_name.strip()
            clean_name = clean_name.replace(" ", "_")
            clean_name = clean_name.replace(".", "_")
            clean_name = clean_name.replace("-", "_")
            clean_name = clean_name.replace("/", "_")
            clean_name = clean_name.replace("\\", "_")
            clean_name = clean_name.replace("(", "_")
            clean_name = clean_name.replace(")", "_")
            clean_name = clean_name.replace("[", "_")
            clean_name = clean_name.replace("]", "_")

            while "__" in clean_name:
                clean_name = clean_name.replace("__", "_")

            clean_name = clean_name.strip("_")
            if not clean_name:
                clean_name = f"col_{df.columns.index(col_name)}"

            cleaned_columns[col_name] = clean_name

        for old_name, new_name in cleaned_columns.items():
            if old_name != new_name:
                df = df.withColumnRenamed(old_name, new_name)

        self.logger.info(
            f"Cleaned {len([1 for old, new in cleaned_columns.items() if old != new])} column names"
        )

        return df

    def split_data(
        self,
        df: DataFrame,
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_col: str = None,
        seed: int = 42,
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
          df (DataFrame): DataFrame to split
          train_ratio (float): Ratio for training set
          validation_ratio (float): Ratio for validation set
          test_ratio (float): Ratio for test set
          stratify_col (str): Column to stratify by (for classification)
          seed (int): Random seed for reproducibility

        Returns:
          Tuple of (train_df, validation_df, test_df)

        Raises:
          ValueErorr: If the ratios don't sum 1.0
        """

        total_ratio = train_ratio + validation_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

        if stratify_col and stratify_col in df.columns:
            return self._stratified_split(
                df, train_ratio, validation_ratio, test_ratio, stratify_col, seed
            )

        else:
            return self._random_split(
                df, train_ratio, validation_ratio, test_ratio, seed
            )

    def _random_split(
        self,
        df: DataFrame,
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Perform random data split

        Args:
          df (DataFrame): DataFrame to split
          train_ratio (float): Ratio for training set
          validation_ratio (float): Ratio for validation set
          test_ratio (float): Ratio for test set
          seed (int): Random seed for reproducibility

        Returns:
          Tuple of (train_df, validation_df, test_df)
        """

        train_val_ratio = train_ratio + validation_ratio
        train_val, test = df.randomSplit([train_val_ratio, test_ratio], seed=seed)

        adjusted_train_ratio = train_ratio / train_val_ratio
        train, validation = train_val.randomSplit(
            [adjusted_train_ratio, 1 - adjusted_train_ratio], seed=seed
        )

        self.logger.info(
            f"Data split completed - Train: {train.count()}, "
            f"Val: {validation.count()}, Test: {test.count()}"
        )

        return train, validation, test

    def _stratified_split(
        self,
        df: DataFrame,
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
        stratify_col: str,
        seed: int,
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Perform stratified data split

        Args:
          df (DataFrame): DataFrame to split
          train_ratio (float): Ratio for training set
          validation_ratio (float): Ratio for validation set
          test_ratio (float): Ratio for test set
          stratify_col (str): Column to stratify by (for classification)
          seed (int): Random seed for reproducibility

        Returns:
          Tuple of (train_df, validation_df, test_df)
        """

        class_counts = df.groupBy(stratify_col).count().collect()

        train_dfs, val_dfs, test_dfs = [], [], []
        for row in class_counts:
            class_value = row[stratify_col]
            class_df = df.filter(col(stratify_col) == class_value)

            train_val_ratio = train_ratio + validation_ratio
            class_train_val, class_test = class_df.randomSplit(
                [train_val_ratio, test_ratio], seed=seed
            )

            adjusted_train_ratio = train_ratio / train_val_ratio
            class_train, class_val = class_train_val.randomSplit(
                [adjusted_train_ratio, 1 - adjusted_train_ratio], seed=seed
            )

            train_dfs.append(class_train)
            val_dfs.append(class_val)
            test_dfs.append(class_test)

        # Combine all classes
        train = train_dfs[0]
        for df_part in train_dfs[1:]:
            train = train.union(df_part)

        validation = val_dfs[0]
        for df_part in val_dfs[1:]:
            validation = validation.union(df_part)

        test = test_dfs[0]
        for df_part in test_dfs[1:]:
            test = test.union(df_part)

        self.logger.info(
            f"Stratified split completed - Train: {train.count()}, "
            f"Val: {validation.count()}, Test: {test.count()}"
        )

        return train, validation, test

    def get_data_summary(self, df: DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive data summary.

        Args:
            df: DataFrame to summarize

        Returns:
            Dictionary with data summary
        """

        summary = {
            "basic_info": {},
            "column_info": {},
            "data_quality": {},
            "statistics": {},
        }

        row_count = df.count()
        col_count = len(df.columns)
        summary["basic_info"] = {
            "row_count": row_count,
            "column_count": col_count,
            "estimated_size_mb": self._estimate_dataframe_size(df),
        }

        type_analysis = self.analyze_data_types(df)
        summary["column_info"] = type_analysis

        missing_analysis = self.analyze_missing_values(df)
        summary["data_quality"] = {
            "missing_values": missing_analysis,
            "duplicate_count": row_count - df.distinct().count(),
        }

        numeric_cols = type_analysis["numeric_columns"]
        if numeric_cols:
            stats_df = df.select(numeric_cols).describe()
            summary["statistics"]["numeric_summary"] = self._convert_describe_to_dict(
                stats_df
            )

        return summary

    def _estimate_dataframe_size(self, df: DataFrame) -> float:
        """
        Estimate DataFrame size in MB

        Args:
          df (DataFrame): DataFrame to be estimated in size

        Returns:
          The amount of MB the dataset uses (estimation), or -1
          if the calculation failed
        """

        try:
            sample_size = min(1000, df.count())
            sample_df = df.limit(sample_size)

            pandas_df = sample_df.toPandas()
            sample_size_mb = pandas_df.memory_usage(deep=True).sum() / (1024 * 1024)

            full_size_mb = sample_size_mb * (df.count() / sample_size)
            return round(full_size_mb, 2)

        except:
            return -1

    def _convert_describe_to_dict(
        self, describe_df: DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Convert Spark describe() output to dictionary

        Args:
          describe_df (DataFrame): DataFrame to be converted

        Returns:
          Dict[str, Dict[str, float]]: Dictionary output of the
              DataFrame
        """

        result = {}
        pandas_describe = describe_df.toPandas()

        for col in pandas_describe.columns:
            if col != "summary":
                result[col] = {}

                for _, row in pandas_describe.iterrows():
                    stat_name = row["summary"]

                    try:
                        stat_value = float(row[col])
                        result[col][stat_name] = stat_value

                    except (ValueError, TypeError):
                        result[col][stat_name] = row[col]

        return result

    def detect_outliers(
        self,
        df: DataFrame,
        columns: List[str] = None,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> DataFrame:
        """
        Detect outliers in numeric columns.

        Args:
            df: DataFrame to analyze
            columns: Columns to check for outliers (None for all numeric)
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outlier flags
        """

        if columns is None:
            type_analysis = self.analyze_data_types(df)
            columns = type_analysis["numeric_columns"]

        result = df
        for col_name in columns:
            if method == "iqr":
                result = self._detect_outliers_iqr(result, col_name, threshold)

            elif method == "zscore":
                result = self._detect_outliers_zscore(result, col_name, threshold)

        return result

    def _detect_outliers_iqr(
        self, df: DataFrame, col_name: str, threshold: float
    ) -> DataFrame:
        """
        Detect outliers using IQR method

        Args:
          df (DataFrame): DataFrame to be used
          col_name (str): Column where to search the outliers
          threshold (float): Threshold for considering outliers

        Returns:
          The outliers in DataFrame format
        """

        quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
        q1, q3 = quantiles[0], quantiles[1]
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        result = df.withColumn(
            f"{col_name}_outlier",
            when(
                (col(col_name) < lower_bound) | (col(col_name) > upper_bound), 1
            ).otherwise(0),
        )

        return result

    def _detect_outliers_zscore(
        self, df: DataFrame, col_name: str, threshold: float
    ) -> DataFrame:
        """
        Detect outliers using Z-score method

        Args:
          df (DataFrame): DataFrame to be used
          col_name (str): Column where to search the outliers
          threshold (float): Threshold for considering outliers

        Returns:
          The outliers in DataFrame format
        """

        stats = df.agg(
            avg(col_name).alias("mean"), stddev(col_name).alias("std")
        ).collect()[0]

        mean_val = stats["mean"]
        std_val = stats["std"]

        if std_val is None or std_val == 0:
            return df.withColumn(f"{col_name}_outlier", lit(0))

        result = df.withColumn(
            f"{col_name}_zscore", abs(col(col_name) - mean_val) / std_val
        )

        result = result.withColumn(
            f"{col_name}_outlier",
            when(col(f"{col_name}_zscore") > threshold, 1).otherwise(0),
        )

        result = result.drop(f"{col_name}_zscore")
        return result

    def balance_dataset(
        self,
        df: DataFrame,
        target_col: str,
        method: str = "undersample",
        seed: int = 42,
    ) -> DataFrame:
        """
        Balance dataset for classification.

        Args:
            df (DataFrame): DataFrame to balance
            target_col (str): Target column name
            method (str): Balancing method ('undersample', 'oversample')
            seed (int): Random seed

        Returns:
            Balanced DataFrame
        """

        class_counts = df.groupBy(target_col).count().collect()
        class_dict = {row[target_col]: row["count"] for row in class_counts}

        if method == "undersample":
            min_count = min(class_dict.values())
            balanced_dfs = []

            for class_val in class_dict.keys():
                class_df = df.filter(col(target_col) == class_val)
                sampled_df = class_df.sample(
                    withReplacement=False,
                    fraction=min_count / class_dict[class_val],
                    seed=seed,
                )

                balanced_dfs.append(sampled_df)

        elif method == "oversample":
            max_count = max(class_dict.values())
            balanced_dfs = []

            for class_val in class_dict.keys():
                class_df = df.filter(col(target_col) == class_val)
                current_count = class_dict[class_val]

                if current_count < max_count:
                    ratio = max_count / current_count
                    sampled_df = class_df.sample(
                        withReplacement=True, fraction=ratio, seed=seed
                    )

                else:
                    sampled_df = class_df

                balanced_dfs.append(sampled_df)

        result = balanced_dfs[0]
        for df_part in balanced_dfs[1:]:
            result = result.union(df_part)

        self.logger.info(f"Dataset balanced using {method}. New size: {result.count()}")
        return result

    def save_data(
        self,
        df: DataFrame,
        path: str,
        format: str = "parquet",
        mode: str = "overwrite",
        options: Dict[str, Any] = None,
    ):
        """
        Save DataFrame to storage.

        Args:
          df (DataFrame): DataFrame to save
          path (str): Output path
          format (str): Output format
          mode (str): Write mode
          options (Dict[str, Any]): Additional options

        Raises:
          Exception: If there's an error when saving data to path
        """

        options = options or {}
        try:
            writer = df.write.format(format).mode(mode)
            for key, value in options.items():
                writer = writer.option(key, value)

            writer.save(path)
            self.logger.info(f"Data saved to {path} in {format} format")

        except Exception as e:
            self.logger.error(f"Error saving data to {path}: {e}")
            raise

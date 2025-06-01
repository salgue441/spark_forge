# SparkForge: Advanced PySpark ML Framework

A comprehensive, production-ready framework for advanced feature engineering and ensemble learning with Apache Spark.

## Features

### 🔧 Advanced Feature Engineering

- **Time Series Features**: Rolling statistics, lag features, seasonal decomposition
- **Text Features**: Advanced NLP features, readability metrics, sentiment analysis
- **Numerical Features**: Polynomial features, interactions, statistical transformations
- **Categorical Features**: Advanced encoding techniques, target encoding

### 🤖 Multi-Model Ensemble System

- **Parallel Training**: Train multiple algorithms simultaneously
- **Advanced Ensembling**: Stacking, blending, dynamic weighting
- **Hyperparameter Tuning**: Distributed grid search and cross-validation
- **Model Selection**: Automated model comparison and selection

### 🚀 Production Ready

- **Scalable**: Designed for large-scale distributed computing
- **Modular**: Clean, extensible architecture
- **Configurable**: YAML-based configuration system
- **Tested**: Comprehensive test suite

## Quick Start

```python
from sparkforge.core.pipelines import AdvancedFeatureEngineeringPipeline
from sparkforge.core.ensemble import EnsembleModelTrainer
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder.appName("SparkForge").getOrCreate()

# Load your data
df = spark.read.parquet("your_data.parquet")

# Build feature engineering pipeline
fe_pipeline = AdvancedFeatureEngineeringPipeline(spark)
pipeline = fe_pipeline.build_pipeline(
    numerical_cols=['feature1', 'feature2'],
    categorical_cols=['category1', 'category2'],
    text_cols=['text_field'],
    enable_pca=True,
    pca_k=50
)

# Train ensemble models
ensemble_trainer = EnsembleModelTrainer(spark)
ensemble_model = ensemble_trainer.train_ensemble(
    data=df,
    feature_pipeline=pipeline,
    target_col='target',
    problem_type='classification'
)

# Make predictions
predictions = ensemble_model.transform(test_df)
```

## Installation

```bash
pip install sparkforge
```

## Documentation

For detailed documentation and examples, see the `examples/` directory.

## Contributing

We welcome contributions! Please see our contributing guidelines.

## License

MIT License

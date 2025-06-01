from setuptools import setup, find_packages

setup(
    name="sparkforge",
    version="0.1.0",
    description="PySpark ML Framework for Feature Engineering and Ensemble Learning",
    author="Carlos Salguero",
    author_email="carlossalguero441@gmail.com",
    packages=find_packages(),
    install_requires=[
        "pyspark>=3.2.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pyyaml>=6.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0.0", "pytest-cov>=3.0.0", "flake8>=4.0.0", "black>=22.0.0"]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

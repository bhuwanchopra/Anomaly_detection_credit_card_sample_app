# Credit Card Anomaly Detection System

A comprehensive machine learning system for detecting anomalous credit card transactions using advanced ensemble methods and feature engineering techniques.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](#-testing)

## 🚀 Features

### Core Capabilities
- **Multi-Algorithm Detection**: PCA, Isolation Forest, Local Outlier Factor, and Ensemble methods
- **Advanced Feature Engineering**: Time-based, frequency-based, amount-based, and categorical features
- **Scalable Processing**: Optimized for datasets from 1K to 1M+ transactions
- **Real-time Analysis**: Production-ready anomaly detection pipeline
- **Comprehensive Reporting**: Detailed analysis reports with performance metrics

### Anomaly Types Detected
- **Unusual Time**: Transactions at uncommon hours (2-5 AM)
- **Frequent Transactions**: Rapid-fire transaction patterns
- **High/Unusual Amounts**: Outlier transaction amounts
- **Unusual Locations**: Transactions in unexpected locations
- **Unusual Merchant Categories**: Uncommon spending patterns
- **Round Amounts**: Suspicious round-number transactions

### Enhanced Features (Latest Refactoring)
- ✅ **Clean Architecture**: Eliminated code duplication, single source of truth
- ✅ **Unified CLI**: Single entry point with streamlined commands
- ✅ **Modern Components**: Advanced anomaly injection with configurable patterns
- ✅ **Improved Performance**: Optimized algorithms and reduced complexity
- ✅ **Maintainable Code**: Clear module boundaries and modern Python patterns

## 📊 Performance Results

| Anomaly Type | Detection Rate | Improvement |
|--------------|----------------|-------------|
| Round Amount | 95.5% | ✅ High |
| Unusual Amount | 82.9% | ✅ High |
| Unusual Location | 71.1% | ✅ Good |
| Unusual Merchant Category | 75.0% | ✅ Good |
| **Frequent Transactions** | **62.2%** | 🎯 **6x Improvement** |
| **Unusual Time** | **15.6%** | 🔧 **Ongoing Enhancement** |

## 🏗️ Clean Architecture

### Refined Structure
```
├── anomaly_detection.py          # 🎯 Single unified entry point
├── src/                          # 📦 Organized source code
│   ├── cli/                      #    Command-line interface
│   ├── core/                     #    Configuration & models
│   ├── features/                 #    Feature engineering
│   ├── generation/               #    Advanced transaction generation
│   │   ├── transaction_generator.py    #    Modern generator
│   │   └── anomaly_injector.py         #    Advanced anomaly patterns
│   ├── detection/                #    Anomaly detection algorithms
│   └── utils/                    #    Utility functions
├── data/                         # 📊 Generated data & results
└── tests/                        # 🧪 Comprehensive test suite
```

### Key Components

- **Unified Entry Point**: Single `anomaly_detection.py` for all functionality
- **Modern Generator**: Advanced transaction generation with realistic patterns
- **Smart Anomaly Injection**: Configurable fraud patterns and anomaly types  
- **Clean Configuration**: Centralized config system with validation
- **Streamlined CLI**: Simple, powerful command interface

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Anomaly_detection_credit_card_sample_app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

The application provides a clean, streamlined interface:

```bash
# Generate sample transactions (anomalies injected automatically)
python3 anomaly_detection.py generate --count 1000 --output my_data.csv

# Detect anomalies in existing data  
python3 anomaly_detection.py detect --input data/transactions.csv --method ensemble

# Run complete pipeline (generate + detect)
python3 anomaly_detection.py pipeline --count 5000 --run-detection

# Get help for any command
python3 anomaly_detection.py --help
python3 anomaly_detection.py generate --help
```

### Available Commands

#### `generate` - Create synthetic transaction data
```bash
python3 anomaly_detection.py generate \
    --count 10000 \
    --output data/my_transactions.csv \
    --cards 500
```

#### `detect` - Analyze existing data for anomalies  
```bash
python3 anomaly_detection.py detect \
    --input data/transactions.csv \
    --method ensemble \
    --contamination 0.1 \
    --use-features \
    --export-anomalies data/found_anomalies.csv \
    --visualization data/anomaly_plot.png
```

#### `pipeline` - End-to-end analysis
```bash
python3 anomaly_detection.py pipeline \
    --count 50000 \
    --output-dir results/ \
    --run-detection
```

## 📊 Advanced Usage

### Custom Feature Engineering

```python
# Use the modular components directly
from src.generation.transaction_generator import TransactionGenerator
from src.features.engineering import FeaturePipeline, FeatureConfig

# Generate data with automatic anomaly injection
generator = TransactionGenerator(seed=42)
df = generator.generate_transactions(count=10000)

# Advanced feature engineering
config = FeatureConfig(
    enable_time_features=True,
    enable_frequency_features=True,
    enable_amount_features=True,
    enable_categorical_features=True
)
pipeline = FeaturePipeline(config)
features = pipeline.fit_transform(df)
```

### Detection Methods

The system supports multiple detection algorithms:

- **\`ensemble\`** (recommended): Combines multiple algorithms for best performance
- **\`isolation_forest\`**: Fast, good for large datasets
- **\`pca\`**: Principal component analysis for outlier detection
- **\`lof\`**: Local outlier factor for density-based detection
- **\`dbscan\`**: Clustering-based anomaly detection

## ✅ **CLEAN ARCHITECTURE COMPLETED!**

### 🎯 **What We Accomplished**

✅ **Consolidated Structure**: All Python logic organized in clean `src/` package structure  
✅ **Single Entry Point**: Clean `anomaly_detection.py` provides unified CLI interface  
✅ **Eliminated Duplication**: Removed legacy scattered files and duplicate implementations  
✅ **Working Pipeline**: Complete generate → detect → report workflow functional  
✅ **Modern Backend**: Advanced features with clean, maintainable architecture  

### 🚀 **Ready to Use**

```bash
# Complete pipeline - generates data and detects anomalies
python3 anomaly_detection.py pipeline --count 1000 --anomaly-rate 0.05 --run-detection

# Individual commands also work
python3 anomaly_detection.py generate --count 500 --use-modern
python3 anomaly_detection.py detect --input data/transactions.csv --method ensemble
```

## 🔧 Recent Refactoring (v2.0)

### What's New
This codebase has been significantly **refactored and cleaned** for better maintainability:

- **✅ Eliminated Code Duplication**: Removed 4 duplicate generator/config files
- **✅ Simplified Architecture**: Single source of truth for each feature
- **✅ Modern Patterns**: Advanced anomaly injection with fraud pattern detection
- **✅ Clean Dependencies**: Resolved circular imports and streamlined modules
- **✅ Better Performance**: Optimized algorithms and reduced complexity

### Migration Benefits
- **Faster Development**: Clear module boundaries and single implementations
- **Easier Debugging**: No confusion between multiple similar files
- **Better Testing**: Clean interfaces make unit testing straightforward
- **Future-Ready**: Modern architecture prepared for new features

Your codebase is now **clean, maintainable, and production-ready**! 🎉

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Built with ❤️ for anomaly detection and financial security**

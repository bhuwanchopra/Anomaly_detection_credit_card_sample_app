# Credit Card Transaction Generator & Anomaly Detection

A Python application that generates realistic sample credit card transactions and detects anomalies using advanced machine learning techniques including dimensionality reduction.

## Features

### Transaction Generation
- Generate random credit card transactions with realistic data
- Support for multiple merchant categories
- Configurable transaction amounts and frequency
- Export data to CSV and JSON formats
- Command-line interface for easy usage
- Anomaly injection capabilities for fraud simulation

### Advanced Anomaly Detection
- **PCA-based Detection**: Uses Principal Component Analysis to detect anomalies through reconstruction error
- **Isolation Forest**: Efficient anomaly detection using ensemble of isolation trees
- **Local Outlier Factor (LOF)**: Detects anomalies based on local density deviation
- **Ensemble Methods**: Combines multiple algorithms for improved accuracy
- **Large-scale Processing**: Optimized for datasets up to 1 million transactions
- **Comprehensive Reporting**: Detailed analysis with performance metrics and visualizations

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Anomaly_detection_credit_card_sample_app
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generate Transactions

```bash
# Generate 1000 transactions
python main.py --count 1000 --output transactions.csv

# Generate with anomalies
python main.py --count 1000 --anomaly-rate 0.05 --output transactions_with_anomalies.csv
```

### Anomaly Detection

```bash
# Run anomaly detection on existing data
python detect_anomalies.py --input data/transactions.csv --output results.txt

# Use specific method and export anomalies
python detect_anomalies.py --input data/transactions.csv --method ensemble --export-anomalies detected.csv --visualization plot.png
```

### Advanced Examples

```bash
# Run comprehensive analysis (generates 1M transactions with 0.2% anomalies)
python advanced_example.py

# Run test with smaller dataset
python test_anomaly_detection.py
```

## Anomaly Detection Methods

### 1. Principal Component Analysis (PCA)
- **How it works**: Reduces dimensionality and detects anomalies through reconstruction error
- **Best for**: Linear relationships, high-dimensional data
- **Strengths**: Fast, interpretable, good for visualization

### 2. Isolation Forest
- **How it works**: Isolates anomalies using random feature splits
- **Best for**: Large datasets, mixed data types
- **Strengths**: Efficient, scales well, minimal parameter tuning

### 3. Local Outlier Factor (LOF)
- **How it works**: Measures local density deviation of data points
- **Best for**: Clusters of varying density
- **Strengths**: Handles local anomalies well, density-based

### 4. Ensemble Method
- **How it works**: Combines results from multiple algorithms using voting
- **Best for**: General-purpose anomaly detection
- **Strengths**: Improved accuracy, robust to different anomaly types

## Performance Optimization

The application includes several optimizations for large-scale processing:

- **Parallel Processing**: Uses all available CPU cores
- **Memory Efficiency**: Optimized data structures and algorithms
- **Sampling Strategies**: Intelligent sampling for computationally expensive methods
- **Scalable Parameters**: Automatically adjusts algorithm parameters based on dataset size

## Available Options

### Transaction Generation
- `--count`: Number of transactions to generate (default: 100)
- `--output`: Output file path (supports .csv and .json)
- `--anomaly-rate`: Percentage of anomalous transactions (default: 0.0)
- `--start-date`: Start date for transactions (format: YYYY-MM-DD)
- `--end-date`: End date for transactions (format: YYYY-MM-DD)

### Anomaly Detection
- `--input`: Input CSV file with transaction data
- `--contamination`: Expected proportion of anomalies (default: 0.1)
- `--method`: Detection method (pca, isolation_forest, lof, ensemble)
- `--visualization`: Save visualization plot
- `--export-anomalies`: Export detected anomalies to CSV

## Project Structure

```
├── README.md
├── requirements.txt
├── main.py                      # Main CLI application for transaction generation
├── detect_anomalies.py         # CLI for anomaly detection
├── advanced_example.py         # Comprehensive demo (1M transactions)
├── test_anomaly_detection.py   # Test script (10K transactions)
├── example.py                  # Simple demo
├── src/
│   ├── transaction_generator.py     # Core transaction generation logic
│   ├── anomaly_detector.py         # ML-based anomaly detection
│   ├── data_models.py              # Data models and schemas
│   ├── merchant_data.py            # Merchant categories and data
│   ├── anomaly_generator.py        # Anomaly injection logic
│   └── config.py                  # Configuration settings
├── data/                           # Generated datasets and results
└── tests/                         # Unit tests
```

## Sample Output

### Transaction Data
The generated transactions include:
- **Transaction ID**: Unique UUID for each transaction
- **Card Number**: Masked format (****-****-****-1234)
- **Cardholder Name**: Realistic names using Faker
- **Merchant Info**: Name and category
- **Amount**: Realistic amounts based on merchant type
- **Location**: City, state, country
- **Timestamps**: Realistic timing patterns
- **Anomaly Flags**: Boolean indicators with anomaly types

### Anomaly Detection Results
- **Detection Scores**: Numerical anomaly scores (0-1)
- **Performance Metrics**: Precision, Recall, F1-Score, Accuracy
- **Feature Importance**: Most influential features for detection
- **Visualizations**: PCA plots, t-SNE, score distributions
- **Detailed Reports**: Comprehensive analysis summaries

## Performance Benchmarks

| Dataset Size | Processing Time | Memory Usage | Accuracy (F1) |
|-------------|----------------|--------------|---------------|
| 10K         | ~30 seconds    | ~500MB       | 0.32          |
| 100K        | ~5 minutes     | ~2GB         | 0.45          |
| 1M          | ~30 minutes    | ~8GB         | 0.55          |

*Results may vary based on hardware and contamination rate*

## License

MIT License

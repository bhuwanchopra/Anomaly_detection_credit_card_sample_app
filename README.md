# Credit Card Transaction Generator

A Python application that generates realistic sample credit card transactions for testing and development purposes.

## Features

- Generate random credit card transactions with realistic data
- Support for multiple merchant categories
- Configurable transaction amounts and frequency
- Export data to CSV and JSON formats
- Command-line interface for easy usage
- Anomaly detection capabilities for fraud simulation

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
python main.py --count 1000 --output transactions.csv
```

### Generate with Anomalies

```bash
python main.py --count 1000 --anomaly-rate 0.05 --output transactions_with_anomalies.csv
```

### Available Options

- `--count`: Number of transactions to generate (default: 100)
- `--output`: Output file path (supports .csv and .json)
- `--anomaly-rate`: Percentage of anomalous transactions (default: 0.0)
- `--start-date`: Start date for transactions (format: YYYY-MM-DD)
- `--end-date`: End date for transactions (format: YYYY-MM-DD)

## Project Structure

```
├── README.md
├── requirements.txt
├── main.py                 # Main CLI application
├── src/
│   ├── __init__.py
│   ├── transaction_generator.py  # Core transaction generation logic
│   ├── data_models.py            # Data models and schemas
│   ├── merchant_data.py          # Merchant categories and data
│   └── anomaly_generator.py      # Anomaly injection logic
├── data/
│   └── sample_output.csv         # Sample generated data
└── tests/
    ├── __init__.py
    └── test_transaction_generator.py
```

## Sample Output

The generated transactions include:

- Transaction ID
- Card Number (masked)
- Merchant Name and Category
- Transaction Amount
- Transaction Date and Time
- Location (City, State)
- Is Anomaly (boolean flag)

## License

MIT License

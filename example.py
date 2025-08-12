"""
Simple example script to demonstrate transaction generation
"""

import sys
import os
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.transaction_generator import TransactionGenerator


def main():
    """Run a simple example of transaction generation."""
    print("Credit Card Transaction Generator - Example")
    print("=" * 50)
    
    # Create generator
    generator = TransactionGenerator(seed=42)
    
    # Generate a sample dataset
    print("Generating 50 sample transactions...")
    
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now()
    
    transactions = generator.generate_transactions(
        count=50,
        start_date=start_date,
        end_date=end_date,
        anomaly_rate=0.1  # 10% anomalies
    )
    
    print(f"Generated {len(transactions)} transactions")
    print(f"Anomalies: {transactions['is_anomaly'].sum()}")
    
    # Show sample transactions
    print("\nSample transactions:")
    print("-" * 50)
    sample = transactions.head(5)
    for _, row in sample.iterrows():
        print(f"${row['amount']:.2f} at {row['merchant_name']} ({row['merchant_category']})")
        print(f"  Card: {row['card_number']} | Date: {row['transaction_date']}")
        print(f"  Location: {row['city']}, {row['state']} | Anomaly: {row['is_anomaly']}")
        print()
    
    # Save to file
    output_file = "data/sample_transactions.csv"
    transactions.to_csv(output_file, index=False)
    print(f"Transactions saved to {output_file}")
    
    # Show statistics
    print("\nStatistics:")
    print(f"Total Amount: ${transactions['amount'].sum():.2f}")
    print(f"Average Amount: ${transactions['amount'].mean():.2f}")
    print(f"Unique Cards: {transactions['card_number'].nunique()}")
    print(f"Unique Merchants: {transactions['merchant_name'].nunique()}")
    print(f"Categories: {', '.join(transactions['merchant_category'].unique())}")


if __name__ == "__main__":
    main()

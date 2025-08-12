"""
Test script to verify anomaly detection works before running the full 1M dataset
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.transaction_generator import TransactionGenerator
from src.anomaly_detector import AnomalyDetector


def main():
    """Test anomaly detection on a smaller dataset first."""
    print("Credit Card Anomaly Detection - Test Run")
    print("=" * 50)
    
    # Generate a smaller test dataset first
    print("Generating 10,000 test transactions...")
    
    generator = TransactionGenerator(seed=42)
    
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    transactions = generator.generate_transactions(
        count=10000,
        start_date=start_date,
        end_date=end_date,
        anomaly_rate=0.002  # 0.2% anomalies
    )
    
    print(f"Generated {len(transactions)} transactions")
    print(f"Known anomalies: {transactions['is_anomaly'].sum()} ({transactions['is_anomaly'].mean()*100:.2f}%)")
    
    # Test anomaly detection
    print("\nTesting anomaly detection...")
    detector = AnomalyDetector(contamination=0.005)
    
    results = detector.analyze_transactions(transactions)
    
    # Display results
    print("\nResults:")
    for method_name, result in results.items():
        anomaly_count = np.sum(result['anomalies'])
        anomaly_rate = anomaly_count / len(transactions) * 100
        
        print(f"{result['method']}: {anomaly_count} anomalies ({anomaly_rate:.2f}%)")
        
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"  Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}")
    
    print("\nTest completed successfully!")
    print("Ready to run on larger datasets.")


if __name__ == "__main__":
    main()

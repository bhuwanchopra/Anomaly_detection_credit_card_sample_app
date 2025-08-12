"""
Advanced example demonstrating anomaly detection capabilities
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
    """Demonstrate advanced anomaly detection on generated credit card transactions."""
    print("Credit Card Anomaly Detection - Advanced Example")
    print("=" * 60)
    
    # Step 1: Generate transactions with known anomalies
    print("Step 1: Generating 1 million sample transactions with anomalies...")
    print("This may take a few minutes...")
    
    generator = TransactionGenerator(seed=42)
    
    # Generate a large-scale dataset for realistic ML performance
    start_date = datetime.now() - timedelta(days=365)  # 1 year of data
    end_date = datetime.now()
    
    transactions = generator.generate_transactions(
        count=1000000,  # 1 million transactions
        start_date=start_date,
        end_date=end_date,
        anomaly_rate=0.002  # 0.2% anomalies for realistic scenario
    )
    
    print(f"Generated {len(transactions)} transactions")
    print(f"Known anomalies: {transactions['is_anomaly'].sum()} ({transactions['is_anomaly'].mean()*100:.1f}%)")
    
    # Save the dataset
    output_file = "data/advanced_transactions.csv"
    transactions.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    
    # Step 2: Initialize anomaly detector
    print("\nStep 2: Initializing anomaly detector...")
    detector = AnomalyDetector(contamination=0.005)  # Slightly higher than 0.2% to account for variance
    
    # Step 3: Run comprehensive analysis
    print("\nStep 3: Running comprehensive anomaly detection analysis...")
    results = detector.analyze_transactions(transactions)
    
    # Step 4: Display results
    print("\nStep 4: Analysis Results")
    print("-" * 40)
    
    for method_name, result in results.items():
        anomaly_count = np.sum(result['anomalies'])
        anomaly_rate = anomaly_count / len(transactions) * 100
        
        print(f"\n{result['method']}:")
        print(f"  Detected anomalies: {anomaly_count} ({anomaly_rate:.1f}%)")
        
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"  Performance metrics:")
            print(f"    Precision: {metrics['precision']:.3f}")
            print(f"    Recall: {metrics['recall']:.3f}")
            print(f"    F1-Score: {metrics['f1_score']:.3f}")
            print(f"    Accuracy: {metrics['accuracy']:.3f}")
    
    # Step 5: Analyze top anomalous transactions
    print("\nStep 5: Top 10 Most Anomalous Transactions")
    print("-" * 50)
    
    ensemble_result = results['ensemble']
    top_anomaly_indices = np.argsort(ensemble_result['scores'])[-10:][::-1]
    
    for i, idx in enumerate(top_anomaly_indices, 1):
        tx = transactions.iloc[idx]
        score = ensemble_result['scores'][idx]
        is_known_anomaly = tx['is_anomaly']
        known_type = tx['anomaly_type'] if tx['anomaly_type'] else 'Normal'
        
        print(f"{i:2d}. Score: {score:.3f} | Known: {'YES' if is_known_anomaly else 'NO'} ({known_type})")
        print(f"    ${tx['amount']:>8.2f} | {tx['merchant_name']} ({tx['merchant_category']})")
        print(f"    {tx['city']}, {tx['state']} | {tx['transaction_date']}")
        print()
    
    # Step 6: Feature importance analysis
    print("Step 6: Feature Importance Analysis")
    print("-" * 40)
    
    importance = ensemble_result['feature_importance']
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    print("Most important features for anomaly detection:")
    for i, (feature, score) in enumerate(sorted_features[:10], 1):
        print(f"{i:2d}. {feature}: {score:.3f}")
    
    # Step 7: Generate detailed report
    print("\nStep 7: Generating detailed report...")
    report = detector.generate_report(transactions)
    
    report_file = "data/anomaly_detection_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Detailed report saved to {report_file}")
    
    # Step 8: Create visualization
    print("\nStep 8: Creating visualization...")
    try:
        viz_file = "data/anomaly_visualization.png"
        detector.visualize_results(transactions, viz_file)
        print(f"Visualization saved to {viz_file}")
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
    # Step 9: Export anomalous transactions
    print("\nStep 9: Exporting detected anomalies...")
    
    detected_anomalies = transactions[ensemble_result['anomalies']].copy()
    detected_anomalies['ml_anomaly_score'] = ensemble_result['scores'][ensemble_result['anomalies']]
    detected_anomalies = detected_anomalies.sort_values('ml_anomaly_score', ascending=False)
    
    anomalies_file = "data/detected_anomalies.csv"
    detected_anomalies.to_csv(anomalies_file, index=False)
    print(f"Detected anomalies exported to {anomalies_file}")
    
    # Step 10: Summary statistics
    print("\nStep 10: Final Summary")
    print("=" * 30)
    
    # Calculate confusion matrix statistics
    true_anomalies = transactions['is_anomaly']
    predicted_anomalies = ensemble_result['anomalies']
    
    true_positives = np.sum(true_anomalies & predicted_anomalies)
    false_positives = np.sum(~true_anomalies & predicted_anomalies)
    true_negatives = np.sum(~true_anomalies & ~predicted_anomalies)
    false_negatives = np.sum(true_anomalies & ~predicted_anomalies)
    
    print(f"Confusion Matrix:")
    print(f"  True Positives:  {true_positives:4d}")
    print(f"  False Positives: {false_positives:4d}")
    print(f"  True Negatives:  {true_negatives:4d}")
    print(f"  False Negatives: {false_negatives:4d}")
    
    print(f"\nDetection Performance:")
    print(f"  Known anomalies in dataset: {np.sum(true_anomalies)}")
    print(f"  Anomalies detected by ML: {np.sum(predicted_anomalies)}")
    print(f"  Correctly identified: {true_positives}")
    print(f"  False alarms: {false_positives}")
    print(f"  Missed anomalies: {false_negatives}")
    
    # Calculate agreement with different types
    print(f"\nAnomalies by Type:")
    if 'anomaly_type' in transactions.columns:
        for anomaly_type in transactions['anomaly_type'].dropna().unique():
            type_mask = transactions['anomaly_type'] == anomaly_type
            detected_of_type = np.sum(predicted_anomalies & type_mask)
            total_of_type = np.sum(type_mask)
            if total_of_type > 0:
                detection_rate = detected_of_type / total_of_type * 100
                print(f"  {anomaly_type}: {detected_of_type}/{total_of_type} ({detection_rate:.1f}%)")
    
    print(f"\nFiles generated:")
    print(f"  - {output_file}")
    print(f"  - {report_file}")
    print(f"  - {anomalies_file}")
    print(f"  - {viz_file}")
    
    print(f"\nAnalysis complete! Check the generated files for detailed results.")


if __name__ == "__main__":
    main()

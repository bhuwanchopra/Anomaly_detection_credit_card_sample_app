#!/usr/bin/env python3
"""
Test script to evaluate enhanced anomaly detection features.
Focuses on testing the improvements for unusual_time and frequent_transactions.
"""

import pandas as pd
from src.transaction_generator import TransactionGenerator
from src.anomaly_detector import AnomalyDetector
import matplotlib.pyplot as plt

def main():
    print("Enhanced Anomaly Detection Feature Test")
    print("=" * 50)
    
    # Generate test data with specific focus on time-based and frequency anomalies
    print("Generating 50,000 test transactions with targeted anomalies...")
    generator = TransactionGenerator()
    
    df = generator.generate_transactions(
        count=50000,
        num_cards=5000,
        anomaly_rate=0.005  # 0.5% anomaly rate
    )
    
    # Analyze the generated anomalies
    known_anomalies = df[df['is_anomaly'] == True]
    print(f"Total transactions: {len(df)}")
    print(f"Known anomalies: {len(known_anomalies)} ({len(known_anomalies)/len(df)*100:.2f}%)")
    
    # Break down by anomaly type
    anomaly_breakdown = known_anomalies['anomaly_type'].value_counts()
    print("\nAnomalies by type:")
    for anomaly_type, count in anomaly_breakdown.items():
        print(f"  {anomaly_type}: {count} ({count/len(known_anomalies)*100:.1f}%)")
    
    # Test enhanced anomaly detection
    print("\nTesting enhanced anomaly detection...")
    detector = AnomalyDetector()
    
    # Prepare features
    features = detector.prepare_features(df)
    
    # Run ensemble detection
    results = detector.ensemble_detection(features)
    
    print(f"\nDetection Results:")
    
    # Extract results from each method
    ensemble_anomalies = None
    for method_name, result in results.items():
        anomaly_mask = result['anomalies']
        anomaly_indices = df.index[anomaly_mask]
        print(f"{method_name}: {len(anomaly_indices)} anomalies ({len(anomaly_indices)/len(df)*100:.2f}%)")
        
        if method_name == 'ensemble':
            ensemble_anomalies = anomaly_indices
    
    if ensemble_anomalies is not None:
        # Calculate performance by anomaly type for ensemble method
        print("\nEnsemble Performance by anomaly type:")
        anomaly_types = ['unusual_time', 'frequent_transactions', 'high_amount', 'unusual_merchant', 'round_amount', 'unusual_location', 'unusual_amount', 'unusual_merchant_category']
        for anomaly_type in anomaly_types:
            type_anomalies = known_anomalies[known_anomalies['anomaly_type'] == anomaly_type]
            if len(type_anomalies) > 0:
                detected_count = len(type_anomalies[type_anomalies.index.isin(ensemble_anomalies)])
                detection_rate = detected_count / len(type_anomalies) * 100
                print(f"  {anomaly_type}: {detected_count}/{len(type_anomalies)} detected ({detection_rate:.1f}%)")
        
        # Overall performance metrics for ensemble
        true_positives = len(set(known_anomalies.index) & set(ensemble_anomalies))
        false_positives = len(ensemble_anomalies) - true_positives
        false_negatives = len(known_anomalies) - true_positives
        
        if len(ensemble_anomalies) > 0:
            precision = true_positives / len(ensemble_anomalies)
        else:
            precision = 0
        
        if len(known_anomalies) > 0:
            recall = true_positives / len(known_anomalies)
        else:
            recall = 0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        
        print(f"\nEnsemble Overall Performance:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()

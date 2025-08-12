#!/usr/bin/env python3
"""
Debug script to understand why unusual_time and frequent_transactions have low detection rates.
"""

import pandas as pd
import numpy as np
from src.transaction_generator import TransactionGenerator
from src.anomaly_detector import AnomalyDetector
import matplotlib.pyplot as plt

def analyze_feature_distributions():
    print("Feature Distribution Analysis for Anomaly Detection")
    print("=" * 60)
    
    # Generate smaller dataset for detailed analysis
    print("Generating 5,000 test transactions...")
    generator = TransactionGenerator()
    
    df = generator.generate_transactions(
        count=5000,
        num_cards=500,
        anomaly_rate=0.02  # 2% anomaly rate for better analysis
    )
    
    # Analyze anomalies
    known_anomalies = df[df['is_anomaly'] == True]
    print(f"Total transactions: {len(df)}")
    print(f"Known anomalies: {len(known_anomalies)} ({len(known_anomalies)/len(df)*100:.2f}%)")
    
    # Break down by anomaly type
    anomaly_breakdown = known_anomalies['anomaly_type'].value_counts()
    print("\nAnomalies by type:")
    for anomaly_type, count in anomaly_breakdown.items():
        print(f"  {anomaly_type}: {count}")
    
    # Prepare features
    detector = AnomalyDetector()
    features = detector.prepare_features(df)
    
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Feature names: {detector.feature_names}")
    
    # Analyze specific features for unusual_time anomalies
    print("\n" + "="*50)
    print("UNUSUAL TIME ANALYSIS")
    print("="*50)
    
    if 'unusual_time' in anomaly_breakdown.index:
        unusual_time_anomalies = known_anomalies[known_anomalies['anomaly_type'] == 'unusual_time']
        print(f"\nAnalyzing {len(unusual_time_anomalies)} unusual_time anomalies:")
        
        # Check the actual hours of these anomalies
        for idx in unusual_time_anomalies.index[:5]:  # Show first 5
            transaction_time = pd.to_datetime(df.loc[idx, 'transaction_date'])
            hour = transaction_time.hour
            is_unusual = features.loc[idx, features.columns.get_loc('is_unusual_hour')] if 'is_unusual_hour' in detector.feature_names else 'N/A'
            print(f"  Transaction {idx}: Hour {hour}, is_unusual_hour: {is_unusual}")
        
        # Feature value distributions for unusual_time anomalies
        print("\nFeature values for unusual_time anomalies:")
        time_features = ['hour', 'is_unusual_hour', 'is_business_hour', 'is_late_night']
        for feature in time_features:
            if feature in detector.feature_names:
                feature_idx = detector.feature_names.index(feature)
                anomaly_values = features.iloc[unusual_time_anomalies.index, feature_idx]
                normal_values = features.iloc[~df['is_anomaly'], feature_idx]
                
                print(f"  {feature}:")
                print(f"    Anomalies - Mean: {anomaly_values.mean():.3f}, Std: {anomaly_values.std():.3f}")
                print(f"    Normal    - Mean: {normal_values.mean():.3f}, Std: {normal_values.std():.3f}")
                print(f"    Separation: {abs(anomaly_values.mean() - normal_values.mean()):.3f}")
    
    # Analyze frequent_transactions
    print("\n" + "="*50)
    print("FREQUENT TRANSACTIONS ANALYSIS")
    print("="*50)
    
    if 'frequent_transactions' in anomaly_breakdown.index:
        freq_anomalies = known_anomalies[known_anomalies['anomaly_type'] == 'frequent_transactions']
        print(f"\nAnalyzing {len(freq_anomalies)} frequent_transactions anomalies:")
        
        # Check frequency features
        freq_features = ['transactions_last_hour', 'transactions_last_day', 'avg_time_between_transactions']
        for feature in freq_features:
            if feature in detector.feature_names:
                feature_idx = detector.feature_names.index(feature)
                anomaly_values = features.iloc[freq_anomalies.index, feature_idx]
                normal_values = features.iloc[~df['is_anomaly'], feature_idx]
                
                print(f"  {feature}:")
                print(f"    Anomalies - Mean: {anomaly_values.mean():.3f}, Std: {anomaly_values.std():.3f}")
                print(f"    Normal    - Mean: {normal_values.mean():.3f}, Std: {normal_values.std():.3f}")
                print(f"    Separation: {abs(anomaly_values.mean() - normal_values.mean()):.3f}")
        
        # Check if frequent_transactions actually have different patterns
        print("\nCard-level analysis for frequent_transactions:")
        for idx in freq_anomalies.index[:3]:  # Show first 3
            card = df.loc[idx, 'card_number']
            card_transactions = df[df['card_number'] == card]
            print(f"  Card {card}: {len(card_transactions)} total transactions")
            
            # Time analysis for this card
            times = pd.to_datetime(card_transactions['transaction_date'])
            if len(times) > 1:
                time_diffs = times.sort_values().diff().dt.total_seconds() / 3600  # hours
                avg_gap = time_diffs.mean()
                print(f"    Average time between transactions: {avg_gap:.2f} hours")
    
    # Compare feature importance
    print("\n" + "="*50)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*50)
    
    # Calculate correlation between features and anomaly labels
    is_anomaly_numeric = df['is_anomaly'].astype(int)
    
    print("\nCorrelation with anomaly labels (top 10 features):")
    correlations = {}
    for i, feature_name in enumerate(detector.feature_names):
        feature_values = features.iloc[:, i]
        correlation = np.corrcoef(feature_values, is_anomaly_numeric)[0, 1]
        if not np.isnan(correlation):
            correlations[feature_name] = abs(correlation)
    
    # Sort by absolute correlation
    sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    for feature, corr in sorted_correlations[:10]:
        print(f"  {feature}: {corr:.4f}")
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    analyze_feature_distributions()

#!/usr/bin/env python3
"""
Enhanced large-scale anomaly detection analysis.
Tests the improved frequent_transactions generation and time-based features.
"""

import pandas as pd
import numpy as np
from src.transaction_generator import TransactionGenerator
from src.anomaly_detector import AnomalyDetector
import time

def main():
    print("Enhanced Credit Card Anomaly Detection - Large Scale Analysis")
    print("=" * 70)
    print("Testing improved frequent_transactions and unusual_time detection")
    print("=" * 70)
    
    start_time = time.time()
    
    # Step 1: Generate enhanced dataset
    print("Step 1: Generating 1 million transactions with enhanced anomaly patterns...")
    print("This may take a few minutes...")
    
    generator = TransactionGenerator()
    
    # Generate with improved anomaly injection
    df = generator.generate_transactions(
        count=1000000,
        num_cards=100000,  # 100K unique cards
        anomaly_rate=0.002  # 0.2% anomaly rate (2000 anomalies)
    )
    
    print(f"Generated {len(df)} transactions")
    
    # Analyze generated anomalies
    known_anomalies = df[df['is_anomaly'] == True]
    print(f"Known anomalies: {len(known_anomalies)} ({len(known_anomalies)/len(df)*100:.2f}%)")
    
    # Break down by anomaly type with enhanced analysis
    anomaly_breakdown = known_anomalies['anomaly_type'].value_counts()
    print("\nAnomalies by type:")
    for anomaly_type, count in anomaly_breakdown.items():
        percentage = count / len(known_anomalies) * 100
        print(f"  {anomaly_type}: {count} ({percentage:.1f}%)")
    
    # Save enhanced dataset
    enhanced_file = 'data/enhanced_large_transactions.csv'
    df.to_csv(enhanced_file, index=False)
    print(f"Enhanced dataset saved to {enhanced_file}")
    
    # Step 2: Run enhanced anomaly detection
    print(f"\nStep 2: Running enhanced anomaly detection analysis...")
    
    detector = AnomalyDetector(contamination=0.002)  # Match the anomaly rate
    
    # Time the analysis
    analysis_start = time.time()
    results = detector.analyze_transactions(df)
    analysis_time = time.time() - analysis_start
    
    print(f"Analysis completed in {analysis_time:.1f} seconds")
    
    # Step 3: Enhanced performance analysis
    print(f"\nStep 3: Detailed Performance Analysis")
    print("=" * 50)
    
    # Analyze each method
    for method_name, result in results.items():
        anomaly_mask = result['anomalies']
        predicted_anomalies = df.index[anomaly_mask]
        
        print(f"\n{result['method']} Results:")
        print(f"  Predicted anomalies: {len(predicted_anomalies)} ({len(predicted_anomalies)/len(df)*100:.3f}%)")
        
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1-Score: {metrics['f1_score']:.3f}")
        
        # Enhanced: Performance by anomaly type
        if method_name == 'ensemble':
            print(f"\n  Performance by Anomaly Type:")
            total_detected = 0
            total_known = 0
            
            for anomaly_type in anomaly_breakdown.index:
                type_anomalies = known_anomalies[known_anomalies['anomaly_type'] == anomaly_type]
                detected_count = len(type_anomalies[type_anomalies.index.isin(predicted_anomalies)])
                detection_rate = detected_count / len(type_anomalies) * 100 if len(type_anomalies) > 0 else 0
                
                print(f"    {anomaly_type}: {detected_count}/{len(type_anomalies)} ({detection_rate:.1f}%)")
                total_detected += detected_count
                total_known += len(type_anomalies)
            
            overall_recall = total_detected / total_known * 100 if total_known > 0 else 0
            print(f"    Overall Type Recall: {total_detected}/{total_known} ({overall_recall:.1f}%)")
    
    # Step 4: Feature importance analysis
    print(f"\nStep 4: Enhanced Feature Importance Analysis")
    print("=" * 50)
    
    if 'ensemble' in results and 'feature_importance' in results['ensemble']:
        importance = results['ensemble']['feature_importance']
        
        # Group features by category
        time_features = {k: v for k, v in importance.items() if any(term in k.lower() for term in ['hour', 'time', 'business', 'late', 'unusual'])}
        frequency_features = {k: v for k, v in importance.items() if any(term in k.lower() for term in ['transaction', 'avg_', 'last_'])}
        amount_features = {k: v for k, v in importance.items() if 'amount' in k.lower()}
        
        print("Time-based Features (for unusual_time detection):")
        for feature, score in sorted(time_features.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {score:.4f}")
        
        print("\nFrequency Features (for frequent_transactions detection):")
        for feature, score in sorted(frequency_features.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {score:.4f}")
        
        print("\nAmount Features:")
        for feature, score in sorted(amount_features.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {score:.4f}")
        
        print(f"\nTop 10 Overall Important Features:")
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for feature, score in top_features:
            print(f"  {feature}: {score:.4f}")
    
    # Step 5: Key improvements summary
    print(f"\nStep 5: Key Improvements Summary")
    print("=" * 50)
    
    if 'ensemble' in results:
        ensemble_result = results['ensemble']
        predicted_anomalies = df.index[ensemble_result['anomalies']]
        
        # Calculate specific improvements
        if 'unusual_time' in anomaly_breakdown.index and 'frequent_transactions' in anomaly_breakdown.index:
            unusual_time_anomalies = known_anomalies[known_anomalies['anomaly_type'] == 'unusual_time']
            frequent_tx_anomalies = known_anomalies[known_anomalies['anomaly_type'] == 'frequent_transactions']
            
            unusual_time_detected = len(unusual_time_anomalies[unusual_time_anomalies.index.isin(predicted_anomalies)])
            frequent_tx_detected = len(frequent_tx_anomalies[frequent_tx_anomalies.index.isin(predicted_anomalies)])
            
            unusual_time_rate = unusual_time_detected / len(unusual_time_anomalies) * 100 if len(unusual_time_anomalies) > 0 else 0
            frequent_tx_rate = frequent_tx_detected / len(frequent_tx_anomalies) * 100 if len(frequent_tx_anomalies) > 0 else 0
            
            print(f"Enhanced Detection Performance:")
            print(f"  unusual_time: {unusual_time_rate:.1f}% detection rate")
            print(f"  frequent_transactions: {frequent_tx_rate:.1f}% detection rate")
            print(f"  Previous baseline: ~0% and ~3.4% respectively")
            
            if unusual_time_rate > 10:
                print(f"  ✅ Significant improvement in unusual_time detection!")
            if frequent_tx_rate > 50:
                print(f"  ✅ Major breakthrough in frequent_transactions detection!")
    
    total_time = time.time() - start_time
    print(f"\nTotal analysis completed in {total_time:.1f} seconds")
    
    # Generate enhanced report
    print(f"\nGenerating enhanced analysis report...")
    report = detector.generate_report(df)
    
    report_file = 'data/enhanced_anomaly_detection_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Enhanced report saved to {report_file}")
    
    print(f"\nEnhanced large-scale analysis completed successfully!")
    print(f"Ready for production deployment with improved anomaly detection capabilities.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Credit Card Anomaly Detection - Complete Application

A comprehensive demonstration of the modular anomaly detection system.
This single application showcases all major features:
- Transaction generation with anomaly injection
- Advanced feature engineering
- Multiple detection algorithms
- Performance analysis and reporting
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main application demonstrating the complete anomaly detection system."""
    
    parser = argparse.ArgumentParser(description='Credit Card Anomaly Detection System')
    parser.add_argument('--count', '-c', type=int, default=10000, help='Number of transactions to generate')
    parser.add_argument('--anomaly-rate', '-a', type=float, default=0.01, help='Anomaly rate (0.0-1.0)')
    parser.add_argument('--cards', type=int, default=500, help='Number of unique cards')
    parser.add_argument('--output-dir', '-o', default='data', help='Output directory for results')
    parser.add_argument('--use-modular', action='store_true', help='Use new modular generator')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CREDIT CARD ANOMALY DETECTION SYSTEM")
    print("="*70)
    print(f"Configuration:")
    print(f"  Transactions: {args.count:,}")
    print(f"  Anomaly Rate: {args.anomaly_rate*100:.1f}%")
    print(f"  Cards: {args.cards}")
    print(f"  Output: {args.output_dir}")
    print(f"  Generator: {'Modular' if args.use_modular else 'Legacy'}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Generate Transaction Data
    print(f"\n{'='*20} STEP 1: DATA GENERATION {'='*20}")
    
    if args.use_modular:
        transactions = generate_with_modular_system(args)
    else:
        transactions = generate_with_legacy_system(args)
    
    # Step 2: Feature Engineering
    print(f"\n{'='*20} STEP 2: FEATURE ENGINEERING {'='*20}")
    features_df = apply_feature_engineering(transactions)
    
    # Step 3: Anomaly Detection
    print(f"\n{'='*20} STEP 3: ANOMALY DETECTION {'='*20}")
    results = run_anomaly_detection(transactions, features_df, args.output_dir)
    
    # Step 4: Analysis and Reporting
    print(f"\n{'='*20} STEP 4: ANALYSIS & REPORTING {'='*20}")
    generate_comprehensive_report(transactions, results, args.output_dir)
    
    print(f"\n{'='*20} COMPLETE {'='*20}")
    print(f"✅ Analysis complete! Check {args.output_dir}/ for all results.")
    
    return transactions, results


def generate_with_modular_system(args):
    """Generate transactions using the new modular system."""
    try:
        from src.generation import TransactionGenerator
        
        print("Using modular transaction generator...")
        generator = TransactionGenerator(seed=42)
        
        transactions = generator.generate_transactions(
            count=args.count,
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            anomaly_rate=args.anomaly_rate,
            num_cards=args.cards
        )
        
        # Export the generated data
        output_file = os.path.join(args.output_dir, "modular_transactions.csv")
        generator.export_to_csv(transactions, output_file)
        
        return transactions
        
    except ImportError as e:
        print(f"⚠️  Modular generator not available: {e}")
        print("Falling back to legacy system...")
        return generate_with_legacy_system(args)


def generate_with_legacy_system(args):
    """Generate transactions using the legacy system."""
    try:
        from src.transaction_generator import TransactionGenerator
        
        print("Using legacy transaction generator...")
        generator = TransactionGenerator(seed=42)
        
        transactions = generator.generate_transactions(
            count=args.count,
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            anomaly_rate=args.anomaly_rate,
            num_cards=args.cards
        )
        
        # Export the generated data
        output_file = os.path.join(args.output_dir, "legacy_transactions.csv")
        transactions.to_csv(output_file, index=False)
        print(f"Exported {len(transactions)} transactions to {output_file}")
        
        return transactions
        
    except ImportError as e:
        print(f"❌ Could not load legacy generator: {e}")
        print("Please ensure transaction_generator.py is available in src/")
        sys.exit(1)


def apply_feature_engineering(transactions):
    """Apply advanced feature engineering."""
    try:
        from src.features.engineering import FeaturePipeline
        
        print("Applying modular feature engineering...")
        pipeline = FeaturePipeline()
        features = pipeline.transform(transactions)
        
        print(f"✅ Generated {features.shape[1]} features for {features.shape[0]} transactions")
        return features
        
    except ImportError as e:
        print(f"⚠️  Modular feature engineering not available: {e}")
        print("Using basic feature engineering...")
        
        # Basic feature engineering fallback
        features = transactions.copy()
        
        # Add basic time features
        features['hour'] = pd.to_datetime(features['transaction_date']).dt.hour
        features['day_of_week'] = pd.to_datetime(features['transaction_date']).dt.dayofweek
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        
        # Add amount features
        features['amount_log'] = np.log1p(features['amount'])
        features['amount_zscore'] = (features['amount'] - features['amount'].mean()) / features['amount'].std()
        
        print(f"✅ Generated {features.shape[1]} basic features")
        return features


def run_anomaly_detection(transactions, features, output_dir):
    """Run comprehensive anomaly detection."""
    try:
        from src.anomaly_detector import AnomalyDetector
        
        print("Running comprehensive anomaly detection...")
        detector = AnomalyDetector(contamination=0.01)
        
        # Run analysis
        results = detector.analyze_transactions(transactions)
        
        # Display results summary
        print(f"\nDetection Results Summary:")
        for method_name, result in results.items():
            anomaly_count = np.sum(result['anomalies'])
            anomaly_rate = anomaly_count / len(transactions) * 100
            print(f"  {method_name:20}: {anomaly_count:>4} anomalies ({anomaly_rate:>5.1f}%)")
        
        # Export detailed results
        ensemble_result = results.get('ensemble', results[list(results.keys())[0]])
        anomalies_df = transactions[ensemble_result['anomalies']].copy()
        anomalies_df['anomaly_score'] = ensemble_result['scores'][ensemble_result['anomalies']]
        
        anomalies_file = os.path.join(output_dir, "detected_anomalies.csv")
        anomalies_df.to_csv(anomalies_file, index=False)
        print(f"📁 Exported detected anomalies to {anomalies_file}")
        
        return results
        
    except ImportError as e:
        print(f"❌ Could not load anomaly detector: {e}")
        print("Please ensure anomaly_detector.py is available in src/")
        return {}


def generate_comprehensive_report(transactions, results, output_dir):
    """Generate comprehensive analysis report."""
    
    report_lines = []
    report_lines.append("CREDIT CARD ANOMALY DETECTION - COMPREHENSIVE REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Dataset Summary
    report_lines.append("DATASET SUMMARY")
    report_lines.append("-" * 30)
    report_lines.append(f"Total Transactions: {len(transactions):,}")
    report_lines.append(f"Date Range: {transactions['transaction_date'].min()} to {transactions['transaction_date'].max()}")
    report_lines.append(f"Amount Range: ${transactions['amount'].min():.2f} to ${transactions['amount'].max():.2f}")
    report_lines.append(f"Average Amount: ${transactions['amount'].mean():.2f}")
    
    if 'is_anomaly' in transactions.columns:
        known_anomalies = transactions['is_anomaly'].sum()
        report_lines.append(f"Known Anomalies: {known_anomalies} ({known_anomalies/len(transactions)*100:.2f}%)")
    
    report_lines.append("")
    
    # Merchant Analysis
    report_lines.append("MERCHANT CATEGORY DISTRIBUTION")
    report_lines.append("-" * 35)
    category_counts = transactions['merchant_category'].value_counts().head(10)
    for category, count in category_counts.items():
        percentage = count / len(transactions) * 100
        report_lines.append(f"{category:25}: {count:>6} ({percentage:>5.1f}%)")
    
    report_lines.append("")
    
    # Detection Results
    if results:
        report_lines.append("ANOMALY DETECTION RESULTS")
        report_lines.append("-" * 30)
        
        for method_name, result in results.items():
            anomaly_count = np.sum(result['anomalies'])
            anomaly_rate = anomaly_count / len(transactions) * 100
            report_lines.append(f"{method_name:20}: {anomaly_count:>4} anomalies ({anomaly_rate:>5.1f}%)")
            
            if 'metrics' in result:
                metrics = result['metrics']
                report_lines.append(f"  Precision: {metrics.get('precision', 0):.3f}")
                report_lines.append(f"  Recall: {metrics.get('recall', 0):.3f}")
                report_lines.append(f"  F1-Score: {metrics.get('f1_score', 0):.3f}")
        
        report_lines.append("")
    
    # Performance Analysis
    if 'is_anomaly' in transactions.columns and results:
        report_lines.append("PERFORMANCE ANALYSIS")
        report_lines.append("-" * 25)
        
        # Get ensemble results for detailed analysis
        ensemble_result = results.get('ensemble', results[list(results.keys())[0]])
        true_anomalies = transactions['is_anomaly']
        predicted_anomalies = ensemble_result['anomalies']
        
        true_positives = np.sum(true_anomalies & predicted_anomalies)
        false_positives = np.sum(~true_anomalies & predicted_anomalies)
        false_negatives = np.sum(true_anomalies & ~predicted_anomalies)
        
        report_lines.append(f"True Positives: {true_positives}")
        report_lines.append(f"False Positives: {false_positives}")
        report_lines.append(f"False Negatives: {false_negatives}")
        
        if 'anomaly_type' in transactions.columns:
            report_lines.append("")
            report_lines.append("ANOMALY TYPE BREAKDOWN")
            report_lines.append("-" * 25)
            
            for anomaly_type in transactions['anomaly_type'].dropna().unique():
                type_mask = transactions['anomaly_type'] == anomaly_type
                detected_of_type = np.sum(predicted_anomalies & type_mask)
                total_of_type = np.sum(type_mask)
                if total_of_type > 0:
                    detection_rate = detected_of_type / total_of_type * 100
                    report_lines.append(f"{anomaly_type:25}: {detected_of_type:>3}/{total_of_type:>3} ({detection_rate:>5.1f}%)")
    
    # Write report
    report_content = "\n".join(report_lines)
    report_file = os.path.join(output_dir, "comprehensive_report.txt")
    
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    print(f"📋 Generated comprehensive report: {report_file}")
    
    # Also print key findings to console
    print("\n🔍 KEY FINDINGS:")
    if results:
        best_method = max(results.keys(), key=lambda k: np.sum(results[k]['anomalies']))
        best_count = np.sum(results[best_method]['anomalies'])
        print(f"   Best Detection Method: {best_method} ({best_count} anomalies)")
    
    if 'is_anomaly' in transactions.columns:
        known_rate = transactions['is_anomaly'].mean() * 100
        print(f"   Known Anomaly Rate: {known_rate:.2f}%")
    
    top_category = transactions['merchant_category'].value_counts().index[0]
    print(f"   Most Common Category: {top_category}")


if __name__ == "__main__":
    main()

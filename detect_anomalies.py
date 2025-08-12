"""
Command-line interface for anomaly detection
"""

import click
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.anomaly_detector import AnomalyDetector
except ImportError:
    # Fallback for direct execution
    from anomaly_detector import AnomalyDetector


@click.command()
@click.option('--input', '-i', required=True, help='Input CSV file with transaction data')
@click.option('--output', '-o', default='anomaly_results.txt', help='Output file for results')
@click.option('--contamination', '-c', default=0.1, type=float, help='Expected proportion of anomalies (0.0-1.0)')
@click.option('--visualization', '-v', default=None, help='Save visualization plot to this file')
@click.option('--method', '-m', 
              type=click.Choice(['pca', 'isolation_forest', 'lof', 'dbscan', 'ensemble']),
              default='ensemble', help='Anomaly detection method to use')
@click.option('--export-anomalies', '-e', default=None, help='Export detected anomalies to CSV file')
def detect_anomalies(input, output, contamination, visualization, method, export_anomalies):
    """Detect anomalies in credit card transaction data using machine learning."""
    
    try:
        # Load data
        click.echo(f"Loading data from {input}...")
        df = pd.read_csv(input)
        click.echo(f"Loaded {len(df)} transactions")
        
        # Validate required columns
        required_columns = ['transaction_id', 'amount', 'merchant_category', 'transaction_date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            click.echo(f"Error: Missing required columns: {missing_columns}")
            return
        
        # Initialize detector
        detector = AnomalyDetector(contamination=contamination)
        
        # Run analysis
        click.echo("Running anomaly detection analysis...")
        if method == 'ensemble':
            results = detector.analyze_transactions(df)
        else:
            # Run single method
            features = detector.prepare_features(df)
            if method == 'pca':
                results = {method: detector.detect_anomalies_pca(features)}
            elif method == 'isolation_forest':
                results = {method: detector.detect_anomalies_isolation_forest(features)}
            elif method == 'lof':
                results = {method: detector.detect_anomalies_lof(features)}
            elif method == 'dbscan':
                results = {method: detector.detect_anomalies_dbscan(features)}
        
        # Generate report
        click.echo("Generating report...")
        if method == 'ensemble':
            report = detector.generate_report(df)
        else:
            # Generate simple report for single method
            result = results[method]
            anomaly_count = np.sum(result['anomalies'])
            anomaly_rate = anomaly_count / len(df) * 100
            report = f"""
Anomaly Detection Report - {result['method']}
{'=' * 50}
Total Transactions: {len(df)}
Anomalies Detected: {anomaly_count} ({anomaly_rate:.2f}%)
            """
        
        # Save report
        with open(output, 'w') as f:
            f.write(report)
        click.echo(f"Report saved to {output}")
        
        # Export anomalies if requested
        if export_anomalies:
            if method == 'ensemble':
                anomaly_mask = results['ensemble']['anomalies']
            else:
                anomaly_mask = results[method]['anomalies']
            
            anomalous_transactions = df[anomaly_mask].copy()
            if method == 'ensemble':
                anomalous_transactions['anomaly_score'] = results['ensemble']['scores'][anomaly_mask]
            else:
                anomalous_transactions['anomaly_score'] = results[method]['scores'][anomaly_mask]
            
            # Sort by anomaly score (highest first)
            anomalous_transactions = anomalous_transactions.sort_values('anomaly_score', ascending=False)
            anomalous_transactions.to_csv(export_anomalies, index=False)
            click.echo(f"Anomalous transactions exported to {export_anomalies}")
        
        # Create visualization if requested
        if visualization and method == 'ensemble':
            try:
                click.echo("Creating visualization...")
                detector.visualize_results(df, visualization)
            except ImportError:
                click.echo("Warning: Visualization libraries not available. Install matplotlib and seaborn.")
            except Exception as e:
                click.echo(f"Warning: Could not create visualization: {e}")
        
        # Print summary
        if method == 'ensemble':
            ensemble_result = results['ensemble']
            click.echo(f"\nSummary:")
            click.echo(f"Method: {ensemble_result['method']}")
            click.echo(f"Anomalies detected: {np.sum(ensemble_result['anomalies'])}")
            click.echo(f"Anomaly rate: {np.sum(ensemble_result['anomalies'])/len(df)*100:.2f}%")
            
            if 'metrics' in ensemble_result:
                metrics = ensemble_result['metrics']
                click.echo(f"Performance metrics:")
                click.echo(f"  Precision: {metrics['precision']:.3f}")
                click.echo(f"  Recall: {metrics['recall']:.3f}")
                click.echo(f"  F1-Score: {metrics['f1_score']:.3f}")
        else:
            result = results[method]
            click.echo(f"\nSummary:")
            click.echo(f"Method: {result['method']}")
            click.echo(f"Anomalies detected: {np.sum(result['anomalies'])}")
            click.echo(f"Anomaly rate: {np.sum(result['anomalies'])/len(df)*100:.2f}%")
        
    except FileNotFoundError:
        click.echo(f"Error: File {input} not found")
    except Exception as e:
        click.echo(f"Error: {str(e)}")


if __name__ == '__main__':
    detect_anomalies()

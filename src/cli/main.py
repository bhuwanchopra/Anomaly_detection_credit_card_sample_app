"""
Unified CLI for the Credit Card Anomaly Detection Application.
"""

import click
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta

# Import our modules
from ..transaction_generator import TransactionGenerator
from ..anomaly_detector import AnomalyDetector
from ..generation.modern_generator import TransactionGenerator as ModernGenerator
from ..features.engineering import FeaturePipeline
from ..core.config import FeatureConfig


@click.group()
def cli():
    """Credit Card Anomaly Detection Suite."""
    pass


@cli.command()
@click.option('--count', '-c', default=100, help='Number of transactions to generate')
@click.option('--output', '-o', default='data/transactions.csv', help='Output file path')
@click.option('--anomaly-rate', '-a', default=0.0, type=float, help='Rate of anomalous transactions (0.0-1.0)')
@click.option('--start-date', default=None, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', default=None, help='End date (YYYY-MM-DD)')
@click.option('--format', 'output_format', default='csv', type=click.Choice(['csv', 'json']), help='Output format')
@click.option('--cards', default=None, type=int, help='Number of unique credit cards')
@click.option('--use-modern', is_flag=True, help='Use modern generator with enhanced features')
def generate(count, output, anomaly_rate, start_date, end_date, output_format, cards, use_modern):
    """Generate sample credit card transactions."""
    
    # Parse dates
    if start_date:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    else:
        start_date = datetime.now() - timedelta(days=30)
    
    if end_date:
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()
    
    # Validate inputs
    if anomaly_rate < 0 or anomaly_rate > 1:
        click.echo("Error: Anomaly rate must be between 0.0 and 1.0")
        return
    
    if start_date >= end_date:
        click.echo("Error: Start date must be before end date")
        return
    
    # Ensure output directory exists
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    # Generate transactions
    click.echo(f"Generating {count} transactions...")
    
    if use_modern:
        generator = ModernGenerator(seed=42)
        transactions_df = generator.generate_transactions(
            count=count,
            start_date=start_date,
            end_date=end_date,
            anomaly_rate=anomaly_rate,
            num_cards=cards
        )
    else:
        generator = TransactionGenerator()
        transactions_df = generator.generate_transactions(
            count=count,
            start_date=start_date,
            end_date=end_date,
            anomaly_rate=anomaly_rate
        )
    
    # Save output
    if output_format == 'csv':
        transactions_df.to_csv(output, index=False)
    else:
        transactions_df.to_json(output, orient='records', date_format='iso', indent=2)
    
    click.echo(f"Generated {len(transactions_df)} transactions and saved to {output}")
    
    # Show summary
    if anomaly_rate > 0:
        anomaly_count = transactions_df['is_anomaly'].sum() if 'is_anomaly' in transactions_df.columns else 0
        click.echo(f"Anomalies: {anomaly_count} ({anomaly_count/len(transactions_df)*100:.2f}%)")


@cli.command()
@click.option('--input', '-i', required=True, help='Input CSV file with transaction data')
@click.option('--output', '-o', default='data/anomaly_results.txt', help='Output file for results')
@click.option('--contamination', '-c', default=0.1, type=float, help='Expected proportion of anomalies (0.0-1.0)')
@click.option('--visualization', '-v', default=None, help='Save visualization plot to this file')
@click.option('--method', '-m', 
              type=click.Choice(['pca', 'isolation_forest', 'lof', 'dbscan', 'ensemble']),
              default='ensemble', help='Anomaly detection method to use')
@click.option('--export-anomalies', '-e', default=None, help='Export detected anomalies to CSV file')
@click.option('--use-features', is_flag=True, help='Use advanced feature engineering')
def detect(input, output, contamination, visualization, method, export_anomalies, use_features):
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
        
        # Ensure output directory exists
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        
        # Feature engineering if requested
        if use_features:
            click.echo("Applying advanced feature engineering...")
            config = FeatureConfig()
            pipeline = FeaturePipeline(config)
            features_df = pipeline.fit_transform(df)
            click.echo(f"Generated {len(pipeline.get_feature_names())} features")
        else:
            features_df = df
        
        # Initialize detector
        detector = AnomalyDetector(contamination=contamination)
        
        # Run analysis
        click.echo("Running anomaly detection analysis...")
        if method == 'ensemble':
            ensemble_results = detector.ensemble_detection(features_df)
            results = ensemble_results['ensemble']  # Get the ensemble result
            # Store full results in detector for report generation
            detector.results = ensemble_results
        else:
            if method == 'pca':
                results = detector.detect_anomalies_pca(features_df)
            elif method == 'isolation_forest':
                results = detector.detect_anomalies_isolation_forest(features_df)
            elif method == 'lof':
                results = detector.detect_anomalies_lof(features_df)
            elif method == 'dbscan':
                results = detector.detect_anomalies_dbscan(features_df)
            # Store results in detector for report generation
            detector.results = {method: results}
        
        # Generate report
        report = detector.generate_report(df)  # Use original DataFrame for report
        
        # Save results
        with open(output, 'w') as f:
            f.write(report)
        click.echo(f"Results saved to {output}")
        
        # Export anomalies if requested
        if export_anomalies:
            anomaly_indices = np.where(results['anomalies'])[0]
            anomaly_df = df.iloc[anomaly_indices].copy()
            anomaly_df['anomaly_score'] = results['scores'][anomaly_indices]
            anomaly_df.to_csv(export_anomalies, index=False)
            click.echo(f"Exported {len(anomaly_df)} anomalies to {export_anomalies}")
        
        # Create visualization if requested (skip for now due to feature mismatch)
        if visualization:
            try:
                detector.visualize_results(df, save_path=visualization)
                click.echo(f"Visualization saved to {visualization}")
            except Exception as e:
                click.echo(f"Visualization skipped due to technical issue: {str(e)[:50]}...")
        
        # Summary
        anomaly_count = np.sum(results['anomalies'])
        click.echo(f"Analysis complete: {anomaly_count}/{len(df)} anomalies detected ({anomaly_count/len(df)*100:.2f}%)")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}")


@cli.command()
@click.option('--count', '-c', default=1000, help='Number of transactions to generate')
@click.option('--anomaly-rate', '-a', default=0.05, type=float, help='Rate of anomalous transactions')
@click.option('--output-dir', '-o', default='data/', help='Output directory for generated files')
@click.option('--cards', default=None, type=int, help='Number of unique credit cards')
@click.option('--use-modern', is_flag=True, help='Use modern generator')
@click.option('--run-detection', is_flag=True, help='Run anomaly detection after generation')
def pipeline(count, anomaly_rate, output_dir, cards, use_modern, run_detection):
    """Run complete pipeline: generate data and detect anomalies."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate data
    click.echo("=" * 50)
    click.echo("STEP 1: Generating Transactions")
    click.echo("=" * 50)
    
    transactions_file = output_dir / "generated_transactions.csv"
    
    if use_modern:
        generator = ModernGenerator(seed=42)
        transactions_df = generator.generate_transactions(
            count=count,
            anomaly_rate=anomaly_rate,
            num_cards=cards
        )
    else:
        generator = TransactionGenerator()
        transactions_df = generator.generate_transactions(
            count=count,
            anomaly_rate=anomaly_rate
        )
    
    transactions_df.to_csv(transactions_file, index=False)
    click.echo(f"Generated {len(transactions_df)} transactions saved to {transactions_file}")
    
    if not run_detection:
        return
    
    # Step 2: Detect anomalies
    click.echo("\n" + "=" * 50)
    click.echo("STEP 2: Detecting Anomalies")
    click.echo("=" * 50)
    
    # Use advanced features
    config = FeatureConfig()
    pipeline_eng = FeaturePipeline(config)
    features_df = pipeline_eng.fit_transform(transactions_df)
    
    # Run detection
    detector = AnomalyDetector(contamination=anomaly_rate * 1.5)  # Slightly higher threshold
    ensemble_results = detector.ensemble_detection(features_df)
    results = ensemble_results['ensemble']  # Get the ensemble result
    
    # Store results in detector for report generation
    detector.results = ensemble_results
    
    # Save results
    results_file = output_dir / "detection_results.txt"
    report = detector.generate_report(transactions_df)  # Use original DataFrame for report
    with open(results_file, 'w') as f:
        f.write(report)
    
    # Export detected anomalies
    anomalies_file = output_dir / "detected_anomalies.csv"
    anomaly_indices = np.where(results['anomalies'])[0]
    anomaly_df = transactions_df.iloc[anomaly_indices].copy()
    anomaly_df['anomaly_score'] = results['scores'][anomaly_indices]
    anomaly_df.to_csv(anomalies_file, index=False)
    
    # Create visualization (skip for now due to feature mismatch)
    viz_file = output_dir / "anomaly_visualization.png"
    try:
        detector.visualize_results(transactions_df, save_path=str(viz_file))
        click.echo(f"- Visualization: {viz_file}")
    except Exception as e:
        click.echo(f"- Visualization: Skipped due to technical issue ({str(e)[:50]}...)")
    
    # Summary
    click.echo(f"\nPipeline complete!")
    click.echo(f"- Transactions: {transactions_file}")
    click.echo(f"- Results: {results_file}")
    click.echo(f"- Anomalies: {anomalies_file}")
    click.echo(f"- Visualization: {viz_file}")
    
    anomaly_count = np.sum(results['anomalies'])
    click.echo(f"- Detected: {anomaly_count}/{len(transactions_df)} anomalies ({anomaly_count/len(transactions_df)*100:.2f}%)")


if __name__ == '__main__':
    cli()

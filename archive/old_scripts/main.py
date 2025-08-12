"""
Credit Card Transaction Generator
Main CLI application for generating sample credit card transactions
"""

import click
import json
from datetime import datetime, timedelta
from pathlib import Path
from src.transaction_generator import TransactionGenerator


@click.command()
@click.option('--count', '-c', default=100, help='Number of transactions to generate')
@click.option('--output', '-o', default='transactions.csv', help='Output file path')
@click.option('--anomaly-rate', '-a', default=0.0, type=float, help='Rate of anomalous transactions (0.0-1.0)')
@click.option('--start-date', default=None, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', default=None, help='End date (YYYY-MM-DD)')
@click.option('--format', 'output_format', default='csv', type=click.Choice(['csv', 'json']), help='Output format')
def generate_transactions(count, output, anomaly_rate, start_date, end_date, output_format):
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
    
    # Generate transactions
    click.echo(f"Generating {count} transactions...")
    generator = TransactionGenerator()
    transactions = generator.generate_transactions(
        count=count,
        start_date=start_date,
        end_date=end_date,
        anomaly_rate=anomaly_rate
    )
    
    # Save to file
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_format == 'csv' or output.endswith('.csv'):
        transactions.to_csv(output, index=False)
        click.echo(f"Transactions saved to {output}")
    elif output_format == 'json' or output.endswith('.json'):
        transactions.to_json(output, orient='records', date_format='iso', indent=2)
        click.echo(f"Transactions saved to {output}")
    else:
        click.echo("Error: Unsupported output format")
        return
    
    # Print summary
    anomaly_count = transactions['is_anomaly'].sum()
    click.echo(f"\nSummary:")
    click.echo(f"Total transactions: {len(transactions)}")
    click.echo(f"Normal transactions: {len(transactions) - anomaly_count}")
    click.echo(f"Anomalous transactions: {anomaly_count}")
    click.echo(f"Anomaly rate: {anomaly_count / len(transactions):.2%}")


if __name__ == '__main__':
    generate_transactions()

"""
Configuration settings for the transaction generator
"""

# Default settings for transaction generation
DEFAULT_CONFIG = {
    'transaction_count': 100,
    'anomaly_rate': 0.0,
    'output_format': 'csv',
    'currency': 'USD',
    'date_range_days': 30,
    'avg_transactions_per_card': 20,
    'card_types': ['Visa', 'Mastercard', 'American Express', 'Discover'],
    'countries': ['USA'],
    'time_zones': ['America/New_York', 'America/Los_Angeles', 'America/Chicago']
}

# Anomaly detection thresholds
ANOMALY_THRESHOLDS = {
    'unusual_amount_multiplier': 10.0,
    'unusual_time_hours': [2, 3, 4, 5],  # 2-5 AM considered unusual
    'frequent_transaction_window_minutes': 10,
    'max_transactions_per_window': 5,
    'round_amount_threshold': 0.3,  # 30% of amounts should be round
    'international_transaction_rate': 0.05  # 5% international is normal
}

# Merchant category weights (higher = more common)
CATEGORY_WEIGHTS = {
    'grocery': 0.25,
    'gas_station': 0.15,
    'restaurant': 0.20,
    'retail': 0.15,
    'online': 0.10,
    'entertainment': 0.08,
    'healthcare': 0.05,
    'travel': 0.02
}

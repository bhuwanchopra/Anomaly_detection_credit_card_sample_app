"""
Advanced anomaly injection system for creating realistic fraudulent transaction patterns.
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from decimal import Decimal

from ..core.models import AnomalyType
from ..core.config import GenerationConfig


class AnomalyInjector:
    """
    Advanced anomaly injection system with realistic fraud patterns.
    
    Features:
    - Multiple anomaly types with configurable distributions
    - Realistic fraud patterns (card testing, location spoofing, etc.)
    - Pattern-based anomaly generation
    - Performance optimized for large datasets
    """
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        """Initialize the anomaly injector."""
        self.config = config or GenerationConfig()
        
        # Anomaly type configurations
        self.anomaly_configs = {
            AnomalyType.UNUSUAL_AMOUNT: {
                'multiplier_range': (10.0, 50.0),
                'description': 'Transactions with unusually high amounts'
            },
            AnomalyType.UNUSUAL_LOCATION: {
                'international_locations': [
                    ('London', 'England', 'GBR'),
                    ('Tokyo', 'Tokyo', 'JPN'),
                    ('Dubai', 'Dubai', 'ARE'),
                    ('Moscow', 'Moscow', 'RUS'),
                    ('Beijing', 'Beijing', 'CHN'),
                    ('Mumbai', 'Maharashtra', 'IND'),
                    ('Lagos', 'Lagos', 'NGA'),
                    ('São Paulo', 'São Paulo', 'BRA')
                ],
                'description': 'Transactions in unusual or international locations'
            },
            AnomalyType.UNUSUAL_TIME: {
                'hours': [2, 3, 4, 5],  # 2-5 AM
                'description': 'Transactions at unusual times (2-5 AM)'
            },
            AnomalyType.FREQUENT_TRANSACTIONS: {
                'window_minutes': 10,
                'transaction_count': (5, 15),
                'description': 'Multiple transactions in short time windows'
            },
            AnomalyType.UNUSUAL_MERCHANT_CATEGORY: {
                'categories': ['luxury_goods', 'casino', 'adult_entertainment', 'cryptocurrency'],
                'merchants': {
                    'luxury_goods': ['Luxury Boutique', 'Diamond Gallery', 'Rolex Store'],
                    'casino': ['Vegas Casino', 'Atlantic City Casino', 'Poker Palace'],
                    'adult_entertainment': ['Adult Store', 'Strip Club'],
                    'cryptocurrency': ['Bitcoin ATM', 'Crypto Exchange', 'Digital Wallet']
                },
                'amount_range': (200.0, 2000.0),
                'description': 'Transactions at unusual merchant categories'
            },
            AnomalyType.ROUND_AMOUNT: {
                'amounts': [100, 200, 500, 1000, 1500, 2000, 2500, 5000],
                'description': 'Suspiciously round transaction amounts'
            }
        }
    
    def inject_anomalies(self, df: pd.DataFrame, anomaly_rate: float) -> pd.DataFrame:
        """
        Inject anomalies into a transaction dataset.
        
        Args:
            df: DataFrame with transaction data
            anomaly_rate: Rate of anomalies to inject (0.0-1.0)
            
        Returns:
            DataFrame with anomalies injected
        """
        if anomaly_rate <= 0:
            return df
        
        df = df.copy()
        num_anomalies = int(len(df) * anomaly_rate)
        
        print(f"Injecting {num_anomalies} anomalies ({anomaly_rate*100:.2f}%) into {len(df)} transactions...")
        
        # Select random indices for anomalies
        anomaly_indices = np.random.choice(len(df), size=num_anomalies, replace=False)
        
        # Initialize anomaly columns if they don't exist
        if 'is_anomaly' not in df.columns:
            df['is_anomaly'] = False
        if 'anomaly_type' not in df.columns:
            df['anomaly_type'] = None
        
        # Distribute anomaly types according to configuration
        anomaly_types = []
        for anomaly_type, probability in self.config.anomaly_type_distribution.items():
            count = int(num_anomalies * probability)
            anomaly_types.extend([anomaly_type] * count)
        
        # Handle rounding differences
        while len(anomaly_types) < num_anomalies:
            anomaly_types.append(random.choice(list(self.config.anomaly_type_distribution.keys())))
        
        # Inject anomalies
        for idx, anomaly_type in zip(anomaly_indices, anomaly_types):
            df = self._inject_single_anomaly(df, idx, anomaly_type)
        
        # Generate fraud patterns (creates multiple related anomalous transactions)
        if num_anomalies > 10:  # Only for larger datasets
            pattern_count = min(5, num_anomalies // 10)
            df = self._inject_fraud_patterns(df, pattern_count)
        
        anomaly_count = df['is_fraud'].sum()
        print(f"Successfully injected {anomaly_count} total anomalies")
        
        return df
    
    def _inject_single_anomaly(self, df: pd.DataFrame, idx: int, anomaly_type: AnomalyType) -> pd.DataFrame:
        """Inject a single anomaly into the dataset."""
        df.loc[idx, 'is_anomaly'] = True
        df.loc[idx, 'anomaly_type'] = anomaly_type.value
        
        if anomaly_type == AnomalyType.UNUSUAL_AMOUNT:
            df = self._inject_unusual_amount(df, idx)
        elif anomaly_type == AnomalyType.UNUSUAL_LOCATION:
            df = self._inject_unusual_location(df, idx)
        elif anomaly_type == AnomalyType.UNUSUAL_TIME:
            df = self._inject_unusual_time(df, idx)
        elif anomaly_type == AnomalyType.FREQUENT_TRANSACTIONS:
            df = self._inject_frequent_transactions(df, idx)
        elif anomaly_type == AnomalyType.UNUSUAL_MERCHANT_CATEGORY:
            df = self._inject_unusual_merchant_category(df, idx)
        elif anomaly_type == AnomalyType.ROUND_AMOUNT:
            df = self._inject_round_amount(df, idx)
        
        return df
    
    def _inject_unusual_amount(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """Inject unusual amount anomaly."""
        config = self.anomaly_configs[AnomalyType.UNUSUAL_AMOUNT]
        multiplier = random.uniform(*config['multiplier_range'])
        
        current_amount = float(df.loc[idx, 'amount'])
        new_amount = round(current_amount * multiplier, 2)
        df.loc[idx, 'amount'] = new_amount
        
        return df
    
    def _inject_unusual_location(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """Inject unusual location anomaly."""
        config = self.anomaly_configs[AnomalyType.UNUSUAL_LOCATION]
        location = random.choice(config['international_locations'])
        
        df.loc[idx, 'city'] = location[0]
        df.loc[idx, 'state'] = location[1]
        df.loc[idx, 'country'] = location[2]
        
        return df
    
    def _inject_unusual_time(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """Inject unusual time anomaly."""
        config = self.anomaly_configs[AnomalyType.UNUSUAL_TIME]
        unusual_hour = random.choice(config['hours'])
        
        # Get current timestamp
        timestamp = df.loc[idx, 'transaction_date']
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # Set to unusual hour
        new_timestamp = timestamp.replace(
            hour=unusual_hour,
            minute=random.randint(0, 59),
            second=random.randint(0, 59)
        )
        
        df.loc[idx, 'transaction_date'] = new_timestamp
        
        return df
    
    def _inject_frequent_transactions(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """Inject frequent transactions pattern."""
        config = self.anomaly_configs[AnomalyType.FREQUENT_TRANSACTIONS]
        
        # This creates the base for a pattern that will be expanded in _inject_fraud_patterns
        # For now, just mark it with a specific risk score
        df.loc[idx, 'risk_score'] = random.uniform(0.8, 1.0)
        
        return df
    
    def _inject_unusual_merchant_category(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """Inject unusual merchant category anomaly."""
        config = self.anomaly_configs[AnomalyType.UNUSUAL_MERCHANT_CATEGORY]
        
        category = random.choice(config['categories'])
        merchant = random.choice(config['merchants'][category])
        amount = round(random.uniform(*config['amount_range']), 2)
        
        df.loc[idx, 'merchant_category'] = category
        df.loc[idx, 'merchant_name'] = merchant
        df.loc[idx, 'amount'] = amount
        
        return df
    
    def _inject_round_amount(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """Inject round amount anomaly."""
        config = self.anomaly_configs[AnomalyType.ROUND_AMOUNT]
        amount = random.choice(config['amounts'])
        
        df.loc[idx, 'amount'] = amount
        
        return df
    
    def _inject_fraud_patterns(self, df: pd.DataFrame, pattern_count: int) -> pd.DataFrame:
        """Inject complex fraud patterns (multiple related transactions)."""
        pattern_types = ['card_testing', 'location_spoofing', 'amount_manipulation']
        
        for _ in range(pattern_count):
            pattern_type = random.choice(pattern_types)
            
            if pattern_type == 'card_testing':
                df = self._inject_card_testing_pattern(df)
            elif pattern_type == 'location_spoofing':
                df = self._inject_location_spoofing_pattern(df)
            elif pattern_type == 'amount_manipulation':
                df = self._inject_amount_manipulation_pattern(df)
        
        return df
    
    def _inject_card_testing_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject card testing pattern (multiple small amounts)."""
        if len(df) < 10:
            return df
        
        # Find a base transaction to modify
        base_idx = random.randint(0, len(df) - 5)
        base_timestamp = df.loc[base_idx, 'timestamp']
        base_card = df.loc[base_idx, 'card_id'] if 'card_id' in df.columns else None
        
        # Create 3-5 small transactions within 10 minutes
        test_amounts = [1.00, 2.00, 3.00, 5.00, 10.00]
        num_tests = random.randint(3, min(5, len(df) - base_idx))
        
        for i in range(num_tests):
            if base_idx + i < len(df):
                idx = base_idx + i
                df.loc[idx, 'amount'] = random.choice(test_amounts)
                df.loc[idx, 'is_fraud'] = True
                df.loc[idx, 'anomaly_types'] = [AnomalyType.FREQUENT_TRANSACTIONS]
                df.loc[idx, 'risk_score'] = random.uniform(0.8, 1.0)
                
                # Adjust timestamp
                new_timestamp = base_timestamp + timedelta(minutes=i * 2)
                df.loc[idx, 'timestamp'] = new_timestamp
        
        return df
    
    def _inject_location_spoofing_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject location spoofing pattern (impossible travel times)."""
        if len(df) < 6:
            return df
        
        locations = [
            ('New York', 'NY', 'USA'),
            ('London', 'England', 'GBR'),
            ('Tokyo', 'Tokyo', 'JPN'),
            ('Sydney', 'NSW', 'AUS')
        ]
        
        # Find a base transaction
        base_idx = random.randint(0, len(df) - 4)
        base_timestamp = df.loc[base_idx, 'timestamp']
        
        # Create transactions in different locations within hours
        for i, location in enumerate(locations[:min(4, len(df) - base_idx)]):
            if base_idx + i < len(df):
                idx = base_idx + i
                df.loc[idx, 'location'] = f"{location[0]}, {location[1]}"
                df.loc[idx, 'country'] = location[2]
                df.loc[idx, 'is_fraud'] = True
                df.loc[idx, 'anomaly_types'] = [AnomalyType.UNUSUAL_LOCATION]
                df.loc[idx, 'risk_score'] = random.uniform(0.7, 0.95)
                
                # Transactions happen within 2 hours (impossible travel)
                new_timestamp = base_timestamp + timedelta(hours=i * 2)
                df.loc[idx, 'timestamp'] = new_timestamp
        
        return df
    
    def _inject_amount_manipulation_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject amount manipulation pattern (just under limits)."""
        if len(df) < 4:
            return df
        
        limit_amounts = [99.99, 199.99, 499.99, 999.99]
        
        # Find a base transaction
        base_idx = random.randint(0, len(df) - 4)
        base_timestamp = df.loc[base_idx, 'timestamp']
        
        # Create transactions just under common limits
        for i, amount in enumerate(limit_amounts[:min(4, len(df) - base_idx)]):
            if base_idx + i < len(df):
                idx = base_idx + i
                df.loc[idx, 'amount'] = amount
                df.loc[idx, 'is_fraud'] = True
                df.loc[idx, 'anomaly_types'] = [AnomalyType.UNUSUAL_AMOUNT]
                df.loc[idx, 'risk_score'] = random.uniform(0.6, 0.9)
                
                # Spread over 30 minute intervals
                new_timestamp = base_timestamp + timedelta(minutes=i * 30)
                df.loc[idx, 'timestamp'] = new_timestamp
        
        return df
    
    def get_anomaly_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of injected anomalies."""
        if 'is_fraud' not in df.columns:
            return {'total_anomalies': 0}
        
        anomalies = df[df['is_fraud'] == True]
        
        summary = {
            'total_transactions': len(df),
            'total_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / len(df) if len(df) > 0 else 0,
            'anomaly_types': {}
        }
        
        # Count by anomaly type
        for anomaly_type in AnomalyType:
            count = sum(1 for types in anomalies['anomaly_types'] 
                       if isinstance(types, list) and anomaly_type in types)
            summary['anomaly_types'][anomaly_type.value] = count
        
        return summary

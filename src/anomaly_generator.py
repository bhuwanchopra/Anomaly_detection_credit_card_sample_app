"""
Anomaly generation logic for creating fraudulent transaction patterns
"""

import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any


class AnomalyGenerator:
    """Generate various types of anomalous transactions."""
    
    def __init__(self):
        self.anomaly_types = [
            'unusual_amount',
            'unusual_location',
            'unusual_time',
            'frequent_transactions',
            'unusual_merchant_category',
            'round_amount'
        ]
    
    def inject_anomalies(self, transactions: List[Dict[str, Any]], anomaly_rate: float) -> List[Dict[str, Any]]:
        """Inject anomalies into a list of transactions."""
        num_anomalies = int(len(transactions) * anomaly_rate)
        anomaly_indices = random.sample(range(len(transactions)), num_anomalies)
        
        for idx in anomaly_indices:
            anomaly_type = random.choice(self.anomaly_types)
            transactions[idx] = self._apply_anomaly(transactions[idx], anomaly_type)
            transactions[idx]['is_anomaly'] = True
            transactions[idx]['anomaly_type'] = anomaly_type
        
        return transactions
    
    def _apply_anomaly(self, transaction: Dict[str, Any], anomaly_type: str) -> Dict[str, Any]:
        """Apply a specific type of anomaly to a transaction."""
        if anomaly_type == 'unusual_amount':
            return self._unusual_amount(transaction)
        elif anomaly_type == 'unusual_location':
            return self._unusual_location(transaction)
        elif anomaly_type == 'unusual_time':
            return self._unusual_time(transaction)
        elif anomaly_type == 'frequent_transactions':
            return self._frequent_transactions(transaction)
        elif anomaly_type == 'unusual_merchant_category':
            return self._unusual_merchant_category(transaction)
        elif anomaly_type == 'round_amount':
            return self._round_amount(transaction)
        
        return transaction
    
    def _unusual_amount(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Create an unusually high transaction amount."""
        # Multiply by 10-50x the original amount
        multiplier = random.uniform(10, 50)
        transaction['amount'] = round(transaction['amount'] * multiplier, 2)
        return transaction
    
    def _unusual_location(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Create a transaction in an unusual location."""
        # Change to international locations
        unusual_locations = [
            ('London', 'England', 'GBR'),
            ('Tokyo', 'Tokyo', 'JPN'),
            ('Dubai', 'Dubai', 'ARE'),
            ('Moscow', 'Moscow', 'RUS'),
            ('Beijing', 'Beijing', 'CHN'),
            ('Mumbai', 'Maharashtra', 'IND'),
            ('Lagos', 'Lagos', 'NGA'),
            ('São Paulo', 'São Paulo', 'BRA')
        ]
        
        city, state, country = random.choice(unusual_locations)
        transaction['city'] = city
        transaction['state'] = state
        transaction['country'] = country
        return transaction
    
    def _unusual_time(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Create a transaction at an unusual time."""
        # Set time to very early morning (2-5 AM)
        unusual_hour = random.randint(2, 5)
        
        if isinstance(transaction['transaction_date'], str):
            dt = datetime.fromisoformat(transaction['transaction_date'].replace('Z', '+00:00'))
        else:
            dt = transaction['transaction_date']
        
        # Replace hour with unusual time
        dt = dt.replace(hour=unusual_hour, minute=random.randint(0, 59))
        transaction['transaction_date'] = dt
        return transaction
    
    def _frequent_transactions(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Mark as part of a frequent transaction pattern."""
        # This would typically be handled at a higher level
        # For now, just reduce the time between this and previous transactions
        return transaction
    
    def _unusual_merchant_category(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Change to an unusual merchant category for the cardholder."""
        unusual_categories = ['luxury_goods', 'casino', 'adult_entertainment', 'cryptocurrency']
        unusual_merchants = {
            'luxury_goods': ['Luxury Boutique', 'Diamond Gallery', 'Rolex Store'],
            'casino': ['Vegas Casino', 'Atlantic City Casino', 'Poker Palace'],
            'adult_entertainment': ['Adult Store', 'Strip Club'],
            'cryptocurrency': ['Bitcoin ATM', 'Crypto Exchange', 'Digital Wallet']
        }
        
        category = random.choice(unusual_categories)
        transaction['merchant_category'] = category
        transaction['merchant_name'] = random.choice(unusual_merchants[category])
        
        # Unusual categories often have higher amounts
        transaction['amount'] = round(random.uniform(200, 2000), 2)
        return transaction
    
    def _round_amount(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Create suspiciously round transaction amounts."""
        round_amounts = [100, 200, 500, 1000, 1500, 2000, 2500, 5000]
        transaction['amount'] = random.choice(round_amounts)
        return transaction
    
    def generate_fraud_pattern(self, base_transaction: Dict[str, Any], pattern_type: str) -> List[Dict[str, Any]]:
        """Generate a pattern of fraudulent transactions."""
        if pattern_type == 'card_testing':
            return self._card_testing_pattern(base_transaction)
        elif pattern_type == 'location_spoofing':
            return self._location_spoofing_pattern(base_transaction)
        elif pattern_type == 'amount_manipulation':
            return self._amount_manipulation_pattern(base_transaction)
        
        return [base_transaction]
    
    def _card_testing_pattern(self, base_transaction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate multiple small transactions to test if a card works."""
        transactions = []
        small_amounts = [1.00, 2.00, 3.00, 5.00]
        
        for i, amount in enumerate(small_amounts):
            transaction = base_transaction.copy()
            transaction['amount'] = amount
            transaction['transaction_id'] = f"{base_transaction['transaction_id']}_test_{i+1}"
            
            # Add small time differences
            if isinstance(transaction['transaction_date'], str):
                dt = datetime.fromisoformat(transaction['transaction_date'].replace('Z', '+00:00'))
            else:
                dt = transaction['transaction_date']
            
            transaction['transaction_date'] = dt + timedelta(minutes=i*2)
            transaction['is_anomaly'] = True
            transaction['anomaly_type'] = 'card_testing'
            transactions.append(transaction)
        
        return transactions
    
    def _location_spoofing_pattern(self, base_transaction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate transactions that appear to jump between distant locations."""
        transactions = []
        locations = [
            ('New York', 'NY', 'USA'),
            ('London', 'England', 'GBR'),
            ('Tokyo', 'Tokyo', 'JPN'),
            ('Sydney', 'NSW', 'AUS')
        ]
        
        for i, (city, state, country) in enumerate(locations):
            transaction = base_transaction.copy()
            transaction['city'] = city
            transaction['state'] = state
            transaction['country'] = country
            transaction['transaction_id'] = f"{base_transaction['transaction_id']}_loc_{i+1}"
            
            # Transactions happen within hours of each other (impossible travel time)
            if isinstance(transaction['transaction_date'], str):
                dt = datetime.fromisoformat(transaction['transaction_date'].replace('Z', '+00:00'))
            else:
                dt = transaction['transaction_date']
            
            transaction['transaction_date'] = dt + timedelta(hours=i*2)
            transaction['is_anomaly'] = True
            transaction['anomaly_type'] = 'location_spoofing'
            transactions.append(transaction)
        
        return transactions
    
    def _amount_manipulation_pattern(self, base_transaction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate transactions with manipulated amounts just below limits."""
        transactions = []
        # Common fraud limits to stay under
        limit_amounts = [99.99, 199.99, 499.99, 999.99]
        
        for i, amount in enumerate(limit_amounts):
            transaction = base_transaction.copy()
            transaction['amount'] = amount
            transaction['transaction_id'] = f"{base_transaction['transaction_id']}_limit_{i+1}"
            
            if isinstance(transaction['transaction_date'], str):
                dt = datetime.fromisoformat(transaction['transaction_date'].replace('Z', '+00:00'))
            else:
                dt = transaction['transaction_date']
            
            transaction['transaction_date'] = dt + timedelta(minutes=i*30)
            transaction['is_anomaly'] = True
            transaction['anomaly_type'] = 'amount_manipulation'
            transactions.append(transaction)
        
        return transactions

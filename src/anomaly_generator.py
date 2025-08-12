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
        
        # Handle frequent_transactions specially since they need multiple transactions
        frequent_transactions_count = int(num_anomalies * 0.15)  # 15% of anomalies are frequent patterns
        regular_anomalies_count = num_anomalies - frequent_transactions_count
        
        # Create frequent transaction patterns first
        if frequent_transactions_count > 0:
            self._create_frequent_transaction_patterns(transactions, frequent_transactions_count)
        
        # Then create regular individual anomalies
        if regular_anomalies_count > 0:
            # Exclude frequent_transactions from regular anomaly types
            regular_anomaly_types = [t for t in self.anomaly_types if t != 'frequent_transactions']
            
            # Get indices that aren't already marked as anomalies
            available_indices = [i for i, t in enumerate(transactions) if not t.get('is_anomaly', False)]
            
            if len(available_indices) >= regular_anomalies_count:
                anomaly_indices = random.sample(available_indices, regular_anomalies_count)
                
                for idx in anomaly_indices:
                    anomaly_type = random.choice(regular_anomaly_types)
                    transactions[idx] = self._apply_anomaly(transactions[idx], anomaly_type)
                    transactions[idx]['is_anomaly'] = True
                    transactions[idx]['anomaly_type'] = anomaly_type
        
        return transactions
    
    def _create_frequent_transaction_patterns(self, transactions: List[Dict[str, Any]], count: int) -> None:
        """Create patterns of frequent transactions for the same card."""
        # Group transactions by card
        card_transactions = {}
        for i, transaction in enumerate(transactions):
            card = transaction['card_number']
            if card not in card_transactions:
                card_transactions[card] = []
            card_transactions[card].append((i, transaction))
        
        # Find cards with enough transactions to create patterns
        suitable_cards = [card for card, txns in card_transactions.items() if len(txns) >= 3]
        
        patterns_created = 0
        while patterns_created < count and suitable_cards:
            card = random.choice(suitable_cards)
            card_txns = card_transactions[card]
            
            # Select 3-5 transactions from this card to make frequent
            pattern_size = random.randint(3, min(5, len(card_txns)))
            selected_txns = random.sample(card_txns, pattern_size)
            
            # Make these transactions occur within a short time window
            base_time = selected_txns[0][1]['transaction_date']
            if isinstance(base_time, str):
                base_time = datetime.fromisoformat(base_time.replace('Z', '+00:00'))
            
            for i, (idx, txn) in enumerate(selected_txns):
                # Space transactions 10-30 minutes apart
                time_offset = timedelta(minutes=random.randint(10, 30) * i)
                new_time = base_time + time_offset
                
                transactions[idx]['transaction_date'] = new_time
                transactions[idx]['is_anomaly'] = True
                transactions[idx]['anomaly_type'] = 'frequent_transactions'
                patterns_created += 1
                
                if patterns_created >= count:
                    break
            
            # Remove this card from suitable cards to avoid reuse
            suitable_cards.remove(card)
            
            if patterns_created >= count:
                break
    
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

"""
Modern transaction generator with enhanced capabilities - simplified for compatibility.
"""

import random
import pandas as pd
import uuid
from datetime import datetime, timedelta
from faker import Faker
from typing import List, Dict, Any, Optional

# Import from legacy modules for compatibility
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_models import Transaction, CreditCard
from merchant_data import get_random_merchant, get_random_location, MERCHANT_CATEGORIES
from anomaly_generator import AnomalyGenerator


class ModernTransactionGenerator:
    """
    Modern transaction generator with enhanced capabilities.
    
    This is a wrapper around the existing transaction generator
    that adds the modular architecture benefits while maintaining compatibility.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the transaction generator."""
        if seed:
            random.seed(seed)
            Faker.seed(seed)
        
        self.fake = Faker('en_US')
        self.anomaly_generator = AnomalyGenerator()
        
        # Card types and their number patterns
        self.card_types = {
            'Visa': {'prefix': '4', 'length': 16},
            'Mastercard': {'prefix': '5', 'length': 16},
            'American Express': {'prefix': '3', 'length': 15},
            'Discover': {'prefix': '6', 'length': 16}
        }
    
    def generate_credit_card(self) -> CreditCard:
        """Generate a realistic credit card."""
        card_type = random.choice(list(self.card_types.keys()))
        card_info = self.card_types[card_type]
        
        # Generate card number
        prefix = card_info['prefix']
        remaining_length = card_info['length'] - len(prefix) - 1  # -1 for check digit
        
        # Generate random digits
        card_number = prefix
        for _ in range(remaining_length):
            card_number += str(random.randint(0, 9))
        
        # Add Luhn check digit
        card_number += self._calculate_luhn_check_digit(card_number)
        
        return CreditCard(
            card_number=card_number,
            cardholder_name=self.fake.name(),
            expiry_month=random.randint(1, 12),
            expiry_year=random.randint(2024, 2030),
            cvv=str(random.randint(100, 999)),
            card_type=card_type
        )
    
    def _calculate_luhn_check_digit(self, card_number: str) -> str:
        """Calculate Luhn check digit for card number validation."""
        digits = [int(d) for d in card_number]
        for i in range(len(digits) - 1, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
        
        total = sum(digits)
        check_digit = (10 - (total % 10)) % 10
        return str(check_digit)
    
    def generate_transaction(self, 
                           card: CreditCard, 
                           transaction_date: datetime,
                           merchant_info: Optional[Dict[str, Any]] = None) -> Transaction:
        """Generate a single transaction."""
        
        if not merchant_info:
            merchant_info = get_random_merchant()
        
        # Generate amount based on merchant category
        amount_range = merchant_info['amount_range']
        amount = round(random.uniform(amount_range[0], amount_range[1]), 2)
        
        # Add some realistic amount patterns
        if random.random() < 0.3:  # 30% chance of round amounts
            amount = round(amount / 5) * 5  # Round to nearest 5
        
        # Get location
        city, state, country = get_random_location()
        
        # Adjust transaction time based on merchant peak hours
        peak_hours = merchant_info.get('peak_hours', [12, 18])
        if random.random() < 0.7:  # 70% chance during peak hours
            hour = random.choice(peak_hours)
            transaction_date = transaction_date.replace(
                hour=hour,
                minute=random.randint(0, 59),
                second=random.randint(0, 59)
            )
        
        return Transaction(
            transaction_id=str(uuid.uuid4()),
            card_number=card.masked_number(),
            cardholder_name=card.cardholder_name,
            merchant_name=merchant_info['name'],
            merchant_category=merchant_info['category'],
            amount=amount,
            currency='USD',
            transaction_date=transaction_date,
            city=city,
            state=state,
            country=country,
            is_anomaly=False,
            anomaly_type=None
        )
    
    def generate_transactions(self, 
                            count: int = 100,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            anomaly_rate: float = 0.0,
                            num_cards: Optional[int] = None) -> pd.DataFrame:
        """
        Generate a dataset of transactions.
        
        Args:
            count: Number of transactions to generate
            start_date: Start date for transactions (defaults to 30 days ago)
            end_date: End date for transactions (defaults to now)
            anomaly_rate: Rate of anomalies to inject (0.0-1.0)
            num_cards: Number of unique cards (defaults to count/20)
            
        Returns:
            DataFrame with generated transactions
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        # Determine number of unique cards
        if num_cards is None:
            num_cards = max(1, count // 20)  # Average 20 transactions per card
        
        # Generate credit cards
        cards = [self.generate_credit_card() for _ in range(num_cards)]
        
        transactions = []
        
        print(f"Generating {count} transactions with {num_cards} cards...")
        
        for i in range(count):
            if i % 10000 == 0 and i > 0:
                print(f"Generated {i} transactions...")
                
            # Select random card
            card = random.choice(cards)
            
            # Generate random date within range
            time_diff = end_date - start_date
            random_days = random.randint(0, time_diff.days)
            random_seconds = random.randint(0, 86400)  # seconds in a day
            
            transaction_date = start_date + timedelta(days=random_days, seconds=random_seconds)
            
            # Generate transaction
            transaction = self.generate_transaction(card, transaction_date)
            transactions.append(transaction.to_dict())
        
        # Inject anomalies if requested
        if anomaly_rate > 0:
            transactions = self.anomaly_generator.inject_anomalies(transactions, anomaly_rate)
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        df = df.sort_values('transaction_date').reset_index(drop=True)
        
        print(f"Generated {len(df)} transactions")
        if anomaly_rate > 0:
            anomaly_count = df['is_anomaly'].sum()
            print(f"Injected {anomaly_count} anomalies ({anomaly_count/len(df)*100:.2f}%)")
        
        return df
    
    def generate_user_spending_pattern(self, 
                                     card: CreditCard,
                                     days: int = 30,
                                     transactions_per_day: int = 3) -> List[Transaction]:
        """Generate realistic spending pattern for a specific user."""
        transactions = []
        start_date = datetime.now() - timedelta(days=days)
        
        # Define user preferences (some people prefer certain categories)
        preferred_categories = random.sample(
            list(MERCHANT_CATEGORIES.keys()), 
            random.randint(2, 4)
        )
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Weekend vs weekday patterns
            is_weekend = current_date.weekday() >= 5
            daily_transactions = transactions_per_day
            
            if is_weekend:
                daily_transactions = random.randint(1, transactions_per_day + 2)
            else:
                daily_transactions = random.randint(1, transactions_per_day)
            
            for _ in range(daily_transactions):
                # 70% chance to use preferred categories
                if random.random() < 0.7:
                    category = random.choice(preferred_categories)
                    merchant_info = get_random_merchant(category)
                else:
                    merchant_info = get_random_merchant()
                
                transaction = self.generate_transaction(card, current_date, merchant_info)
                transactions.append(transaction)
        
        return transactions
    
    def export_to_csv(self, transactions: pd.DataFrame, filename: str):
        """Export transactions to CSV file."""
        transactions.to_csv(filename, index=False)
        print(f"Exported {len(transactions)} transactions to {filename}")
    
    def export_to_json(self, transactions: pd.DataFrame, filename: str):
        """Export transactions to JSON file."""
        transactions.to_json(filename, orient='records', date_format='iso', indent=2)
        print(f"Exported {len(transactions)} transactions to {filename}")


# Alias for easier imports
TransactionGenerator = ModernTransactionGenerator

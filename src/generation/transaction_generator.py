"""
Modern transaction generator with enhanced capabilities and modular design.
"""

import random
import pandas as pd
import uuid
from datetime import datetime, timedelta
from faker import Faker
from typing import List, Dict, Any, Optional
from decimal import Decimal

from ..core.models import Transaction, CreditCard, AnomalyType
from ..core.config import GenerationConfig
from .anomaly_injector import AnomalyInjector


class TransactionGenerator:
    """
    Modern transaction generator with enhanced capabilities.
    
    Features:
    - Configurable generation parameters
    - Realistic transaction patterns
    - Advanced anomaly injection
    - Multiple output formats
    - Performance optimized for large datasets
    """
    
    def __init__(self, config: Optional[GenerationConfig] = None, seed: Optional[int] = None):
        """Initialize the transaction generator."""
        self.config = config or GenerationConfig()
        
        if seed:
            random.seed(seed)
            Faker.seed(seed)
        
        self.fake = Faker('en_US')
        self.anomaly_injector = AnomalyInjector(config)
        
        # Card types and their number patterns
        self.card_types = {
            'Visa': {'prefix': '4', 'length': 16},
            'Mastercard': {'prefix': '5', 'length': 16},
            'American Express': {'prefix': '3', 'length': 15},
            'Discover': {'prefix': '6', 'length': 16}
        }
        
        # Merchant categories with realistic patterns
        self.merchant_categories = {
            'grocery_stores': {
                'names': [
                    'Fresh Market', 'SuperSave Grocery', 'Green Valley Foods',
                    'City Market', 'Organic Plus', 'Corner Store', 'FoodMart',
                    'Whole Foods Market', "Trader Joe's", 'Safeway'
                ],
                'amount_range': (5.0, 150.0),
                'peak_hours': [8, 9, 17, 18, 19]
            },
            'gas_stations': {
                'names': [
                    'Shell Station', 'Exxon Mobil', 'BP Gas', 'Chevron',
                    'Speedway', 'Circle K', 'Wawa', '7-Eleven', 'QuikTrip'
                ],
                'amount_range': (20.0, 100.0),
                'peak_hours': [7, 8, 17, 18]
            },
            'restaurants': {
                'names': [
                    "McDonald's", "Burger King", "Subway", "Pizza Hut",
                    "Starbucks", "Chipotle", "Olive Garden", "Applebee's",
                    "Local Bistro", "Corner Cafe", "Food Truck", "Diner"
                ],
                'amount_range': (8.0, 85.0),
                'peak_hours': [12, 13, 18, 19, 20]
            },
            'retail_stores': {
                'names': [
                    'Walmart', 'Target', 'Best Buy', 'Home Depot',
                    'Macy\'s', 'Nordstrom', 'REI', 'GameStop',
                    'Local Shop', 'Boutique Store'
                ],
                'amount_range': (15.0, 300.0),
                'peak_hours': [14, 15, 16, 19, 20]
            },
            'online_purchases': {
                'names': [
                    'Amazon', 'eBay', 'Etsy', 'PayPal',
                    'Apple Store', 'Google Play', 'Netflix', 'Spotify'
                ],
                'amount_range': (5.0, 200.0),
                'peak_hours': [14, 15, 16, 20, 21, 22]
            },
            'entertainment': {
                'names': [
                    'Movie Theater', 'Concert Hall', 'Sports Arena',
                    'Bowling Alley', 'Arcade', 'Theme Park'
                ],
                'amount_range': (12.0, 150.0),
                'peak_hours': [18, 19, 20, 21]
            },
            'healthcare': {
                'names': [
                    'City Hospital', 'Medical Clinic', 'Pharmacy',
                    'Dental Office', 'Eye Care Center', 'Urgent Care'
                ],
                'amount_range': (25.0, 500.0),
                'peak_hours': [9, 10, 11, 14, 15, 16]
            },
            'travel': {
                'names': [
                    'Hilton Hotel', 'Marriott', 'Budget Inn',
                    'Enterprise', 'Hertz', 'Airlines', 'Hotels.com'
                ],
                'amount_range': (50.0, 800.0),
                'peak_hours': [6, 7, 8, 14, 15, 16]
            }
        }
        
        # Locations for realistic geographic distribution
        self.locations = [
            ('New York', 'NY', 'USA'),
            ('Los Angeles', 'CA', 'USA'),
            ('Chicago', 'IL', 'USA'),
            ('Houston', 'TX', 'USA'),
            ('Phoenix', 'AZ', 'USA'),
            ('Philadelphia', 'PA', 'USA'),
            ('San Antonio', 'TX', 'USA'),
            ('San Diego', 'CA', 'USA'),
            ('Dallas', 'TX', 'USA'),
            ('San Jose', 'CA', 'USA'),
            ('Austin', 'TX', 'USA'),
            ('Jacksonville', 'FL', 'USA'),
            ('Fort Worth', 'TX', 'USA'),
            ('Columbus', 'OH', 'USA'),
            ('Charlotte', 'NC', 'USA'),
            ('San Francisco', 'CA', 'USA'),
            ('Indianapolis', 'IN', 'USA'),
            ('Seattle', 'WA', 'USA'),
            ('Denver', 'CO', 'USA'),
            ('Boston', 'MA', 'USA'),
            ('Nashville', 'TN', 'USA'),
            ('Detroit', 'MI', 'USA'),
            ('Portland', 'OR', 'USA'),
            ('Memphis', 'TN', 'USA'),
            ('Las Vegas', 'NV', 'USA'),
            ('Louisville', 'KY', 'USA'),
            ('Miami', 'FL', 'USA'),
            ('Atlanta', 'GA', 'USA'),
            ('Tampa', 'FL', 'USA'),
            ('Orlando', 'FL', 'USA')
        ]
    
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
            number=card_number,
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
    
    def _get_random_merchant(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Get random merchant information."""
        if category is None:
            category = random.choice(list(self.merchant_categories.keys()))
        
        if category not in self.merchant_categories:
            category = 'retail_stores'  # Default fallback
        
        category_info = self.merchant_categories[category]
        
        return {
            'name': random.choice(category_info['names']),
            'category': category,
            'amount_range': category_info['amount_range'],
            'peak_hours': category_info['peak_hours']
        }
    
    def _get_random_location(self) -> tuple:
        """Get random location (city, state, country)."""
        return random.choice(self.locations)
    
    def generate_transaction(self, 
                           card: CreditCard, 
                           transaction_date: datetime,
                           merchant_info: Optional[Dict[str, Any]] = None) -> Transaction:
        """Generate a single transaction."""
        
        if not merchant_info:
            merchant_info = self._get_random_merchant()
        
        # Generate amount based on merchant category
        amount_range = merchant_info['amount_range']
        amount = Decimal(str(round(random.uniform(amount_range[0], amount_range[1]), 2)))
        
        # Add some realistic amount patterns
        if random.random() < 0.3:  # 30% chance of round amounts
            amount = Decimal(str(round(float(amount) / 5) * 5))  # Round to nearest 5
        
        # Get location
        city, state, country = self._get_random_location()
        
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
            card_number=card.number,
            amount=float(amount),
            transaction_date=transaction_date,
            merchant_name=merchant_info['name'],
            merchant_category=merchant_info['category'],
            city=city,
            state=state,
            country=country,
            is_anomaly=False,
            anomaly_type=None
        )
    
    def generate_transactions(self, 
                            count: Optional[int] = None,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            num_cards: Optional[int] = None) -> pd.DataFrame:
        """
        Generate a dataset of transactions.
        
        Args:
            count: Number of transactions to generate (defaults to config)
            start_date: Start date for transactions (defaults to 30 days ago)
            end_date: End date for transactions (defaults to now)
            num_cards: Number of unique cards (defaults to count/20)
            
        Returns:
            DataFrame with generated transactions
        """
        count = count or self.config.default_transaction_count
        
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
            transactions.append(transaction)
        
        # Convert to DataFrame
        df = pd.DataFrame([t.to_dict() for t in transactions])
        df = df.sort_values('transaction_date').reset_index(drop=True)
        
        # Inject anomalies if configured
        if self.config.default_anomaly_rate > 0:
            df = self.anomaly_injector.inject_anomalies(df, self.config.default_anomaly_rate)
        
        print(f"Generated {len(df)} transactions")
        if self.config.default_anomaly_rate > 0:
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
            list(self.merchant_categories.keys()), 
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
                    merchant_info = self._get_random_merchant(category)
                else:
                    merchant_info = self._get_random_merchant()
                
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

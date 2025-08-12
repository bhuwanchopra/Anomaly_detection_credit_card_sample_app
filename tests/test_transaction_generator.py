"""
Unit tests for transaction generator
"""

import unittest
import sys
import os
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the modules we need for testing  
import transaction_generator
import data_models
from transaction_generator import TransactionGenerator
from data_models import CreditCard


class TestTransactionGenerator(unittest.TestCase):
    """Test cases for TransactionGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = TransactionGenerator(seed=42)  # Fixed seed for reproducible tests
    
    def test_generate_credit_card(self):
        """Test credit card generation."""
        card = self.generator.generate_credit_card()
        
        self.assertIsInstance(card, CreditCard)
        self.assertIsInstance(card.card_number, str)
        self.assertGreaterEqual(len(card.card_number), 15)
        self.assertLessEqual(len(card.card_number), 16)
        self.assertIn(card.card_type, ['Visa', 'Mastercard', 'American Express', 'Discover'])
    
    def test_luhn_check_digit(self):
        """Test Luhn algorithm implementation."""
        # Test with known valid card number (without check digit)
        test_number = "424242424242424"  # Visa test number
        check_digit = self.generator._calculate_luhn_check_digit(test_number)
        
        # The complete number should pass Luhn validation
        complete_number = test_number + check_digit
        self.assertTrue(self._is_valid_luhn(complete_number))
    
    def _is_valid_luhn(self, card_number):
        """Validate card number using Luhn algorithm."""
        digits = [int(d) for d in card_number[::-1]]
        for i in range(1, len(digits), 2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
        return sum(digits) % 10 == 0
    
    def test_generate_transaction(self):
        """Test single transaction generation."""
        card = self.generator.generate_credit_card()
        transaction_date = datetime.now()
        
        transaction = self.generator.generate_transaction(card, transaction_date)
        
        self.assertIsNotNone(transaction.transaction_id)
        self.assertEqual(transaction.card_number, card.masked_number())
        self.assertEqual(transaction.cardholder_name, card.cardholder_name)
        self.assertGreater(transaction.amount, 0)
        self.assertEqual(transaction.currency, 'USD')
        self.assertFalse(transaction.is_anomaly)
    
    def test_generate_transactions_count(self):
        """Test that correct number of transactions are generated."""
        count = 50
        df = self.generator.generate_transactions(count=count)
        
        self.assertEqual(len(df), count)
    
    def test_generate_transactions_with_anomalies(self):
        """Test generation with anomalies."""
        count = 100
        anomaly_rate = 0.1
        df = self.generator.generate_transactions(count=count, anomaly_rate=anomaly_rate)
        
        anomaly_count = df['is_anomaly'].sum()
        expected_anomalies = int(count * anomaly_rate)
        
        # Allow for some variance due to random sampling
        self.assertGreaterEqual(anomaly_count, expected_anomalies - 2)
        self.assertLessEqual(anomaly_count, expected_anomalies + 2)
    
    def test_date_range(self):
        """Test that transactions fall within specified date range."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        df = self.generator.generate_transactions(
            count=20,
            start_date=start_date,
            end_date=end_date
        )
        
        # Convert transaction_date to datetime if it's string
        dates = pd.to_datetime(df['transaction_date'])
        
        self.assertTrue(all(dates >= start_date))
        self.assertTrue(all(dates <= end_date))
    
    def test_merchant_categories(self):
        """Test that valid merchant categories are used."""
        df = self.generator.generate_transactions(count=50)
        
        valid_categories = [
            'grocery', 'gas_station', 'restaurant', 'retail', 
            'online', 'entertainment', 'healthcare', 'travel'
        ]
        
        # All transactions should have valid categories (unless anomalous)
        normal_transactions = df[df['is_anomaly'] == False]
        categories = normal_transactions['merchant_category'].unique()
        
        for category in categories:
            self.assertIn(category, valid_categories)


if __name__ == '__main__':
    # Import pandas here to avoid import error in tests
    try:
        import pandas as pd
        unittest.main()
    except ImportError:
        print("Pandas not installed. Install requirements first: pip install -r requirements.txt")
        sys.exit(1)

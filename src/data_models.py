"""
Data models for credit card transactions
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Transaction:
    """Credit card transaction data model."""
    transaction_id: str
    card_number: str
    cardholder_name: str
    merchant_name: str
    merchant_category: str
    amount: float
    currency: str
    transaction_date: datetime
    city: str
    state: str
    country: str
    is_anomaly: bool
    anomaly_type: Optional[str] = None
    
    def to_dict(self):
        """Convert transaction to dictionary."""
        return {
            'transaction_id': self.transaction_id,
            'card_number': self.card_number,
            'cardholder_name': self.cardholder_name,
            'merchant_name': self.merchant_name,
            'merchant_category': self.merchant_category,
            'amount': self.amount,
            'currency': self.currency,
            'transaction_date': self.transaction_date,  # Keep as datetime object
            'city': self.city,
            'state': self.state,
            'country': self.country,
            'is_anomaly': self.is_anomaly,
            'anomaly_type': self.anomaly_type
        }


@dataclass
class CreditCard:
    """Credit card data model."""
    card_number: str
    cardholder_name: str
    expiry_month: int
    expiry_year: int
    cvv: str
    card_type: str
    
    def masked_number(self):
        """Return masked card number."""
        return f"****-****-****-{self.card_number[-4:]}"


@dataclass
class Merchant:
    """Merchant data model."""
    name: str
    category: str
    city: str
    state: str
    country: str

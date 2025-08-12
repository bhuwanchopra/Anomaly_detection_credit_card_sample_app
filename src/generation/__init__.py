"""
Transaction generation module.
"""

try:
    from .modern_generator import TransactionGenerator
except ImportError as e:
    class TransactionGenerator:
        def __init__(self, seed=None):
            pass
        def generate_transactions(self, count=100, **kwargs):
            import pandas as pd
            return pd.DataFrame({'message': ['Please use the legacy transaction_generator.py']})

__all__ = ['TransactionGenerator']

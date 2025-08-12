"""
Integration tests for the complete anomaly detection pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.core.config import SystemConfig, FeatureConfig, DetectionConfig
from src.core.models import AnomalyType, DetectionMethod
from src.features.engineering import FeaturePipeline


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline integration."""
    
    def setup_method(self):
        """Setup test data and configuration."""
        self.config = SystemConfig()
        self.config.detection.contamination = 0.1
        
        # Create synthetic transaction data
        np.random.seed(42)
        n_transactions = 1000
        
        # Generate dates over 30 days
        start_date = datetime(2025, 1, 1)
        dates = []
        for i in range(n_transactions):
            # Most transactions during business hours
            if np.random.random() < 0.1:  # 10% unusual time
                hour = np.random.choice([2, 3, 4, 5])  # Unusual hours
            else:
                hour = np.random.choice(range(9, 18))  # Business hours
            
            days_offset = np.random.randint(0, 30)
            minutes_offset = np.random.randint(0, 60)
            
            date = start_date + timedelta(days=days_offset, hours=hour, minutes=minutes_offset)
            dates.append(date)
        
        # Generate amounts - mostly normal with some outliers
        amounts = []
        for i in range(n_transactions):
            if np.random.random() < 0.05:  # 5% unusual amounts
                amount = np.random.choice([5000, 7500, 10000])  # High amounts
            else:
                amount = np.random.lognormal(4, 1)  # Normal distribution
            amounts.append(max(1.0, amount))
        
        # Generate cards
        cards = [f"{np.random.randint(1000, 9999)}{np.random.randint(1000, 9999)}{np.random.randint(1000, 9999)}{np.random.randint(1000, 9999)}" 
                for _ in range(n_transactions)]
        
        # Generate merchant data
        categories = ['retail', 'restaurant', 'gas', 'grocery', 'online']
        merchant_categories = np.random.choice(categories, n_transactions)
        
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Boston']
        city_list = np.random.choice(cities, n_transactions)
        
        states = ['NY', 'CA', 'IL', 'TX', 'MA']
        state_list = np.random.choice(states, n_transactions)
        
        self.df = pd.DataFrame({
            'transaction_id': [f'TXN{i:06d}' for i in range(n_transactions)],
            'transaction_date': dates,
            'amount': amounts,
            'card_number': cards,
            'merchant_category': merchant_categories,
            'merchant_name': [f'Merchant {i}' for i in range(n_transactions)],
            'city': city_list,
            'state': state_list,
            'country': ['USA'] * n_transactions
        })
        
        # Add ground truth anomalies for testing
        anomaly_indices = np.random.choice(n_transactions, size=50, replace=False)
        self.df['is_anomaly'] = False
        self.df.loc[anomaly_indices, 'is_anomaly'] = True
    
    def test_feature_pipeline_integration(self):
        """Test feature engineering pipeline integration."""
        pipeline = FeaturePipeline(self.config.features)
        
        # Should handle the complete pipeline
        features = pipeline.fit_transform(self.df)
        
        # Verify output
        assert len(features) == len(self.df)
        assert len(features.columns) > 0
        
        # Check for expected feature categories
        feature_names = pipeline.get_feature_names()
        
        # Time features
        time_features = [f for f in feature_names if f in ['hour', 'is_unusual_hour', 'time_risk_score']]
        assert len(time_features) > 0
        
        # Amount features
        amount_features = [f for f in feature_names if 'amount' in f]
        assert len(amount_features) > 0
        
        # No missing values in critical features
        assert not features.isnull().any().any()
    
    def test_configuration_integration(self):
        """Test configuration system integration."""
        # Test custom configuration
        custom_config = SystemConfig()
        custom_config.features.enable_frequency_features = False
        custom_config.detection.contamination = 0.05
        
        pipeline = FeaturePipeline(custom_config.features)
        features = pipeline.fit_transform(self.df)
        
        # Should not have frequency features
        feature_names = pipeline.get_feature_names()
        frequency_features = [f for f in feature_names if 'transaction' in f and 'last' in f]
        assert len(frequency_features) == 0
    
    def test_large_dataset_handling(self):
        """Test handling of larger datasets."""
        # Create a larger dataset
        large_df = pd.concat([self.df] * 10, ignore_index=True)
        large_df['transaction_id'] = [f'TXN{i:07d}' for i in range(len(large_df))]
        
        pipeline = FeaturePipeline(self.config.features)
        
        # Should handle large dataset without errors
        features = pipeline.fit_transform(large_df)
        
        assert len(features) == len(large_df)
        assert not features.isnull().any().any()
    
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        # Create dataframe with some missing values
        test_df = self.df.copy()
        
        # Add some missing merchant categories
        missing_indices = np.random.choice(len(test_df), size=10, replace=False)
        test_df.loc[missing_indices, 'merchant_category'] = np.nan
        
        pipeline = FeaturePipeline(self.config.features)
        
        # Should handle missing data gracefully
        features = pipeline.fit_transform(test_df)
        
        assert len(features) == len(test_df)
    
    def test_datetime_format_variations(self):
        """Test handling of different datetime formats."""
        test_df = self.df.copy()
        
        # Convert some dates to string format
        test_df['transaction_date'] = test_df['transaction_date'].astype(str)
        
        pipeline = FeaturePipeline(self.config.features)
        
        # Should handle string datetime conversion
        features = pipeline.fit_transform(test_df)
        
        assert len(features) == len(test_df)
        assert 'hour' in pipeline.get_feature_names()


class TestErrorHandling:
    """Test error handling across the pipeline."""
    
    def test_invalid_configuration(self):
        """Test handling of invalid configurations."""
        config = SystemConfig()
        config.generation.default_anomaly_rate = 2.0  # Invalid
        
        with pytest.raises(ValueError):
            config.validate()
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Very small dataset
        small_df = pd.DataFrame({
            'transaction_date': [datetime.now()],
            'amount': [100],
            'card_number': ['1234567890123456'],
            'merchant_category': ['retail'],
            'city': ['New York'],
            'state': ['NY'],
            'country': ['USA']
        })
        
        config = FeatureConfig()
        pipeline = FeaturePipeline(config)
        
        # Should handle small dataset
        features = pipeline.fit_transform(small_df)
        assert len(features) == 1


class TestPerformanceRequirements:
    """Test performance requirements and benchmarks."""
    
    def test_feature_extraction_performance(self):
        """Test feature extraction performance on medium dataset."""
        # Create medium-sized dataset
        n_transactions = 10000
        dates = [datetime.now() - timedelta(days=np.random.randint(0, 30)) for _ in range(n_transactions)]
        
        df = pd.DataFrame({
            'transaction_date': dates,
            'amount': np.random.lognormal(4, 1, n_transactions),
            'card_number': [f'{np.random.randint(1000, 9999)}'*4 for _ in range(n_transactions)],
            'merchant_category': np.random.choice(['retail', 'restaurant', 'gas'], n_transactions),
            'city': np.random.choice(['New York', 'Boston', 'Chicago'], n_transactions),
            'state': np.random.choice(['NY', 'MA', 'IL'], n_transactions),
            'country': ['USA'] * n_transactions
        })
        
        config = FeatureConfig()
        pipeline = FeaturePipeline(config)
        
        import time
        start_time = time.time()
        features = pipeline.fit_transform(df)
        execution_time = time.time() - start_time
        
        # Should complete in reasonable time (< 30 seconds for 10K transactions)
        assert execution_time < 30
        assert len(features) == n_transactions
        
        print(f"Feature extraction for {n_transactions} transactions: {execution_time:.2f} seconds")
    
    def test_memory_usage(self):
        """Test memory usage doesn't explode with larger datasets."""
        # This is a basic test - in practice you'd use memory profiling tools
        n_transactions = 50000
        
        df = pd.DataFrame({
            'transaction_date': [datetime.now()] * n_transactions,
            'amount': np.random.random(n_transactions) * 1000,
            'card_number': ['1234567890123456'] * n_transactions,
            'merchant_category': ['retail'] * n_transactions,
            'city': ['New York'] * n_transactions,
            'state': ['NY'] * n_transactions,
            'country': ['USA'] * n_transactions
        })
        
        config = FeatureConfig()
        pipeline = FeaturePipeline(config)
        
        # Should not raise memory errors
        features = pipeline.fit_transform(df)
        assert len(features) == n_transactions


if __name__ == "__main__":
    pytest.main([__file__])

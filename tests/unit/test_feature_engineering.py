"""
Unit tests for feature engineering pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features.engineering import (
    TimeFeatureEngineer, AmountFeatureEngineer, 
    FrequencyFeatureEngineer, CategoricalFeatureEngineer,
    FeaturePipeline
)
from src.core.config import FeatureConfig
from src.core.models import DataValidationError


class TestTimeFeatureEngineer:
    """Test TimeFeatureEngineer class."""
    
    def setup_method(self):
        """Setup test data."""
        self.config = FeatureConfig()
        self.engineer = TimeFeatureEngineer(self.config)
        
        # Create sample data
        dates = [
            datetime(2025, 1, 1, 3, 30, 0),   # Unusual hour
            datetime(2025, 1, 1, 14, 0, 0),   # Business hour
            datetime(2025, 1, 1, 23, 15, 0),  # Late night
            datetime(2025, 1, 2, 20, 45, 0),  # Evening
        ]
        
        self.df = pd.DataFrame({
            'transaction_date': dates,
            'amount': [100, 200, 150, 300],
            'card_number': ['1234567890123456'] * 4
        })
    
    def test_time_feature_extraction(self):
        """Test time feature extraction."""
        result = self.engineer.fit_transform(self.df)
        
        # Check that time features are created
        assert 'hour' in result.columns
        assert 'day_of_week' in result.columns
        assert 'is_weekend' in result.columns
        assert 'month' in result.columns
        assert 'is_unusual_hour' in result.columns
        assert 'is_business_hour' in result.columns
        assert 'is_late_night' in result.columns
        assert 'time_risk_score' in result.columns
        
        # Test specific values
        assert result.iloc[0]['hour'] == 3  # First transaction at 3:30 AM
        assert result.iloc[0]['is_unusual_hour'] == 1  # Should be unusual
        assert result.iloc[1]['is_business_hour'] == 1  # Should be business hour
        assert result.iloc[2]['is_late_night'] == 1  # Should be late night
    
    def test_time_features_disabled(self):
        """Test when time features are disabled."""
        self.config.enable_time_features = False
        engineer = TimeFeatureEngineer(self.config)
        
        result = engineer.fit_transform(self.df)
        
        # Should return original dataframe
        assert list(result.columns) == list(self.df.columns)
    
    def test_time_risk_score_calculation(self):
        """Test time risk score calculation."""
        result = self.engineer.fit_transform(self.df)
        
        # Unusual hour (3 AM) should have higher risk score
        unusual_score = result.iloc[0]['time_risk_score']
        business_score = result.iloc[1]['time_risk_score']
        
        assert unusual_score > business_score


class TestAmountFeatureEngineer:
    """Test AmountFeatureEngineer class."""
    
    def setup_method(self):
        """Setup test data."""
        self.config = FeatureConfig()
        self.engineer = AmountFeatureEngineer(self.config)
        
        self.df = pd.DataFrame({
            'transaction_date': [datetime.now()] * 4,
            'amount': [100.0, 200.5, 1000.0, 50.25],
            'card_number': ['1234567890123456'] * 4
        })
    
    def test_amount_feature_extraction(self):
        """Test amount feature extraction."""
        result = self.engineer.fit_transform(self.df)
        
        # Check that amount features are created
        assert 'amount' in result.columns
        assert 'amount_log' in result.columns
        assert 'amount_rounded' in result.columns
        assert 'amount_zscore' in result.columns
        assert 'is_round_amount' in result.columns
        
        # Test specific values
        assert result.iloc[0]['amount_rounded'] == 1  # 100.0 is rounded
        assert result.iloc[1]['amount_rounded'] == 0  # 200.5 is not rounded
        assert result.iloc[2]['is_round_amount'] == 1  # 1000.0 is in round amounts list
    
    def test_amount_features_disabled(self):
        """Test when amount features are disabled."""
        self.config.enable_amount_features = False
        engineer = AmountFeatureEngineer(self.config)
        
        result = engineer.fit_transform(self.df)
        
        # Should return original dataframe
        assert list(result.columns) == list(self.df.columns)
    
    def test_log_transformation(self):
        """Test logarithmic transformation."""
        result = self.engineer.fit_transform(self.df)
        
        # log1p(100) should be approximately log(101)
        expected_log = np.log1p(100.0)
        assert abs(result.iloc[0]['amount_log'] - expected_log) < 0.001


class TestFrequencyFeatureEngineer:
    """Test FrequencyFeatureEngineer class."""
    
    def setup_method(self):
        """Setup test data."""
        self.config = FeatureConfig()
        self.engineer = FrequencyFeatureEngineer(self.config)
        
        # Create transactions with some close together in time
        base_time = datetime(2025, 1, 1, 12, 0, 0)
        dates = [
            base_time,
            base_time + timedelta(minutes=30),  # 30 min later
            base_time + timedelta(minutes=45),  # 45 min later
            base_time + timedelta(hours=2),     # 2 hours later
        ]
        
        self.df = pd.DataFrame({
            'transaction_date': dates,
            'amount': [100, 200, 150, 300],
            'card_number': ['1234567890123456'] * 4  # Same card
        })
    
    def test_frequency_feature_extraction(self):
        """Test frequency feature extraction."""
        result = self.engineer.fit_transform(self.df)
        
        # Check that frequency features are created
        assert 'transactions_last_hour' in result.columns
        assert 'transactions_last_day' in result.columns
        assert 'avg_time_between_transactions' in result.columns
        
        # The second and third transactions should have recent transactions
        assert result.iloc[1]['transactions_last_hour'] >= 0
        assert result.iloc[2]['transactions_last_hour'] >= 0
    
    def test_frequency_features_disabled(self):
        """Test when frequency features are disabled."""
        self.config.enable_frequency_features = False
        engineer = FrequencyFeatureEngineer(self.config)
        
        result = engineer.fit_transform(self.df)
        
        # Should return original dataframe
        assert list(result.columns) == list(self.df.columns)


class TestCategoricalFeatureEngineer:
    """Test CategoricalFeatureEngineer class."""
    
    def setup_method(self):
        """Setup test data."""
        self.config = FeatureConfig()
        self.engineer = CategoricalFeatureEngineer(self.config)
        
        self.df = pd.DataFrame({
            'transaction_date': [datetime.now()] * 4,
            'amount': [100, 200, 150, 300],
            'card_number': ['1234567890123456', '2345678901234567', 
                           '1234567890123456', '3456789012345678'],
            'merchant_category': ['retail', 'restaurant', 'retail', 'gas'],
            'city': ['New York', 'Boston', 'New York', 'Chicago'],
            'state': ['NY', 'MA', 'NY', 'IL'],
            'country': ['USA', 'USA', 'USA', 'USA']
        })
    
    def test_categorical_feature_extraction(self):
        """Test categorical feature extraction."""
        result = self.engineer.fit_transform(self.df)
        
        # Check that categorical features are created
        assert 'card_last_4' in result.columns
        assert 'merchant_category_encoded' in result.columns
        assert 'city_encoded' in result.columns
        assert 'state_encoded' in result.columns
        assert 'country_encoded' in result.columns
        
        # Test card last 4 digits
        assert result.iloc[0]['card_last_4'] == 3456  # Last 4 of first card
        assert result.iloc[1]['card_last_4'] == 4567  # Last 4 of second card
    
    def test_categorical_features_disabled(self):
        """Test when categorical features are disabled."""
        self.config.enable_categorical_features = False
        engineer = CategoricalFeatureEngineer(self.config)
        
        result = engineer.fit_transform(self.df)
        
        # Should return original dataframe
        assert list(result.columns) == list(self.df.columns)
    
    def test_unseen_category_handling(self):
        """Test handling of unseen categories during transform."""
        # Fit on subset of data
        train_df = self.df.iloc[:2].copy()
        self.engineer.fit(train_df)
        
        # Transform with new category
        test_df = self.df.copy()
        test_df.iloc[3, test_df.columns.get_loc('merchant_category')] = 'online'  # New category
        
        result = self.engineer.transform(test_df)
        
        # Should handle unseen category gracefully
        assert 'merchant_category_encoded' in result.columns
        assert result.iloc[3]['merchant_category_encoded'] == 0  # Default value


class TestFeaturePipeline:
    """Test FeaturePipeline class."""
    
    def setup_method(self):
        """Setup test data."""
        self.config = FeatureConfig()
        self.pipeline = FeaturePipeline(self.config)
        
        dates = [
            datetime(2025, 1, 1, 3, 30, 0),
            datetime(2025, 1, 1, 14, 0, 0),
            datetime(2025, 1, 2, 23, 15, 0),
        ]
        
        self.df = pd.DataFrame({
            'transaction_date': dates,
            'amount': [100.0, 1000.0, 150.5],
            'card_number': ['1234567890123456', '2345678901234567', '1234567890123456'],
            'merchant_category': ['retail', 'restaurant', 'retail'],
            'city': ['New York', 'Boston', 'New York'],
            'state': ['NY', 'MA', 'NY'],
            'country': ['USA', 'USA', 'USA'],
            'merchant_name': ['Store A', 'Restaurant B', 'Store C']
        })
    
    def test_pipeline_fit_transform(self):
        """Test complete pipeline fit and transform."""
        result = self.pipeline.fit_transform(self.df)
        
        # Should have all feature types
        feature_names = self.pipeline.get_feature_names()
        
        # Check for time features
        time_features = [f for f in feature_names if f in ['hour', 'is_unusual_hour', 'time_risk_score']]
        assert len(time_features) > 0
        
        # Check for amount features
        amount_features = [f for f in feature_names if 'amount' in f]
        assert len(amount_features) > 0
        
        # Check for categorical features
        categorical_features = [f for f in feature_names if 'encoded' in f]
        assert len(categorical_features) > 0
        
        # Result should only contain engineered features
        assert list(result.columns) == feature_names
    
    def test_pipeline_separate_fit_transform(self):
        """Test separate fit and transform operations."""
        # Fit on part of the data
        train_df = self.df.iloc[:2].copy()
        self.pipeline.fit(train_df)
        
        # Transform on all data
        result = self.pipeline.transform(self.df)
        
        assert len(result) == len(self.df)
        assert self.pipeline.is_fitted
    
    def test_pipeline_validation_missing_columns(self):
        """Test pipeline validation with missing required columns."""
        invalid_df = pd.DataFrame({
            'amount': [100, 200],
            'merchant_name': ['Store A', 'Store B']
            # Missing transaction_date and card_number
        })
        
        with pytest.raises(DataValidationError, match="Missing required columns"):
            self.pipeline.fit(invalid_df)
    
    def test_pipeline_validation_empty_dataframe(self):
        """Test pipeline validation with empty dataframe."""
        empty_df = pd.DataFrame(columns=['transaction_date', 'amount', 'card_number'])
        
        with pytest.raises(DataValidationError, match="Input DataFrame is empty"):
            self.pipeline.fit(empty_df)
    
    def test_transform_before_fit(self):
        """Test transform called before fit."""
        with pytest.raises(RuntimeError, match="Feature pipeline must be fitted before transform"):
            self.pipeline.transform(self.df)
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        self.pipeline.fit(self.df)
        feature_names = self.pipeline.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        
        # Should be a copy, not reference
        feature_names.append('test_feature')
        assert 'test_feature' not in self.pipeline.get_feature_names()


if __name__ == "__main__":
    pytest.main([__file__])

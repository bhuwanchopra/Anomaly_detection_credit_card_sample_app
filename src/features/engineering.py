"""
Modular feature engineering for anomaly detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from ..core.models import DataValidationError
from ..core.config import FeatureConfig


class BaseFeatureEngineer:
    """Base class for feature engineering."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.feature_names: List[str] = []
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'BaseFeatureEngineer':
        """Fit the feature engineer to the data."""
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        if not self.is_fitted:
            raise RuntimeError("Feature engineer must be fitted before transform")
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data."""
        return self.fit(df).transform(df)


class TimeFeatureEngineer(BaseFeatureEngineer):
    """Engineer time-based features."""
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features."""
        if not self.config.enable_time_features:
            return df
        
        features_df = df.copy()
        
        # Ensure datetime conversion
        if not pd.api.types.is_datetime64_any_dtype(features_df['transaction_date']):
            features_df['transaction_date'] = pd.to_datetime(features_df['transaction_date'])
        
        # Basic time features
        features_df['hour'] = features_df['transaction_date'].dt.hour
        features_df['day_of_week'] = features_df['transaction_date'].dt.dayofweek
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        features_df['month'] = features_df['transaction_date'].dt.month
        
        # Enhanced time-based features
        features_df['is_unusual_hour'] = features_df['hour'].isin(self.config.unusual_hours).astype(int)
        features_df['is_business_hour'] = features_df['hour'].between(
            self.config.business_hours_start, 
            self.config.business_hours_end
        ).astype(int)
        features_df['is_late_night'] = features_df['hour'].isin(self.config.late_night_hours).astype(int)
        
        # Composite time risk score
        features_df['time_risk_score'] = (
            features_df['is_unusual_hour'] * 3 +  # Heavy weight for unusual hours
            features_df['is_late_night'] * 1.5 +  # Moderate weight for late night
            (1 - features_df['is_business_hour']) * 0.5  # Light weight for non-business hours
        )
        
        # Update feature names
        time_features = [
            'hour', 'day_of_week', 'is_weekend', 'month',
            'is_unusual_hour', 'is_business_hour', 'is_late_night', 'time_risk_score'
        ]
        self.feature_names.extend(time_features)
        
        return features_df


class AmountFeatureEngineer(BaseFeatureEngineer):
    """Engineer amount-based features."""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.amount_scaler = StandardScaler()
    
    def fit(self, df: pd.DataFrame) -> 'AmountFeatureEngineer':
        """Fit the amount scaler."""
        if self.config.enable_amount_features:
            self.amount_scaler.fit(df[['amount']])
        super().fit(df)
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract amount-based features."""
        if not self.config.enable_amount_features:
            return df
        
        features_df = df.copy()
        
        # Basic amount features
        features_df['amount_log'] = np.log1p(features_df['amount'])
        features_df['amount_rounded'] = (features_df['amount'] % 1 == 0).astype(int)
        
        # Z-score normalization
        features_df['amount_zscore'] = self.amount_scaler.transform(features_df[['amount']]).flatten()
        
        # Round amount detection
        features_df['is_round_amount'] = features_df['amount'].isin(self.config.round_amounts).astype(int)
        
        # Update feature names
        amount_features = ['amount', 'amount_log', 'amount_rounded', 'amount_zscore', 'is_round_amount']
        self.feature_names.extend(amount_features)
        
        return features_df


class FrequencyFeatureEngineer(BaseFeatureEngineer):
    """Engineer frequency-based features for detecting frequent transactions."""
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract frequency-based features."""
        if not self.config.enable_frequency_features:
            return df
        
        features_df = df.copy()
        
        # Sort by transaction date for temporal analysis
        features_df = features_df.sort_values('transaction_date').reset_index(drop=True)
        
        # Initialize frequency features
        features_df['transactions_last_hour'] = 0
        features_df['transactions_last_day'] = 0
        features_df['avg_time_between_transactions'] = 24.0  # Default 24 hours
        
        # For large datasets, use optimized approach
        if len(features_df) > 100000:
            features_df = self._add_frequency_features_optimized(features_df)
        else:
            features_df = self._add_frequency_features_detailed(features_df)
        
        # Update feature names
        frequency_features = ['transactions_last_hour', 'transactions_last_day', 'avg_time_between_transactions']
        self.feature_names.extend(frequency_features)
        
        return features_df
    
    def _add_frequency_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized frequency features for large datasets."""
        for card in df['card_number'].unique():
            card_mask = df['card_number'] == card
            card_data = df[card_mask].copy()
            
            if len(card_data) > 1:
                # Calculate time differences in hours
                time_diffs = card_data['transaction_date'].diff()
                time_diffs_hours = time_diffs.dt.total_seconds() / 3600
                avg_hours = time_diffs_hours.mean()
                
                # Set average time between transactions
                df.loc[card_mask, 'avg_time_between_transactions'] = avg_hours
                
                # Mark transactions that are very close in time
                close_transactions = time_diffs_hours < self.config.frequency_window_hours
                df.loc[card_data.index[close_transactions], 'transactions_last_hour'] = 1
                
                # Mark transactions that are part of many in a day
                daily_counts = card_data.groupby(card_data['transaction_date'].dt.date).size()
                for date, count in daily_counts.items():
                    if count > self.config.max_daily_transactions_threshold:
                        date_mask = card_data['transaction_date'].dt.date == date
                        df.loc[card_data.index[date_mask], 'transactions_last_day'] = count
        
        return df
    
    def _add_frequency_features_detailed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detailed frequency features for smaller datasets."""
        for card in df['card_number'].unique():
            card_mask = df['card_number'] == card
            card_transactions = df[card_mask].copy().reset_index()
            
            if len(card_transactions) > 1:
                card_times = pd.to_datetime(card_transactions['transaction_date'])
                time_diffs = card_times.diff().dt.total_seconds().fillna(3600)
                
                # For each transaction, count recent transactions
                for i in range(len(card_transactions)):
                    current_time = card_times.iloc[i]
                    orig_idx = card_transactions.iloc[i]['index']
                    
                    # Count transactions in last hour
                    hour_ago = current_time - pd.Timedelta(hours=self.config.frequency_window_hours)
                    recent_hour_count = sum(
                        (card_times >= hour_ago) & (card_times < current_time)
                    )
                    df.loc[orig_idx, 'transactions_last_hour'] = recent_hour_count
                    
                    # Count transactions in last day
                    day_ago = current_time - pd.Timedelta(days=self.config.frequency_window_days)
                    recent_day_count = sum(
                        (card_times >= day_ago) & (card_times < current_time)
                    )
                    df.loc[orig_idx, 'transactions_last_day'] = recent_day_count
                
                # Average time between transactions
                avg_time_diff = time_diffs.mean()
                orig_indices = card_transactions['index'].values
                df.loc[orig_indices, 'avg_time_between_transactions'] = avg_time_diff / 3600
        
        return df


class CategoricalFeatureEngineer(BaseFeatureEngineer):
    """Engineer categorical features."""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.label_encoders: Dict[str, LabelEncoder] = {}
    
    def fit(self, df: pd.DataFrame) -> 'CategoricalFeatureEngineer':
        """Fit label encoders for categorical features."""
        if self.config.enable_categorical_features:
            categorical_cols = ['merchant_category', 'city', 'state', 'country']
            for col in categorical_cols:
                if col in df.columns:
                    self.label_encoders[col] = LabelEncoder()
                    self.label_encoders[col].fit(df[col])
        
        super().fit(df)
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features."""
        if not self.config.enable_categorical_features:
            return df
        
        features_df = df.copy()
        
        # Card-based features
        features_df['card_last_4'] = features_df['card_number'].str[-4:].astype(int)
        
        # Categorical encoding
        categorical_features = ['card_last_4']
        for col, encoder in self.label_encoders.items():
            if col in features_df.columns:
                # Handle unseen categories
                mask = features_df[col].isin(encoder.classes_)
                features_df[f'{col}_encoded'] = 0  # Default for unseen categories
                if mask.any():
                    encoded_values = encoder.transform(features_df.loc[mask, col])
                    features_df.loc[mask, f'{col}_encoded'] = encoded_values
                categorical_features.append(f'{col}_encoded')
        
        # Update feature names
        self.feature_names.extend(categorical_features)
        
        return features_df


class FeaturePipeline:
    """Complete feature engineering pipeline."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.engineers = [
            TimeFeatureEngineer(config),
            AmountFeatureEngineer(config),
            FrequencyFeatureEngineer(config),
            CategoricalFeatureEngineer(config)
        ]
        self.feature_names: List[str] = []
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'FeaturePipeline':
        """Fit all feature engineers."""
        self._validate_input(df)
        
        current_df = df.copy()
        for engineer in self.engineers:
            engineer.fit(current_df)
            current_df = engineer.transform(current_df)
        
        # Collect all feature names
        self.feature_names = []
        for engineer in self.engineers:
            self.feature_names.extend(engineer.feature_names)
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data through the complete pipeline."""
        if not self.is_fitted:
            raise RuntimeError("Feature pipeline must be fitted before transform")
        
        self._validate_input(df)
        
        current_df = df.copy()
        for engineer in self.engineers:
            current_df = engineer.transform(current_df)
        
        # Select only the engineered features
        if self.feature_names:
            current_df = current_df[self.feature_names]
        
        return current_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data."""
        return self.fit(df).transform(df)
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input data."""
        required_columns = ['transaction_date', 'amount', 'card_number']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            raise DataValidationError("Input DataFrame is empty")
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names.copy()

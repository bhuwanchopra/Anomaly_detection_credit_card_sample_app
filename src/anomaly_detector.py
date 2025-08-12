"""
Anomaly detection using dimensionality reduction and machine learning techniques
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """Advanced anomaly detection using multiple machine learning techniques."""
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers in the data
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.pca = None
        self.isolation_forest = None
        self.lof = None
        self.dbscan = None
        self.feature_names = []
        self.results = {}
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for anomaly detection.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with engineered features
        """
        features_df = df.copy()
        
        # Convert transaction_date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(features_df['transaction_date']):
            features_df['transaction_date'] = pd.to_datetime(features_df['transaction_date'])
        
        # Sort by transaction date for temporal features
        features_df = features_df.sort_values('transaction_date').reset_index(drop=True)
        
        # Extract enhanced time-based features
        features_df['hour'] = features_df['transaction_date'].dt.hour
        features_df['day_of_week'] = features_df['transaction_date'].dt.dayofweek
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        features_df['month'] = features_df['transaction_date'].dt.month
        
        # Enhanced time-based features
        features_df['is_unusual_hour'] = features_df['hour'].isin([2, 3, 4, 5]).astype(int)
        features_df['is_business_hour'] = features_df['hour'].between(9, 17).astype(int)
        features_df['is_late_night'] = features_df['hour'].isin([22, 23, 0, 1]).astype(int)
        
        # Add additional time risk indicators to amplify unusual time patterns
        features_df['time_risk_score'] = (
            features_df['is_unusual_hour'] * 3 +  # Heavy weight for 2-5 AM
            features_df['is_late_night'] * 1.5 +  # Moderate weight for late night
            (1 - features_df['is_business_hour']) * 0.5  # Light weight for non-business hours
        )
        
        # Amount-based features
        features_df['amount_log'] = np.log1p(features_df['amount'])
        features_df['amount_rounded'] = (features_df['amount'] % 1 == 0).astype(int)
        
        # Enhanced amount features
        features_df['amount_zscore'] = (features_df['amount'] - features_df['amount'].mean()) / features_df['amount'].std()
        features_df['is_round_amount'] = features_df['amount'].isin([100, 200, 500, 1000, 1500, 2000, 2500, 5000]).astype(int)
        
        # Card-based features (extract last 4 digits)
        features_df['card_last_4'] = features_df['card_number'].str[-4:].astype(int)
        
        # Enhanced temporal features for frequent_transactions detection
        features_df = self._add_frequency_features(features_df)
        
        # Categorical encoding
        categorical_cols = ['merchant_category', 'city', 'state', 'country']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                features_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(features_df[col])
            else:
                # Handle unseen categories
                mask = features_df[col].isin(self.label_encoders[col].classes_)
                features_df[f'{col}_encoded'] = 0  # Default for unseen categories
                features_df.loc[mask, f'{col}_encoded'] = self.label_encoders[col].transform(features_df.loc[mask, col])
        
        # Select enhanced numerical features for ML
        feature_columns = [
            'amount', 'amount_log', 'amount_rounded', 'amount_zscore', 'is_round_amount',
            'hour', 'day_of_week', 'is_weekend', 'month', 
            'is_unusual_hour', 'is_business_hour', 'is_late_night', 'time_risk_score',
            'card_last_4',
            'transactions_last_hour', 'transactions_last_day', 'avg_time_between_transactions',
            'merchant_category_encoded', 'city_encoded', 'state_encoded', 'country_encoded'
        ]
        
        self.feature_names = feature_columns
        return features_df[feature_columns]
    
    def _add_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add frequency-based features to detect frequent transactions.
        
        Args:
            df: DataFrame with transaction data (sorted by time)
            
        Returns:
            DataFrame with additional frequency features
        """
        # Initialize frequency features
        df['transactions_last_hour'] = 0
        df['transactions_last_day'] = 0
        df['avg_time_between_transactions'] = 24.0  # Default 24 hours
        
        # For large datasets, use a more efficient approach
        if len(df) > 100000:
            # Simplified frequency features for large datasets
            # Group by card and calculate basic frequency metrics
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
                    
                    # Mark transactions that are very close in time (< 1 hour apart)
                    close_transactions = time_diffs_hours < 1
                    df.loc[card_data.index[close_transactions], 'transactions_last_hour'] = 1
                    
                    # Mark transactions that are part of many in a day
                    daily_counts = card_data.groupby(card_data['transaction_date'].dt.date).size()
                    for date, count in daily_counts.items():
                        if count > 5:  # More than 5 transactions in a day
                            date_mask = card_data['transaction_date'].dt.date == date
                            df.loc[card_data.index[date_mask], 'transactions_last_day'] = count
            
            return df
        
        # More detailed calculation for smaller datasets
        # Group by card to analyze per-card patterns
        for card in df['card_number'].unique():
            card_mask = df['card_number'] == card
            card_transactions = df[card_mask].copy().reset_index()
            
            if len(card_transactions) > 1:
                # Calculate time differences
                card_times = pd.to_datetime(card_transactions['transaction_date'])
                time_diffs = card_times.diff().dt.total_seconds().fillna(3600)
                
                # For each transaction, count recent transactions
                for i in range(len(card_transactions)):
                    current_time = card_times.iloc[i]
                    orig_idx = card_transactions.iloc[i]['index']
                    
                    # Count transactions in last hour
                    hour_ago = current_time - pd.Timedelta(hours=1)
                    recent_hour_count = sum(
                        (card_times >= hour_ago) & (card_times < current_time)
                    )
                    df.loc[orig_idx, 'transactions_last_hour'] = recent_hour_count
                    
                    # Count transactions in last day
                    day_ago = current_time - pd.Timedelta(days=1)
                    recent_day_count = sum(
                        (card_times >= day_ago) & (card_times < current_time)
                    )
                    df.loc[orig_idx, 'transactions_last_day'] = recent_day_count
                
                # Average time between transactions
                avg_time_diff = time_diffs.mean()
                orig_indices = card_transactions['index'].values
                df.loc[orig_indices, 'avg_time_between_transactions'] = avg_time_diff / 3600  # Convert to hours
        
        return df
    
    def detect_anomalies_pca(self, features: pd.DataFrame, n_components: int = 2) -> Dict:
        """
        Detect anomalies using PCA reconstruction error.
        
        Args:
            features: Prepared feature matrix
            n_components: Number of PCA components
            
        Returns:
            Dictionary with results
        """
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA
        self.pca = PCA(n_components=n_components)
        features_pca = self.pca.fit_transform(features_scaled)
        
        # Reconstruct original features
        features_reconstructed = self.pca.inverse_transform(features_pca)
        
        # Calculate reconstruction error
        reconstruction_error = np.sum((features_scaled - features_reconstructed) ** 2, axis=1)
        
        # Determine threshold (95th percentile)
        threshold = np.percentile(reconstruction_error, 95)
        
        # Identify anomalies
        anomalies = reconstruction_error > threshold
        
        return {
            'method': 'PCA',
            'anomalies': anomalies,
            'scores': reconstruction_error,
            'threshold': threshold,
            'pca_components': features_pca,
            'explained_variance_ratio': self.pca.explained_variance_ratio_
        }
    
    def detect_anomalies_isolation_forest(self, features: pd.DataFrame) -> Dict:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            features: Prepared feature matrix
            
        Returns:
            Dictionary with results
        """
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply Isolation Forest with optimizations for large datasets
        n_estimators = min(100, max(50, len(features) // 10000))  # Scale estimators with data size
        max_samples = min(1000, len(features))  # Limit sample size for efficiency
        
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=n_estimators,
            max_samples=max_samples,
            n_jobs=-1  # Use all available CPU cores
        )
        
        anomaly_labels = self.isolation_forest.fit_predict(features_scaled)
        anomaly_scores = self.isolation_forest.score_samples(features_scaled)
        
        # Convert labels (-1 for anomaly, 1 for normal) to boolean
        anomalies = anomaly_labels == -1
        
        return {
            'method': 'Isolation Forest',
            'anomalies': anomalies,
            'scores': -anomaly_scores,  # Negative scores for consistency (higher = more anomalous)
            'threshold': -np.percentile(anomaly_scores, 100 - self.contamination * 100)
        }
    
    def detect_anomalies_lof(self, features: pd.DataFrame, n_neighbors: Optional[int] = None) -> Dict:
        """
        Detect anomalies using Local Outlier Factor.
        
        Args:
            features: Prepared feature matrix
            n_neighbors: Number of neighbors to consider
            
        Returns:
            Dictionary with results
        """
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Scale n_neighbors based on dataset size for efficiency
        if n_neighbors is None:
            n_neighbors = min(20, max(5, int(np.sqrt(len(features)))))
        
        # For very large datasets, use a sample for LOF
        if len(features) > 100000:
            print(f"  Large dataset detected. Using sample of 100,000 for LOF...")
            sample_indices = np.random.choice(len(features), 100000, replace=False)
            features_sample = features_scaled[sample_indices]
            
            # Apply LOF on sample
            lof_sample = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=self.contamination,
                n_jobs=-1
            )
            
            sample_labels = lof_sample.fit_predict(features_sample)
            sample_scores = -lof_sample.negative_outlier_factor_
            
            # Predict on full dataset using fitted model
            # Note: LOF doesn't have predict method, so we use a workaround
            # For production, consider using approximate methods
            threshold = np.percentile(sample_scores, 100 - self.contamination * 100)
            
            # For simplicity, mark anomalies based on sample threshold
            # This is a simplified approach for demonstration
            anomaly_labels = np.ones(len(features))
            anomaly_scores = np.zeros(len(features))
            
            # Set sample results
            anomaly_labels[sample_indices] = sample_labels
            anomaly_scores[sample_indices] = sample_scores
            
            # For non-sampled points, use a simple heuristic based on distance to sample
            # This is a simplified approach - in practice, you'd want a more sophisticated method
            
        else:
            # Apply LOF normally for smaller datasets
            self.lof = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=self.contamination,
                n_jobs=-1
            )
            
            anomaly_labels = self.lof.fit_predict(features_scaled)
            anomaly_scores = -self.lof.negative_outlier_factor_
        
        # Convert labels (-1 for anomaly, 1 for normal) to boolean
        anomalies = anomaly_labels == -1
        
        return {
            'method': 'Local Outlier Factor',
            'anomalies': anomalies,
            'scores': anomaly_scores,
            'threshold': np.percentile(anomaly_scores, 100 - self.contamination * 100)
        }
    
    def detect_anomalies_dbscan(self, features: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> Dict:
        """
        Detect anomalies using DBSCAN clustering.
        
        Args:
            features: Prepared feature matrix
            eps: Maximum distance between samples in a neighborhood
            min_samples: Minimum number of samples in a neighborhood
            
        Returns:
            Dictionary with results
        """
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply DBSCAN
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = self.dbscan.fit_predict(features_scaled)
        
        # Points labeled as -1 are considered noise (anomalies)
        anomalies = cluster_labels == -1
        
        # Calculate distance to nearest cluster center as anomaly score
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)  # Remove noise label
        
        if len(unique_labels) > 0:
            # Calculate cluster centers
            cluster_centers = []
            for label in unique_labels:
                cluster_points = features_scaled[cluster_labels == label]
                center = np.mean(cluster_points, axis=0)
                cluster_centers.append(center)
            
            cluster_centers = np.array(cluster_centers)
            
            # Calculate distance to nearest cluster center
            scores = np.zeros(len(features_scaled))
            for i, point in enumerate(features_scaled):
                if len(cluster_centers) > 0:
                    distances = np.linalg.norm(cluster_centers - point, axis=1)
                    scores[i] = np.min(distances)
                else:
                    scores[i] = 1.0  # High score for noise points
        else:
            scores = np.ones(len(features_scaled))  # All points are anomalies
        
        return {
            'method': 'DBSCAN',
            'anomalies': anomalies,
            'scores': scores,
            'threshold': 0,  # No specific threshold for DBSCAN
            'cluster_labels': cluster_labels
        }
    
    def ensemble_detection(self, features: pd.DataFrame) -> Dict:
        """
        Combine multiple anomaly detection methods.
        
        Args:
            features: Prepared feature matrix
            
        Returns:
            Dictionary with ensemble results
        """
        methods = ['pca', 'isolation_forest', 'lof']
        results = {}
        
        # Run individual methods
        print("  Running PCA-based detection...")
        results['pca'] = self.detect_anomalies_pca(features)
        
        print("  Running Isolation Forest...")
        results['isolation_forest'] = self.detect_anomalies_isolation_forest(features)
        
        print("  Running Local Outlier Factor...")
        results['lof'] = self.detect_anomalies_lof(features)
        
        print("  Combining results...")
        # Combine results using voting
        anomaly_votes = np.zeros(len(features))
        score_sum = np.zeros(len(features))
        
        for method_name, result in results.items():
            anomaly_votes += result['anomalies'].astype(int)
            # Normalize scores to 0-1 range
            normalized_scores = (result['scores'] - np.min(result['scores'])) / (np.max(result['scores']) - np.min(result['scores']) + 1e-8)
            score_sum += normalized_scores
        
        # Ensemble predictions (majority vote)
        ensemble_anomalies = anomaly_votes >= 2  # At least 2 methods agree
        ensemble_scores = score_sum / len(methods)
        
        results['ensemble'] = {
            'method': 'Ensemble',
            'anomalies': ensemble_anomalies,
            'scores': ensemble_scores,
            'votes': anomaly_votes,
            'threshold': 2
        }
        
        return results
    
    def analyze_transactions(self, df: pd.DataFrame) -> Dict:
        """
        Complete anomaly analysis pipeline.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Dictionary with comprehensive results
        """
        print(f"Preparing features for {len(df):,} transactions...")
        features = self.prepare_features(df)
        
        print("Running ensemble anomaly detection...")
        print("  This may take several minutes for large datasets...")
        results = self.ensemble_detection(features)
        
        # Add original data for analysis
        print("Calculating feature importance...")
        for method_name, result in results.items():
            result['feature_importance'] = self.calculate_feature_importance(features, result['anomalies'])
        
        # Store results
        self.results = results
        
        # Calculate performance metrics if ground truth is available
        if 'is_anomaly' in df.columns:
            print("Calculating performance metrics...")
            for method_name, result in results.items():
                metrics = self.calculate_metrics(df['is_anomaly'], result['anomalies'])
                result['metrics'] = metrics
        
        return results
    
    def calculate_feature_importance(self, features: pd.DataFrame, anomalies: np.ndarray) -> Dict:
        """
        Calculate feature importance for anomaly detection.
        
        Args:
            features: Feature matrix
            anomalies: Boolean array of anomaly labels
            
        Returns:
            Dictionary with feature importance scores
        """
        importance = {}
        
        for i, feature_name in enumerate(self.feature_names):
            feature_values = features.iloc[:, i]
            
            # Calculate difference in means between normal and anomalous transactions
            normal_mean = feature_values[~anomalies].mean()
            anomaly_mean = feature_values[anomalies].mean()
            
            # Normalize by standard deviation
            std_dev = feature_values.std()
            if std_dev > 0:
                importance[feature_name] = abs(anomaly_mean - normal_mean) / std_dev
            else:
                importance[feature_name] = 0
        
        return importance
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with metrics
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        return {
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred)
        }
    
    def visualize_results(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Create visualizations of anomaly detection results.
        
        Args:
            df: Original DataFrame
            save_path: Path to save plots
        """
        if not self.results:
            print("No results to visualize. Run analyze_transactions first.")
            return
        
        # Prepare features for visualization
        features = self.prepare_features(df)
        features_scaled = self.scaler.transform(features)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Anomaly Detection Results', fontsize=16)
        
        # 1. PCA scatter plot
        if 'pca' in self.results:
            pca_result = self.results['pca']
            pca_components = pca_result['pca_components']
            
            scatter = axes[0, 0].scatter(
                pca_components[:, 0], 
                pca_components[:, 1],
                c=pca_result['anomalies'], 
                cmap='viridis', 
                alpha=0.6
            )
            axes[0, 0].set_title(f'PCA (Explained Variance: {sum(pca_result["explained_variance_ratio"]):.2%})')
            axes[0, 0].set_xlabel('PC1')
            axes[0, 0].set_ylabel('PC2')
            plt.colorbar(scatter, ax=axes[0, 0])
        
        # 2. t-SNE visualization
        if len(features_scaled) <= 1000:  # t-SNE is computationally expensive
            tsne = TSNE(n_components=2, random_state=42)
            tsne_components = tsne.fit_transform(features_scaled)
            
            ensemble_anomalies = self.results['ensemble']['anomalies']
            scatter = axes[0, 1].scatter(
                tsne_components[:, 0], 
                tsne_components[:, 1],
                c=ensemble_anomalies, 
                cmap='viridis', 
                alpha=0.6
            )
            axes[0, 1].set_title('t-SNE Visualization')
            axes[0, 1].set_xlabel('t-SNE 1')
            axes[0, 1].set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=axes[0, 1])
        else:
            axes[0, 1].text(0.5, 0.5, 'Too many points\nfor t-SNE', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('t-SNE (Skipped)')
        
        # 3. Anomaly scores distribution
        ensemble_scores = self.results['ensemble']['scores']
        axes[0, 2].hist(ensemble_scores, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Ensemble Anomaly Scores')
        axes[0, 2].set_xlabel('Anomaly Score')
        axes[0, 2].set_ylabel('Frequency')
        
        # 4. Method comparison
        method_names = ['pca', 'isolation_forest', 'lof', 'ensemble']
        anomaly_counts = [np.sum(self.results[method]['anomalies']) for method in method_names]
        
        axes[1, 0].bar(method_names, anomaly_counts)
        axes[1, 0].set_title('Anomalies Detected by Method')
        axes[1, 0].set_ylabel('Number of Anomalies')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Feature importance
        ensemble_importance = self.results['ensemble']['feature_importance']
        features_sorted = sorted(ensemble_importance.items(), key=lambda x: x[1], reverse=True)
        feature_names, importance_scores = zip(*features_sorted[:10])  # Top 10 features
        
        axes[1, 1].barh(range(len(feature_names)), importance_scores)
        axes[1, 1].set_yticks(range(len(feature_names)))
        axes[1, 1].set_yticklabels(feature_names)
        axes[1, 1].set_title('Top 10 Important Features')
        axes[1, 1].set_xlabel('Importance Score')
        
        # 6. Performance metrics (if available)
        if 'metrics' in self.results['ensemble']:
            metrics = self.results['ensemble']['metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = axes[1, 2].bar(metric_names, metric_values)
            axes[1, 2].set_title('Performance Metrics')
            axes[1, 2].set_ylabel('Score')
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        else:
            axes[1, 2].text(0.5, 0.5, 'No ground truth\navailable', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Performance Metrics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """
        Generate a text report of anomaly detection results.
        
        Args:
            df: Original DataFrame
            
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No analysis results available. Run analyze_transactions first."
        
        report = []
        report.append("=" * 60)
        report.append("ANOMALY DETECTION REPORT")
        report.append("=" * 60)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Transactions: {len(df)}")
        report.append("")
        
        # Method results
        for method_name, result in self.results.items():
            anomaly_count = np.sum(result['anomalies'])
            anomaly_rate = anomaly_count / len(df) * 100
            
            report.append(f"{result['method']} Results:")
            report.append(f"  Anomalies Detected: {anomaly_count} ({anomaly_rate:.2f}%)")
            
            if 'metrics' in result:
                metrics = result['metrics']
                report.append(f"  Precision: {metrics['precision']:.3f}")
                report.append(f"  Recall: {metrics['recall']:.3f}")
                report.append(f"  F1-Score: {metrics['f1_score']:.3f}")
                report.append(f"  Accuracy: {metrics['accuracy']:.3f}")
            
            report.append("")
        
        # Top anomalous transactions
        if 'ensemble' in self.results:
            ensemble_result = self.results['ensemble']
            top_anomaly_indices = np.argsort(ensemble_result['scores'])[-10:][::-1]
            
            report.append("TOP 10 MOST ANOMALOUS TRANSACTIONS:")
            report.append("-" * 40)
            
            for i, idx in enumerate(top_anomaly_indices, 1):
                tx = df.iloc[idx]
                score = ensemble_result['scores'][idx]
                report.append(f"{i:2d}. Score: {score:.3f} | ${tx['amount']:>8.2f} | {tx['merchant_name']} ({tx['merchant_category']})")
                report.append(f"    {tx['city']}, {tx['state']} | {tx['transaction_date']}")
                report.append("")
        
        # Feature importance (if available)
        if 'ensemble' in self.results and 'feature_importance' in self.results['ensemble']:
            importance = self.results['ensemble']['feature_importance']
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            report.append("TOP 5 IMPORTANT FEATURES:")
            report.append("-" * 30)
            for feature, score in top_features:
                report.append(f"  {feature}: {score:.3f}")
            report.append("")
        
        return "\n".join(report)

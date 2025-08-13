"""
Core data models and types for the anomaly detection system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union
from enum import Enum
import pandas as pd
import numpy as np


class AnomalyType(Enum):
    """Enumeration of supported anomaly types."""
    UNUSUAL_AMOUNT = "unusual_amount"
    UNUSUAL_LOCATION = "unusual_location"
    UNUSUAL_TIME = "unusual_time"
    FREQUENT_TRANSACTIONS = "frequent_transactions"
    UNUSUAL_MERCHANT_CATEGORY = "unusual_merchant_category"
    ROUND_AMOUNT = "round_amount"


class DetectionMethod(Enum):
    """Enumeration of detection methods."""
    PCA = "pca"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    DBSCAN = "dbscan"
    ENSEMBLE = "ensemble"


@dataclass
class CreditCard:
    """Credit card data model."""
    number: str
    expiry_month: int
    expiry_year: int
    cvv: str
    card_type: str
    
    def __post_init__(self):
        """Validate credit card data."""
        if len(self.number) not in [15, 16]:  # Support Amex (15) and Visa/MC/Discover (16)
            raise ValueError("Credit card number must be 15 or 16 digits")
        if not (1 <= self.expiry_month <= 12):
            raise ValueError("Expiry month must be between 1 and 12")
        if len(self.cvv) != 3:
            raise ValueError("CVV must be 3 digits")


@dataclass
class Transaction:
    """Transaction data model."""
    transaction_id: str
    card_number: str
    amount: float
    transaction_date: datetime
    merchant_name: str
    merchant_category: str
    city: str
    state: str
    country: str
    is_anomaly: bool = False
    anomaly_type: Optional[AnomalyType] = None
    
    def to_dict(self) -> Dict:
        """Convert transaction to dictionary."""
        return {
            'transaction_id': self.transaction_id,
            'card_number': self.card_number,
            'amount': self.amount,
            'transaction_date': self.transaction_date,
            'merchant_name': self.merchant_name,
            'merchant_category': self.merchant_category,
            'city': self.city,
            'state': self.state,
            'country': self.country,
            'is_anomaly': self.is_anomaly,
            'anomaly_type': self.anomaly_type.value if self.anomaly_type else None
        }


@dataclass
class DetectionResult:
    """Result of anomaly detection."""
    method: DetectionMethod
    anomalies: np.ndarray  # Boolean array
    scores: np.ndarray     # Anomaly scores
    threshold: float
    metadata: Optional[Dict] = None
    
    @property
    def anomaly_count(self) -> int:
        """Number of detected anomalies."""
        return int(np.sum(self.anomalies))
    
    @property
    def anomaly_rate(self) -> float:
        """Rate of detected anomalies."""
        return self.anomaly_count / len(self.anomalies) if len(self.anomalies) > 0 else 0.0


@dataclass
class PerformanceMetrics:
    """Performance metrics for anomaly detection."""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'accuracy': self.accuracy
        }


@dataclass
class FeatureImportance:
    """Feature importance scores."""
    features: Dict[str, float]
    
    def top_features(self, n: int = 10) -> List[tuple]:
        """Get top N most important features."""
        return sorted(self.features.items(), key=lambda x: x[1], reverse=True)[:n]


@dataclass
class AnalysisReport:
    """Comprehensive analysis report."""
    total_transactions: int
    anomaly_count: int
    anomaly_rate: float
    detection_results: Dict[DetectionMethod, DetectionResult]
    performance_metrics: Dict[DetectionMethod, PerformanceMetrics]
    feature_importance: FeatureImportance
    execution_time: float
    timestamp: datetime
    
    def generate_summary(self) -> str:
        """Generate a summary report."""
        lines = [
            "=" * 60,
            "ANOMALY DETECTION ANALYSIS REPORT",
            "=" * 60,
            f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Transactions: {self.total_transactions:,}",
            f"Known Anomalies: {self.anomaly_count:,} ({self.anomaly_rate:.2%})",
            f"Execution Time: {self.execution_time:.1f} seconds",
            "",
            "DETECTION RESULTS:",
            "-" * 30
        ]
        
        for method, result in self.detection_results.items():
            lines.append(f"{method.value}: {result.anomaly_count:,} anomalies ({result.anomaly_rate:.2%})")
            
            if method in self.performance_metrics:
                metrics = self.performance_metrics[method]
                lines.extend([
                    f"  Precision: {metrics.precision:.3f}",
                    f"  Recall: {metrics.recall:.3f}",
                    f"  F1-Score: {metrics.f1_score:.3f}",
                    ""
                ])
        
        lines.extend([
            "TOP 5 IMPORTANT FEATURES:",
            "-" * 25
        ])
        
        for feature, importance in self.feature_importance.top_features(5):
            lines.append(f"  {feature}: {importance:.3f}")
        
        return "\n".join(lines)


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass


class DataValidationError(Exception):
    """Exception raised for data validation errors."""
    pass


class DetectionError(Exception):
    """Exception raised for detection errors."""
    pass

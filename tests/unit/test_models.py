"""
Unit tests for core models and data structures.
"""

import pytest
import numpy as np
from datetime import datetime
from src.core.models import (
    CreditCard, Transaction, AnomalyType, DetectionMethod,
    DetectionResult, PerformanceMetrics, FeatureImportance,
    AnalysisReport, ConfigurationError, DataValidationError
)


class TestCreditCard:
    """Test CreditCard model."""
    
    def test_valid_credit_card(self):
        """Test creating a valid credit card."""
        card = CreditCard(
            number="1234567890123456",
            expiry_month=12,
            expiry_year=2025,
            cvv="123",
            card_type="Visa"
        )
        assert card.number == "1234567890123456"
        assert card.expiry_month == 12
        assert card.expiry_year == 2025
        assert card.cvv == "123"
        assert card.card_type == "Visa"
    
    def test_invalid_card_number_length(self):
        """Test invalid card number length."""
        with pytest.raises(ValueError, match="Credit card number must be 16 digits"):
            CreditCard(
                number="123456789012345",  # 15 digits
                expiry_month=12,
                expiry_year=2025,
                cvv="123",
                card_type="Visa"
            )
    
    def test_invalid_expiry_month(self):
        """Test invalid expiry month."""
        with pytest.raises(ValueError, match="Expiry month must be between 1 and 12"):
            CreditCard(
                number="1234567890123456",
                expiry_month=13,  # Invalid month
                expiry_year=2025,
                cvv="123",
                card_type="Visa"
            )
    
    def test_invalid_cvv_length(self):
        """Test invalid CVV length."""
        with pytest.raises(ValueError, match="CVV must be 3 digits"):
            CreditCard(
                number="1234567890123456",
                expiry_month=12,
                expiry_year=2025,
                cvv="12",  # 2 digits
                card_type="Visa"
            )


class TestTransaction:
    """Test Transaction model."""
    
    def test_valid_transaction(self):
        """Test creating a valid transaction."""
        transaction_date = datetime(2025, 1, 1, 12, 0, 0)
        transaction = Transaction(
            transaction_id="TXN001",
            card_number="1234567890123456",
            amount=100.50,
            transaction_date=transaction_date,
            merchant_name="Test Store",
            merchant_category="retail",
            city="New York",
            state="NY",
            country="USA"
        )
        
        assert transaction.transaction_id == "TXN001"
        assert transaction.amount == 100.50
        assert transaction.is_anomaly is False
        assert transaction.anomaly_type is None
    
    def test_anomalous_transaction(self):
        """Test creating an anomalous transaction."""
        transaction_date = datetime(2025, 1, 1, 3, 0, 0)  # Unusual time
        transaction = Transaction(
            transaction_id="TXN002",
            card_number="1234567890123456",
            amount=5000.00,
            transaction_date=transaction_date,
            merchant_name="Casino",
            merchant_category="gambling",
            city="Las Vegas",
            state="NV",
            country="USA",
            is_anomaly=True,
            anomaly_type=AnomalyType.UNUSUAL_TIME
        )
        
        assert transaction.is_anomaly is True
        assert transaction.anomaly_type == AnomalyType.UNUSUAL_TIME
    
    def test_to_dict(self):
        """Test transaction to dictionary conversion."""
        transaction_date = datetime(2025, 1, 1, 12, 0, 0)
        transaction = Transaction(
            transaction_id="TXN001",
            card_number="1234567890123456",
            amount=100.50,
            transaction_date=transaction_date,
            merchant_name="Test Store",
            merchant_category="retail",
            city="New York",
            state="NY",
            country="USA",
            is_anomaly=True,
            anomaly_type=AnomalyType.UNUSUAL_AMOUNT
        )
        
        result = transaction.to_dict()
        
        assert result['transaction_id'] == "TXN001"
        assert result['amount'] == 100.50
        assert result['is_anomaly'] is True
        assert result['anomaly_type'] == "unusual_amount"


class TestDetectionResult:
    """Test DetectionResult model."""
    
    def test_detection_result_properties(self):
        """Test detection result properties."""
        anomalies = np.array([True, False, True, False, False])
        scores = np.array([0.8, 0.3, 0.9, 0.2, 0.1])
        
        result = DetectionResult(
            method=DetectionMethod.ISOLATION_FOREST,
            anomalies=anomalies,
            scores=scores,
            threshold=0.5
        )
        
        assert result.anomaly_count == 2
        assert result.anomaly_rate == 0.4  # 2/5
        assert result.method == DetectionMethod.ISOLATION_FOREST
        assert result.threshold == 0.5


class TestPerformanceMetrics:
    """Test PerformanceMetrics model."""
    
    def test_performance_metrics(self):
        """Test performance metrics creation and conversion."""
        metrics = PerformanceMetrics(
            precision=0.75,
            recall=0.60,
            f1_score=0.67,
            accuracy=0.85
        )
        
        assert metrics.precision == 0.75
        assert metrics.recall == 0.60
        assert metrics.f1_score == 0.67
        assert metrics.accuracy == 0.85
        
        result_dict = metrics.to_dict()
        expected = {
            'precision': 0.75,
            'recall': 0.60,
            'f1_score': 0.67,
            'accuracy': 0.85
        }
        assert result_dict == expected


class TestFeatureImportance:
    """Test FeatureImportance model."""
    
    def test_feature_importance(self):
        """Test feature importance creation and top features."""
        features = {
            'amount': 0.8,
            'hour': 0.6,
            'merchant_category': 0.4,
            'location': 0.2,
            'card_type': 0.1
        }
        
        importance = FeatureImportance(features=features)
        top_3 = importance.top_features(3)
        
        assert len(top_3) == 3
        assert top_3[0] == ('amount', 0.8)
        assert top_3[1] == ('hour', 0.6)
        assert top_3[2] == ('merchant_category', 0.4)


class TestAnalysisReport:
    """Test AnalysisReport model."""
    
    def test_analysis_report_creation(self):
        """Test analysis report creation."""
        # Create mock detection results
        anomalies = np.array([True, False, True, False])
        scores = np.array([0.8, 0.3, 0.9, 0.2])
        
        detection_results = {
            DetectionMethod.ISOLATION_FOREST: DetectionResult(
                method=DetectionMethod.ISOLATION_FOREST,
                anomalies=anomalies,
                scores=scores,
                threshold=0.5
            )
        }
        
        performance_metrics = {
            DetectionMethod.ISOLATION_FOREST: PerformanceMetrics(
                precision=0.75,
                recall=0.60,
                f1_score=0.67,
                accuracy=0.85
            )
        }
        
        feature_importance = FeatureImportance(features={'amount': 0.8, 'hour': 0.6})
        
        report = AnalysisReport(
            total_transactions=1000,
            anomaly_count=20,
            anomaly_rate=0.02,
            detection_results=detection_results,
            performance_metrics=performance_metrics,
            feature_importance=feature_importance,
            execution_time=45.5,
            timestamp=datetime(2025, 1, 1, 12, 0, 0)
        )
        
        assert report.total_transactions == 1000
        assert report.anomaly_count == 20
        assert report.anomaly_rate == 0.02
        assert report.execution_time == 45.5
        
        # Test summary generation
        summary = report.generate_summary()
        assert "ANOMALY DETECTION ANALYSIS REPORT" in summary
        assert "Total Transactions: 1,000" in summary
        assert "Known Anomalies: 20 (2.00%)" in summary
        assert "Execution Time: 45.5 seconds" in summary


class TestEnumTypes:
    """Test enumeration types."""
    
    def test_anomaly_type_enum(self):
        """Test AnomalyType enum values."""
        assert AnomalyType.UNUSUAL_AMOUNT.value == "unusual_amount"
        assert AnomalyType.UNUSUAL_TIME.value == "unusual_time"
        assert AnomalyType.FREQUENT_TRANSACTIONS.value == "frequent_transactions"
    
    def test_detection_method_enum(self):
        """Test DetectionMethod enum values."""
        assert DetectionMethod.PCA.value == "pca"
        assert DetectionMethod.ISOLATION_FOREST.value == "isolation_forest"
        assert DetectionMethod.ENSEMBLE.value == "ensemble"


if __name__ == "__main__":
    pytest.main([__file__])

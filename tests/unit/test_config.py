"""
Unit tests for configuration management.
"""

import pytest
import tempfile
import json
from pathlib import Path
from src.core.config import (
    GenerationConfig, DetectionConfig, FeatureConfig, SystemConfig,
    DEFAULT_CONFIG
)
from src.core.models import AnomalyType, DetectionMethod


class TestGenerationConfig:
    """Test GenerationConfig class."""
    
    def test_default_config(self):
        """Test default generation configuration."""
        config = GenerationConfig()
        
        assert config.default_transaction_count == 1000
        assert config.default_card_count == 100
        assert config.default_anomaly_rate == 0.01
        assert config.date_range_days == 30
        
        # Test anomaly type distribution sums to 1.0
        total = sum(config.anomaly_type_distribution.values())
        assert abs(total - 1.0) < 0.001
    
    def test_config_validation_success(self):
        """Test successful configuration validation."""
        config = GenerationConfig(
            default_anomaly_rate=0.05,
            anomaly_type_distribution={
                AnomalyType.UNUSUAL_AMOUNT: 0.5,
                AnomalyType.UNUSUAL_TIME: 0.3,
                AnomalyType.FREQUENT_TRANSACTIONS: 0.2
            }
        )
        
        # Should not raise any exception
        config.validate()
    
    def test_config_validation_invalid_anomaly_rate(self):
        """Test validation with invalid anomaly rate."""
        config = GenerationConfig(default_anomaly_rate=1.5)  # > 1.0
        
        with pytest.raises(ValueError, match="Anomaly rate must be between 0 and 1"):
            config.validate()
    
    def test_config_validation_invalid_distribution(self):
        """Test validation with invalid distribution."""
        config = GenerationConfig(
            anomaly_type_distribution={
                AnomalyType.UNUSUAL_AMOUNT: 0.6,
                AnomalyType.UNUSUAL_TIME: 0.6  # Total = 1.2 > 1.0
            }
        )
        
        with pytest.raises(ValueError, match="Anomaly type distribution must sum to 1.0"):
            config.validate()


class TestDetectionConfig:
    """Test DetectionConfig class."""
    
    def test_default_config(self):
        """Test default detection configuration."""
        config = DetectionConfig()
        
        assert config.contamination == 0.1
        assert config.pca_components == 2
        assert config.isolation_forest_estimators == 100
        assert config.ensemble_voting_threshold == 2
        
        # Test default enabled methods
        expected_methods = [
            DetectionMethod.PCA,
            DetectionMethod.ISOLATION_FOREST,
            DetectionMethod.LOCAL_OUTLIER_FACTOR
        ]
        assert config.enabled_methods == expected_methods
    
    def test_config_validation_success(self):
        """Test successful detection configuration validation."""
        config = DetectionConfig(
            contamination=0.05,
            enabled_methods=[DetectionMethod.ISOLATION_FOREST]
        )
        
        # Should not raise any exception
        config.validate()
    
    def test_config_validation_invalid_contamination(self):
        """Test validation with invalid contamination."""
        config = DetectionConfig(contamination=0.6)  # > 0.5
        
        with pytest.raises(ValueError, match="Contamination must be between 0 and 0.5"):
            config.validate()
    
    def test_config_validation_empty_methods(self):
        """Test validation with no enabled methods."""
        config = DetectionConfig(enabled_methods=[])
        
        with pytest.raises(ValueError, match="At least one detection method must be enabled"):
            config.validate()


class TestFeatureConfig:
    """Test FeatureConfig class."""
    
    def test_default_config(self):
        """Test default feature configuration."""
        config = FeatureConfig()
        
        assert config.enable_time_features is True
        assert config.enable_frequency_features is True
        assert config.enable_amount_features is True
        assert config.enable_categorical_features is True
        
        assert config.unusual_hours == [2, 3, 4, 5]
        assert config.business_hours_start == 9
        assert config.business_hours_end == 17
        assert config.late_night_hours == [22, 23, 0, 1]


class TestSystemConfig:
    """Test SystemConfig class."""
    
    def test_default_config(self):
        """Test default system configuration."""
        config = SystemConfig()
        
        assert isinstance(config.generation, GenerationConfig)
        assert isinstance(config.detection, DetectionConfig)
        assert isinstance(config.features, FeatureConfig)
        
        assert config.random_seed == 42
        assert config.parallel_processing is True
        assert config.log_level == "INFO"
        assert config.output_directory == "data"
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = SystemConfig()
        config_dict = config.to_dict()
        
        assert 'generation' in config_dict
        assert 'detection' in config_dict
        assert 'features' in config_dict
        assert 'random_seed' in config_dict
        
        # Test nested structure
        assert 'default_transaction_count' in config_dict['generation']
        assert 'contamination' in config_dict['detection']
        assert 'enable_time_features' in config_dict['features']
    
    def test_config_from_dict(self):
        """Test configuration from dictionary creation."""
        config_dict = {
            'generation': {
                'default_transaction_count': 5000,
                'default_anomaly_rate': 0.02
            },
            'detection': {
                'contamination': 0.05,
                'enabled_methods': ['isolation_forest', 'pca']
            },
            'features': {
                'enable_time_features': False
            },
            'random_seed': 123,
            'log_level': 'DEBUG'
        }
        
        config = SystemConfig.from_dict(config_dict)
        
        assert config.generation.default_transaction_count == 5000
        assert config.generation.default_anomaly_rate == 0.02
        assert config.detection.contamination == 0.05
        assert config.features.enable_time_features is False
        assert config.random_seed == 123
        assert config.log_level == 'DEBUG'
        
        # Test enum conversion
        expected_methods = [DetectionMethod.ISOLATION_FOREST, DetectionMethod.PCA]
        assert config.detection.enabled_methods == expected_methods
    
    def test_config_file_operations_json(self):
        """Test configuration file save/load operations with JSON."""
        config = SystemConfig()
        config.generation.default_transaction_count = 2000
        config.detection.contamination = 0.03
        config.random_seed = 999
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save configuration
            config.save_to_file(temp_path)
            
            # Load configuration
            loaded_config = SystemConfig.from_file(temp_path)
            
            assert loaded_config.generation.default_transaction_count == 2000
            assert loaded_config.detection.contamination == 0.03
            assert loaded_config.random_seed == 999
        
        finally:
            Path(temp_path).unlink()
    
    def test_config_validation(self):
        """Test system configuration validation."""
        config = SystemConfig()
        
        # Should not raise any exception with default config
        config.validate()
        
        # Test with invalid nested config
        config.generation.default_anomaly_rate = 2.0  # Invalid
        
        with pytest.raises(ValueError):
            config.validate()
    
    def test_config_file_not_found(self):
        """Test loading configuration from non-existent file."""
        with pytest.raises(FileNotFoundError):
            SystemConfig.from_file("non_existent_file.json")
    
    def test_config_unsupported_format(self):
        """Test saving/loading with unsupported file format."""
        config = SystemConfig()
        
        with pytest.raises(ValueError, match="Unsupported configuration file format"):
            config.save_to_file("config.txt")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name
            f.write("invalid config")
        
        try:
            with pytest.raises(ValueError, match="Unsupported configuration file format"):
                SystemConfig.from_file(temp_path)
        finally:
            Path(temp_path).unlink()


class TestDefaultConfig:
    """Test default configuration instance."""
    
    def test_default_config_instance(self):
        """Test that DEFAULT_CONFIG is properly initialized."""
        assert isinstance(DEFAULT_CONFIG, SystemConfig)
        assert isinstance(DEFAULT_CONFIG.generation, GenerationConfig)
        assert isinstance(DEFAULT_CONFIG.detection, DetectionConfig)
        assert isinstance(DEFAULT_CONFIG.features, FeatureConfig)
        
        # Should be valid
        DEFAULT_CONFIG.validate()


if __name__ == "__main__":
    pytest.main([__file__])

"""
Core configuration management for the anomaly detection system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import json

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .models import AnomalyType, DetectionMethod


@dataclass
class GenerationConfig:
    """Configuration for transaction generation."""
    default_transaction_count: int = 1000
    default_card_count: int = 100
    default_anomaly_rate: float = 0.01
    date_range_days: int = 30
    
    anomaly_type_distribution: Dict[AnomalyType, float] = field(default_factory=lambda: {
        AnomalyType.UNUSUAL_AMOUNT: 0.20,
        AnomalyType.UNUSUAL_LOCATION: 0.20,
        AnomalyType.UNUSUAL_TIME: 0.15,
        AnomalyType.FREQUENT_TRANSACTIONS: 0.15,
        AnomalyType.UNUSUAL_MERCHANT_CATEGORY: 0.15,
        AnomalyType.ROUND_AMOUNT: 0.15
    })
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.default_anomaly_rate < 0 or self.default_anomaly_rate > 1:
            raise ValueError("Anomaly rate must be between 0 and 1")
        
        total_distribution = sum(self.anomaly_type_distribution.values())
        if abs(total_distribution - 1.0) > 0.001:
            raise ValueError(f"Anomaly type distribution must sum to 1.0, got {total_distribution}")


@dataclass
class DetectionConfig:
    """Configuration for anomaly detection."""
    contamination: float = 0.1
    pca_components: int = 2
    isolation_forest_estimators: int = 100
    lof_neighbors: int = 20
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    ensemble_voting_threshold: int = 2
    
    # Performance optimization settings
    large_dataset_threshold: int = 100000
    lof_sample_size: int = 100000
    max_isolation_forest_samples: int = 1000
    
    enabled_methods: List[DetectionMethod] = field(default_factory=lambda: [
        DetectionMethod.PCA,
        DetectionMethod.ISOLATION_FOREST,
        DetectionMethod.LOCAL_OUTLIER_FACTOR
    ])
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.contamination < 0 or self.contamination > 0.5:
            raise ValueError("Contamination must be between 0 and 0.5")
        
        if not self.enabled_methods:
            raise ValueError("At least one detection method must be enabled")


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    enable_time_features: bool = True
    enable_frequency_features: bool = True
    enable_amount_features: bool = True
    enable_categorical_features: bool = True
    
    # Time feature settings
    unusual_hours: List[int] = field(default_factory=lambda: [2, 3, 4, 5])
    business_hours_start: int = 9
    business_hours_end: int = 17
    late_night_hours: List[int] = field(default_factory=lambda: [22, 23, 0, 1])
    
    # Amount feature settings
    round_amounts: List[float] = field(default_factory=lambda: [100, 200, 500, 1000, 1500, 2000, 2500, 5000])
    
    # Frequency feature settings
    frequency_window_hours: int = 1
    frequency_window_days: int = 1
    max_daily_transactions_threshold: int = 5


@dataclass
class SystemConfig:
    """Main system configuration."""
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    
    # System settings
    random_seed: Optional[int] = 42
    parallel_processing: bool = True
    log_level: str = "INFO"
    output_directory: str = "data"
    
    @classmethod
    def from_file(cls, config_path: str) -> 'SystemConfig':
        """Load configuration from file."""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        elif path.suffix.lower() in ['.yml', '.yaml']:
            if not HAS_YAML:
                raise ValueError("PyYAML not installed. Install with: pip install pyyaml")
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SystemConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        if 'generation' in data:
            config.generation = GenerationConfig(**data['generation'])
        
        if 'detection' in data:
            # Convert string enums back to enum objects
            detection_data = data['detection'].copy()
            if 'enabled_methods' in detection_data:
                detection_data['enabled_methods'] = [
                    DetectionMethod(method) for method in detection_data['enabled_methods']
                ]
            config.detection = DetectionConfig(**detection_data)
        
        if 'features' in data:
            config.features = FeatureConfig(**data['features'])
        
        # System level settings
        for key in ['random_seed', 'parallel_processing', 'log_level', 'output_directory']:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'generation': {
                'default_transaction_count': self.generation.default_transaction_count,
                'default_card_count': self.generation.default_card_count,
                'default_anomaly_rate': self.generation.default_anomaly_rate,
                'date_range_days': self.generation.date_range_days,
                'anomaly_type_distribution': {
                    anomaly_type.value: weight 
                    for anomaly_type, weight in self.generation.anomaly_type_distribution.items()
                }
            },
            'detection': {
                'contamination': self.detection.contamination,
                'pca_components': self.detection.pca_components,
                'isolation_forest_estimators': self.detection.isolation_forest_estimators,
                'lof_neighbors': self.detection.lof_neighbors,
                'dbscan_eps': self.detection.dbscan_eps,
                'dbscan_min_samples': self.detection.dbscan_min_samples,
                'ensemble_voting_threshold': self.detection.ensemble_voting_threshold,
                'large_dataset_threshold': self.detection.large_dataset_threshold,
                'lof_sample_size': self.detection.lof_sample_size,
                'max_isolation_forest_samples': self.detection.max_isolation_forest_samples,
                'enabled_methods': [method.value for method in self.detection.enabled_methods]
            },
            'features': {
                'enable_time_features': self.features.enable_time_features,
                'enable_frequency_features': self.features.enable_frequency_features,
                'enable_amount_features': self.features.enable_amount_features,
                'enable_categorical_features': self.features.enable_categorical_features,
                'unusual_hours': self.features.unusual_hours,
                'business_hours_start': self.features.business_hours_start,
                'business_hours_end': self.features.business_hours_end,
                'late_night_hours': self.features.late_night_hours,
                'round_amounts': self.features.round_amounts,
                'frequency_window_hours': self.features.frequency_window_hours,
                'frequency_window_days': self.features.frequency_window_days,
                'max_daily_transactions_threshold': self.features.max_daily_transactions_threshold
            },
            'random_seed': self.random_seed,
            'parallel_processing': self.parallel_processing,
            'log_level': self.log_level,
            'output_directory': self.output_directory
        }
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to file."""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        if path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        elif path.suffix.lower() in ['.yml', '.yaml']:
            if not HAS_YAML:
                raise ValueError("PyYAML not installed. Install with: pip install pyyaml")
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")
    
    def validate(self) -> None:
        """Validate entire configuration."""
        self.generation.validate()
        self.detection.validate()


# Default configuration instance
DEFAULT_CONFIG = SystemConfig()

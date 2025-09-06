from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import yaml


class ConfigParsingFailed(Exception):
    pass


@dataclass
class DataConfig:
    version: str
    dataset_name: str
    target_column: str
    binary_target_column: str
    numerical_features: List[str]
    categorical_features: List[str]
    imputation_strategy: str


@dataclass
class FeaturesConfig:
    create_engineered_features: bool
    eps_value: float


@dataclass
class ModelEvalConfig:
    split_ratio: float
    evaluation_metric: str
    cv_folds: int
    stratify: bool


@dataclass
class ModelConfig:
    model_type: str
    model_params: Dict[str, Any]
    n_features_select: int
    hyperparameter_tuning: bool


@dataclass
class Config:
    data_prep: DataConfig
    feature_prep: FeaturesConfig
    model_evaluation: ModelEvalConfig
    model: ModelConfig

    @staticmethod
    def from_yaml(config_file: str):
        with open(config_file, 'r', encoding='utf-8') as stream:
            try:
                config_data = yaml.safe_load(stream)
                return Config(
                    data_prep=DataConfig(**config_data['data_prep']),
                    feature_prep=FeaturesConfig(**config_data['feature_prep']),
                    model_evaluation=ModelEvalConfig(**config_data['model_evaluation']),
                    model=ModelConfig(**config_data['model'])
                )
            except (yaml.YAMLError, OSError) as e:
                raise ConfigParsingFailed from e
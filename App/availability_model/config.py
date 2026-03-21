"""
Configuration file for the availability model project.
Centralized settings for data paths, model parameters, and compute resources.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ERCOT data source
ERCOT_HISTORICAL_DATA = r"C:\Users\tkchu\Downloads\stability_model_starter_kit\combined.csv"

# GPU Configuration
@dataclass
class GPUConfig:
    """GPU and compute configuration"""
    use_gpu: bool = False  # Set to False to avoid CuPy/CUDA issues in Colab
    device: str = "cpu"  # "cuda" or "cpu"
    num_workers: int = 4
    batch_size: int = 256
    mixed_precision: bool = False  # Disabled when using CPU

    # Multi-GPU settings (for supercomputer with 2x A100)
    multi_gpu: bool = False
    gpu_ids: list = None

    def __post_init__(self):
        if self.gpu_ids is None:
            self.gpu_ids = [0, 1]  # Default to 2 GPUs


# Model training parameters
@dataclass
class ModelConfig:
    """Machine learning model configuration"""
    # XGBoost parameters (with regularization to prevent overfitting)
    xgb_n_estimators: int = 500  # Reduced from 1000 to prevent overfitting
    xgb_max_depth: int = 5  # Reduced from 8 for shallower trees (less overfitting)
    xgb_learning_rate: float = 0.05  # Slightly higher for faster convergence
    xgb_tree_method: str = "hist"  # XGBoost 3.1+: use 'hist' with 'device' parameter for GPU
    xgb_gpu_id: int = 0  # Deprecated in XGBoost 3.1+ (use device parameter instead)
    xgb_early_stopping_rounds: int = 20  # Reduced from 50 for earlier stopping
    xgb_subsample: float = 0.8  # Subsample 80% of data per tree
    xgb_colsample_bytree: float = 0.8  # Use 80% of features per tree
    xgb_reg_alpha: float = 0.5  # L1 regularization (moderate)
    xgb_reg_lambda: float = 1.0  # L2 regularization (moderate)

    # Time series parameters
    train_test_split: float = 0.8
    validation_split: float = 0.1
    forecast_horizon: int = 24  # hours
    lookback_window: int = 168  # 7 days of hourly data

    # STL decomposition
    stl_seasonal_period: int = 24  # Daily seasonality
    stl_trend_window: int = 169  # Weekly trend window (must be odd and > period)

    # Random seed for reproducibility
    random_seed: int = 42


# Monte Carlo simulation parameters
@dataclass
class SimulationConfig:
    """Monte Carlo simulation configuration for reliability metrics"""
    n_scenarios: int = 10000  # Number of Monte Carlo samples

    # Contingency parameters (for frequency stability simulation)
    contingency_rate: float = 0.004  # Hourly probability of a generator trip
    contingency_size_mw: float = 1350.0  # Size of largest generator (MW)

    # Grid stability parameters
    system_inertia_h: float = 3.5  # System inertia constant (seconds)
    governor_response_mw_per_hz: float = 2000.0  # Governor response (MW/Hz)
    load_damping: float = 0.6  # Load damping factor

    # Frequency thresholds
    ffr_service_limit_hz: float = 59.7  # FFR service activation threshold (Hz)
    nadir_limit_hz: float = 59.3  # Minimum acceptable frequency nadir (Hz)

    # Legacy parameters (kept for backward compatibility)
    n_contingencies: int = 100  # Number of N-1 contingencies to simulate
    min_contingency_size: float = 2000  # MW (3.3% of 60 GW system)
    max_contingency_size: float = 4500  # MW (7.5% - major event like unit trip)
    ercot_inertia_constant: float = 3.5  # seconds (typical for ERCOT)
    frequency_nadir_threshold: float = 59.4  # Hz (ERCOT under-frequency threshold)
    simulation_timestep: float = 0.01  # seconds
    simulation_duration: float = 10.0  # seconds per contingency
    use_parallel: bool = True  # Parallel processing flag
    n_jobs: int = -1  # -1 = use all available cores


# ERCOT reserve product parameters
@dataclass
class ERCOTReserveConfig:
    """ERCOT-specific ancillary service product definitions"""

    # Regulation Up/Down
    reg_up_response_time: float = 5.0  # minutes
    reg_down_response_time: float = 5.0  # minutes

    # Responsive Reserve Service (RRS)
    rrs_response_time: float = 10.0  # minutes
    rrs_deployment_time: float = 600.0  # seconds

    # Fast Frequency Response (FFR)
    ffr_response_time: float = 0.5  # seconds
    ffr_sustain_duration: float = 15.0  # minutes

    # Emergency Condition Reserve Service (ECRS)
    ecrs_response_time: float = 10.0  # minutes
    ecrs_deployment_time: float = 600.0  # seconds

    # Non-Spinning Reserve (Non-Spin)
    nonspin_response_time: float = 30.0  # minutes


# LLM Configuration for news extraction
@dataclass
class LLMConfig:
    """Configuration for LLM-based news extraction pipeline"""

    # API keys (set via environment variables)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # Model selection
    primary_model: str = "claude"  # Options: "claude", "gpt4", "gemini", "llama"
    fallback_models: list = None

    # Extraction parameters
    max_tokens: int = 2000
    temperature: float = 0.0  # Deterministic for structured extraction

    # Hallucination testing
    n_control_articles: int = 10  # Unrelated articles for hallucination detection
    hallucination_threshold: float = 0.1  # Max acceptable false positive rate

    def __post_init__(self):
        if self.fallback_models is None:
            self.fallback_models = ["gpt4", "gemini"]

        # Try to load API keys from environment
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")


# Data preprocessing parameters
@dataclass
class DataConfig:
    """Data preprocessing and feature engineering configuration"""

    # ERCOT regions
    regions: list = None

    # Feature engineering
    add_cyclical_features: bool = True  # sin/cos encoding for hour, day, month
    add_lag_features: bool = True
    lag_hours: list = None
    add_rolling_features: bool = True
    rolling_windows: list = None

    # Weather features (to be integrated)
    include_weather: bool = False
    weather_vars: list = None

    # Missing data handling
    interpolation_method: str = "linear"
    max_missing_pct: float = 0.05  # Max 5% missing data per column

    def __post_init__(self):
        if self.regions is None:
            self.regions = ["COAST", "EAST", "FWEST", "NORTH", "NCENT",
                          "SOUTH", "SCENT", "WEST", "FAR_WEST",
                          "NORTH_C", "SOUTHERN", "SOUTH_C", "TOTAL"]

        if self.lag_hours is None:
            self.lag_hours = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h to 1 week

        if self.rolling_windows is None:
            self.rolling_windows = [3, 6, 12, 24, 168]  # Various window sizes

        if self.weather_vars is None:
            self.weather_vars = ["temperature", "humidity", "wind_speed"]


# Create default configuration instances
gpu_config = GPUConfig()
model_config = ModelConfig()
simulation_config = SimulationConfig()
ercot_config = ERCOTReserveConfig()
llm_config = LLMConfig()
data_config = DataConfig()


def get_device():
    """Get the appropriate device (GPU/CPU) for PyTorch/XGBoost"""
    import torch
    if gpu_config.use_gpu and torch.cuda.is_available():
        return torch.device(gpu_config.device)
    return torch.device("cpu")


def print_config():
    """Print current configuration settings"""
    configs = {
        "GPU Configuration": gpu_config,
        "Model Configuration": model_config,
        "Simulation Configuration": simulation_config,
        "ERCOT Reserve Configuration": ercot_config,
        "LLM Configuration": llm_config,
        "Data Configuration": data_config
    }

    for name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")
        for field, value in config.__dict__.items():
            print(f"{field}: {value}")


if __name__ == "__main__":
    print_config()

    # Check GPU availability
    import torch
    print(f"\n{'='*60}")
    print("GPU Availability Check")
    print(f"{'='*60}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

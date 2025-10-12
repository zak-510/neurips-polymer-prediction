"""Configuration utilities"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager"""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default"""
        return self.config.get(key, default)

    def save(self, output_path: str):
        """Save configuration to file"""
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

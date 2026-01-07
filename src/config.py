import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    def __init__(self, config_file='config.yaml'):
        config_path = Path(config_file)
        if not config_path.exists():
            config_path = Path(__file__).parent.parent / config_file
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key: str, default=None) -> Any:
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
    
    def __getitem__(self, key: str):
        return self.get(key)

config = Config()

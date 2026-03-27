from pathlib import Path
from src.summarizer_model.utils.utils import Load_yaml,create_dir
from src.summarizer_model.config.artifacts import DataIngestionConfig
from src.summarizer_model.constants.constants import CONFIG_YAML
import os 

class ConfigManager:
    def __init__(self):
        self.config = Load_yaml(CONFIG_YAML)
        self.root_dir = create_dir([self.config.main_root_dir])
    
    def get_di(self)->DataIngestionConfig:
        config = self.config.DataIngestion
        create_dir([config.root_dir])
        dataingestconfig = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_url=str(config.source_url),
            test_csv=Path(config.test_csv),
            train_csv=Path(config.train_csv),
            validation_csv=Path(config.validation_csv)
        )


        


from operator import index
from pathlib import Path
from src.summarizer_model.utils.utils import Load_yaml,create_dir
from src.summarizer_model.constants.constants import CONFIG_YAML
from src.summarizer_model.config.config import ConfigManager
from src.summarizer_model.config.artifacts import DataIngestionArtifact
import pandas as pd
import os 
from datasets import load_dataset
from src.summarizer_model.logging.logger import logger
class DataIngestion:
    def __init__(self,config:DataIngestionArtifact):
        self.config = config
    
    def Load_Dataset(self):
        dataset = load_dataset(self.config.source_url)
        for split,data in dataset.items():
            split_path = os.path.join(self.config.root_dir,split)
            create_dir([split_path])
            csv_path = os.path.join(self.config.root_dir, f"{split}.csv")
            data.to_csv(csv_path, index=False)            

        


from src.summarizer_model.logging.logger import logger
from src.summarizer_model.utils.utils import create_dir
from src.summarizer_model.config.config import *
from src.summarizer_model.config.artifacts import ValidationArtifact,DataIngestionArtifact,DataTransArtifact
from transformers import AutoTokenizer
from src.summarizer_model.constants.constants import *
import pandas as pd
import json
import yaml
from datasets import load_from_disk
class DataTransform:
    def __init__(self,config:DataTransformationConfig,dv:ValidationArtifact):
        self.config = config
        self.dv= dv
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        
    def get_exmaples_to_features(self,example_batch):
        logger.info(f'starting the tokenization')
        input_encodes = self.tokenizer(
            [str(d) for d in example_batch['dialogue']],
            max_length = 1024,
            truncation = True,
            padding = True
        )
        target_encodes = self.tokenizer(
           [str(s) for s in example_batch['summary']],
           max_length = 128,
           truncation= True,
           padding = True
    )
        return {
            'input_ids':input_encodes['input_ids'],
            'attention_mask':input_encodes['attention_mask'],
            'labels': target_encodes['input_ids']
        }        
        
    def Transform(self):
        logger.info(f'now we are starting TransformStage')
        valid_train = pd.read_csv(self.dv.valid_train_csv)
        valid_test = pd.read_csv(self.dv.valid_test_csv)
        train_batch = {
            'dialogue': valid_train['dialogue'].astype(str).tolist(),
            'summary': valid_train['summary'].astype(str).tolist()
        }
        test_batch = {
            'dialogue': valid_test['dialogue'].astype(str).tolist(),
            'summary': valid_test['summary'].astype(str).tolist()
        }
        train = self.get_exmaples_to_features(example_batch=train_batch)
        test = self.get_exmaples_to_features(example_batch=test_batch)
        train_csv = pd.DataFrame(train)
        test_csv = pd.DataFrame(test)
        train_csv.to_csv(self.config.transformed_train_csv,index=False)
        test_csv.to_csv(self.config.transformed_test_csv,index=False)
        return test,train
    
        



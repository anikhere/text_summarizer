from pathlib import Path
from typing import Any
import os 
from box.exceptions import BoxValueError
import yaml
from src.summarizer_model.logging.logger import logger
from ensure import ensure_annotations
from box import ConfigBox

@ensure_annotations
def Load_yaml(yaml_file:Path)->ConfigBox:
    try:
        with open(yaml_file,'r') as file:
            content = yaml.safe_load(file)
            logger.info('loaded the yaml file')
            return ConfigBox(content)
    except Exception as e:
        logger.info('failed to load yaml')
        raise(e)

@ensure_annotations
def create_dir(list_of_paths:list,verbose=True):
    for files in list_of_paths:
        os.makedirs(os.path.dirname(files),exist_ok=True)
        logger.info(f'created the path {os.path.dirname(files)}')
        with open(files,'w') as f:
            f.write('created the file ')
            logger.info(f'created file {files}')


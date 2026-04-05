from pathlib import Path
from src.summarizer_model.utils.utils import Load_yaml,create_dir
from src.summarizer_model.config.artifacts import DataIngestionArtifact,ValidationArtifact,DataTransArtifact,TrainerArtifact
from src.summarizer_model.constants.constants import CONFIG_YAML, PARAMS_YAML
from src.summarizer_model.logging.logger import logger
import os 

class ConfigManager:
    def __init__(self):
        self.config = Load_yaml(CONFIG_YAML)
        self.root_dir = os.makedirs(self.config.main_root_dir,exist_ok=True)
    
    def dataIngestion_artifact(self)->DataIngestionArtifact:
        config = self.config.DataIngestion
        create_dir([config.root_dir])
        dataingestconfig = DataIngestionArtifact(
            root_dir=Path(config.root_dir),
            source_url=str(config.source_url),
            test_csv=Path(config.test_csv),
            train_csv=Path(config.train_csv),
            validation_csv=Path(config.validation_csv)
        )
        return dataingestconfig
   
class ValidatorConfig:
    def __init__(self):
        self.config = Load_yaml(CONFIG_YAML)
        logger.info(f'created the {self.config.main_root_dir}')
        self.root_dir = os.makedirs(self.config.main_root_dir,exist_ok=True)
     
    def start_validation(self):
        valid = self.config.DataValidation
        os.makedirs(valid.validation_dir,exist_ok=True)
        logger.info(f'created the {valid.validation_dir}')
        logger.info(f'starting the validation----')
        validation = ValidationArtifact(
            valid_train_csv=Path(valid.valid_train_csv),
            valid_test_csv=Path(valid.valid_test_csv),
            root_dir=Path(valid.validation_dir),
            valid_report=Path(valid.valid_report),
            status = valid.valid_status
        )
        return validation

class DataTransformationConfig:
    def __init__(self):
        self.config = Load_yaml(CONFIG_YAML)
        logger.info(f'created the {self.config.main_root_dir}')
        self.root_dir = os.makedirs(self.config.main_root_dir,exist_ok=True)

    def get_transform(self):
        trans = self.config.DataTransformation
        os.makedirs(trans.root_dir,exist_ok=True)
        trans_artifact = DataTransArtifact(
            root_dir=Path(trans.root_dir),
            transformed_train_path =Path(trans.transformed_train_csv),
            transformed_test_path =Path(trans.transformed_test_csv),
            tokenizer_name=trans.tokenizer_name
        )
        return trans_artifact

class model_trainer:
    def __init__(self):
       config_filepath = CONFIG_YAML
       params_filepath = PARAMS_YAML

       self.config = Load_yaml(CONFIG_YAML)
       self.params = Load_yaml(PARAMS_YAML)
       os.makedirs(self.config.main_root_dir,exist_ok=True)
    
    def get_model_artifact(self)-> TrainerArtifact:
        config = self.config.ModelTrainer
        params = self.params.TrainingArgument
        os.makedirs(config.root_dir,exist_ok=True)
        model_artifacts = TrainerArtifact(
            root_dir=config.root_dir,
            model_path=config.model_path,
            model_cpkt=config.model_ckpt,
        )
        return model_artifacts

class ModelEvaluation:
    def __init__(self):
        pass
        

        


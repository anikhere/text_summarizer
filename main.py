from src.summarizer_model.logging.logger import logger
from src.summarizer_model.config.config import ConfigManager,ValidatorConfig,DataTransformationConfig
from src.summarizer_model.config.artifacts import DataIngestionArtifact,ValidationArtifact,DataTransArtifact,TrainerArtifact
from src.summarizer_model.components.components import DataIngestion
from src.summarizer_model.constants.constants import CONFIG_YAML
from src.summarizer_model.components.valid import ValidationComponents
from src.summarizer_model.components.transformation import DataTransform
from src.summarizer_model.components.trainer import ModelTrainer
from src.summarizer_model.config.config import model_trainer

if __name__ == "__main__":
   logger.info('started data_ingestion')
   config_manager=ConfigManager()
   data_ingestion_artifacts = config_manager.dataIngestion_artifact()
   logger.info(f'started the getting the artifacts')
   data_ingestion = DataIngestion(data_ingestion_artifacts)
   logger.info(f'check your artifacts folder all good')
   data_ingestion.Load_Dataset()
   valid = ValidatorConfig()
   valid_artifacts = valid.start_validation()
   print(f'donnnnneeeeeeeeeee-------------')
   valid_comp = ValidationComponents(config=valid_artifacts,di=data_ingestion_artifacts)
   valid_comp.initiate_val()
   logger.info(f'now we are starting the transformation')
   trans = DataTransformationConfig()
   trans_artifacts = trans.get_transform()
   transform = DataTransform(config=trans_artifacts,dv = valid_artifacts)
   transformed = transform.Transform()
   trainer = model_trainer()
   train_artifacts = trainer.get_model_artifact()
   main_model = ModelTrainer(config=train_artifacts,trans_art=trans_artifacts)
   main_model.train()


   
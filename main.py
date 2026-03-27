from src.summarizer_model.logging.logger import logger
from src.summarizer_model.config.config import ConfigManager
from src.summarizer_model.config.artifacts import DataIngestionArtifact
from src.summarizer_model.components.components import DataIngestion
from src.summarizer_model.constants.constants import CONFIG_YAML

if __name__ == "__main__":
   logger.info('started data_ingestion')
   config_manager=ConfigManager()
   data_ingestion_artifacts = config_manager.dataIngestion_artifact()
   logger.info(f'started the getting the artifacts')
   data_ingestion = DataIngestion(data_ingestion_artifacts)
   logger.info(f'check your artifacts folder all good')
   data_ingestion.Load_Dataset()

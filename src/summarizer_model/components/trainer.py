from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoTokenizer
from datasets import load_from_disk
from src.summarizer_model.logging.logger import logger
from src.summarizer_model.config.artifacts import TrainerArtifact,DataTransArtifact
from src.summarizer_model.utils.utils import Load_yaml
from src.summarizer_model.constants.constants import *
from src.summarizer_model.config.config import DataTransformationConfig,model_trainer
from pathlib import Path
import torch

class ModelTrainer:
    def __init__(self, config:model_trainer, trans_art:DataTransArtifact):
        self.config = config
        self.dt = trans_art
        logger.info(f'starting the training phase now ')
    def train(self):
        train_data = load_from_disk(self.dt.transformed_train_path)
        eval_data = load_from_disk(self.dt.transformed_test_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegassus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq_to_seq = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_pegassus)
        trainer_args = TrainingArguments(
            output_dir = self.params, num_train_epochs=1,warmup_steps=500,
            per_device_train_batch_size = 1,per_device_eval_batch_size = 1,
            weight_decay = 0.01,logging_steps=10,evaluation_strategy = 'steps',
            eval_steps = 500,save_steps=1e6,gradient_accumulation_steps = 16
        )
        trainer = Trainer(model = model_pegassus,args=trainer_args,
                          tokenizer = tokenizer,data_collator = seq_to_seq,
                          train_dataset = train_data,
                          eval_dataset = eval_data
                          )
        logger.info(f'everything set good for the training phase')
        trainer.train()
        model_pegassus.save_pretrained(self.config.model_path)
        tokenizer.save_pretrained(self.config.model_path)
        logger.info(f'Model saved at {self.config.model_path}')
        

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
        self.params = Load_yaml(PARAMS_YAML).TrainingArgument

        logger.info(f'starting the training phase now ')
    def train(self):
        train_data = load_from_disk(self.dt.transformed_train_path)
        eval_data = load_from_disk(self.dt.transformed_test_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_cpkt)
        model_pegassus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_cpkt).to(device)
        seq_to_seq = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_pegassus)
        trainer_args = TrainingArguments(
            output_dir =str(self.config.root_dir)
            ,num_train_epochs=self.params.num_train_epochs,warmup_steps=self.params.warmup_steps,
            per_device_train_batch_size = self.params.per_device_train_batch_size,per_device_eval_batch_size = self.params.per_device_eval_batch_size,
            weight_decay = self.params.weight_decay,logging_steps=self.params.logging_steps,eval_strategy= self.params.eval_strategy
,
            eval_steps = self.params.eval_steps,save_steps=self.params.save_steps,gradient_accumulation_steps = self.params.gradient_accumulation_steps
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
        

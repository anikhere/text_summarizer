from operator import index
from src.summarizer_model.utils.utils import create_dir
from src.summarizer_model.config.config import *
from src.summarizer_model.config.artifacts import ValidationArtifact,DataIngestionArtifact
import pandas as pd
import json
from datasets import load_from_disk
class ValidationComponents:
    def __init__(self,config:ValidationArtifact,di:DataIngestionArtifact ):
        self.config = config
        self.di = di
        self.train_csv = pd.read_csv(di.train_csv)
        self.test_csv = pd.read_csv(di.test_csv)
        
    
    def check_nulls(self):
        logger.info(f'starting to check for nulls====')
        nulls = self.train_csv.isnull().sum()
        nullest = self.test_csv.isnull().sum()
        logger.info(f'the total nulls in train {nulls} and total in test {nullest}')
        logger.info(f'now removing all the nulls')
        no_null_train_csv = self.train_csv.dropna()
        no_null_test_csv = self.test_csv.dropna()
        no_dup_train_csv = no_null_train_csv.drop_duplicates()
        no_dup_test_csv = no_null_test_csv.drop_duplicates()
        return nulls,nullest,no_dup_train_csv,no_dup_test_csv
    
    def generate_report(self,funct):
        logger.info(f'started to generate the report')
        nulls,nullest,no_dup_train_csv,no_dup_test_csv = funct
        report = {
            'nulls_before_in_train':nulls.to_dict(),
            'nulls_before_in_test':nullest.to_dict(),
            'rows_before_IN_train':self.train_csv.shape[0],
            'rows_after_train':no_dup_train_csv.shape[0],
            'rows_before_in_test':self.test_csv.shape[0],
            'rows_after_test':no_dup_test_csv.shape[0],
            'status':self.config.status
        }
        if len(no_dup_train_csv)>0 and len(no_dup_test_csv)>0:
            status = 'True'
            self.train_csv_cleaned = no_dup_train_csv
            self.test_csv_cleaned = no_dup_test_csv
            create_dir([self.config.root_dir])
            logger.info(f'created the valid_dir')
        else:
            status= self.config.status
        report['status'] = status
        with open(self.config.valid_report,'w') as jsonfile:
            json.dump(report,jsonfile,indent=4)
        return status
    def save_valid_data(self):
        logger.info(f'now we are saving the csvs cleaned')
        self.train_csv_cleaned.to_csv(self.config.valid_train_csv,index=False)
        self.test_csv_cleaned.to_csv(self.config.valid_test_csv,index=False)
    def initiate_val(self):
        nulls = self.check_nulls()
        print(nulls)
        status = self.generate_report(nulls)
        print(f'the status is {status}')
        self.save_valid_data()

        

    
    
        



from dataclasses import dataclass
from pathlib import Path
@dataclass
class DataIngestionArtifact:
    root_dir: Path
    source_url: str
    train_csv: str
    test_csv: str 
    validation_csv:str
    
@dataclass
class ValidationArtifact:
    root_dir:Path
    valid_train_csv:Path
    valid_test_csv:Path
    valid_report:Path
    status:bool

@dataclass
class DataTransArtifact:
    root_dir:Path
    transformed_train_csv:Path
    transformed_test_csv:Path
    tokenizer_name:Path
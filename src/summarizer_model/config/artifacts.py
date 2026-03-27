from dataclasses import dataclass
from pathlib import Path
@dataclass
class DataIngestionArtifact:
    root_dir: Path
    source_url: str
    train_csv: str
    test_csv: str 
    validation_csv:str
    
  
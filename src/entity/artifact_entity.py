from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    raw_file_path: str
    trained_file_path:str 
    test_file_path:str
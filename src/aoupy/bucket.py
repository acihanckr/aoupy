import os
import polars as pl
from polars import DataFrame

def copy_from_bucket(file_name: str, bucket_id: str = None):
    """Copies a file from from specified bucket into the enviroment workspace"""
    if bucket_id == None:
        bucket_id = os.getenv('WORKSPACE_BUCKET')

    os.system(f"gsutil cp '{bucket_id}/{file_name}' .")
    print(f'[INFO] {file_name} is successfully downloaded into your working space')
        
def read_from_bucket(file_name: str, bucket_id: str = None, lazy: bool = True):
    """Copies and reads a csv file from bucket"""
    if file_name.split(".")[-1] != "csv":
            raise ValueError("The specified file is not csv format hence cannot be loaded")
    
    if bucket_id == None:
        bucket_id = os.getenv('WORKSPACE_BUCKET')

    os.system(f"gsutil cp '{bucket_id}/{file_name}' 'bucket_io/{file_name}'")
    print(f'[INFO] {file_name} is successfully downloaded into bucket_io folder')
    
    if lazy:
        return pl.scan_csv(f'bucket_io/{file_name}')
    else:
        return pl.read_csv(f'bucket_io/{file_name}')

def copy_to_bucket(file_name: str, target: str, bucket_id: str = None):
    """Copies a file from enviroment workspace to designated bucket space"""
    if bucket_id == None:
       bucket_id = os.getenv('WORKSPACE_BUCKET')
        
    os.system(f"gsutil cp {file_name} {bucket_id}/{target}")

def ls_bucket(target: str = None, bucket_id: str = None):
    """List the files in the given directory in the given bucket"""
    if bucket_id == None:
       bucket_id = os.getenv('WORKSPACE_BUCKET')
    
    if target == None:
        os.system(f"gsutil ls {bucket_id}")
    else:
        os.system(f"gsutil ls {bucket_id}/{target}")

def remove_from_bucket(file_path: str, bucket_id:str = None):
    """Removes the file from the bucket"""
    
    if bucket_id == None:
       bucket_id = os.getenv('WORKSPACE_BUCKET')
    os.system(f"gsutil rm {bucket_id}/{file_path}")

def write_to_bucket(file: DataFrame, target: str, bucket_id: str =  None):
    """Writes the given file to the given bucket location"""
    file.write_csv(f'bucket_io/temp.csv')
    copy_to_bucket(file_name = 'bucket_io/temp.csv', target=target, bucket_id=bucket_id)
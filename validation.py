
import os

class Validation:

    @staticmethod
    def val_iter(iterable, name):
        if not isinstance(iterable, (list, tuple)):
            raise TypeError(f'{name} must be a list or tuple')
        
        if len(iterable) != 2:
            raise ValueError(f'{name} must have exactly 2 elements')
        
    @staticmethod
    def val_path(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found.")
        
    @staticmethod
    def val_filetype(path, filetypes: tuple):
        if not path.endswith(filetypes):
            raise ValueError(f"File {path} must be of type {filetypes}")
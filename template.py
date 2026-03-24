import os 
import pathlib
import logging
project_name = 'summarizer_model'
files = [
    f'src/{project_name}/__init__.py',
    f'src/{project_name}/components/__init__.py',
    f'src/{project_name}/utils/__init__.py',
    f'src/{project_name}/config/__init__.py',
    f'src/{project_name}/logging/__init__.py',
    f'src/{project_name}/exceptions/__init__.py',
    f'src/{project_name}/cloud/__init__.py',
    'github/workflows/main.yaml',

]

for file in files:
    dirname = os.path.dirname(file)
    os.makedirs(dirname,exist_ok=True)
    print(f'created the {dirname}')
    if not os.path.exists(file):
        with open(file,'wb') as f:
            print(f'created the {file} in dir {dirname}')
    else:
        print(f'the {file} already exists {dirname}')

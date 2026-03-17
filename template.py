import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    "setup.py",
    "app.py",
    "research/trials.ipynb"      
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = filepath.parent
    filedir.mkdir(parents=True, exist_ok=True)
    if not filepath.exists():
        filepath.touch()
        logging.info(f"Created file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")
        

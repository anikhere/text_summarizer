import os 
import sys
import logging

logs_dir = 'log_info'
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_file = os.path.join(logs_dir,'continuous_logs.log')
os.makedirs(logs_dir,exist_ok=True)
logging.basicConfig(
    level= logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('summarizerLogger')
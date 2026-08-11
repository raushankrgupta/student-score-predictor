import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d')}"
log_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(log_path,exist_ok=True)

LOG_FILE_NAME = f"{datetime.now().strftime('%Y_%m_%d_%H')}.log"
LOG_FILE_PATH = os.path.join(log_path,LOG_FILE_NAME)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
)



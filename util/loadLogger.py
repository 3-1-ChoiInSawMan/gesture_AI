import logging
import os

log_path = "./logs/app.log"

os.makedirs(os.path.dirname(log_path), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode="a", encoding="utf-8")
    ],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("gesture")
logger.info("로깅끝")
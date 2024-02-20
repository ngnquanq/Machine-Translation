import logging
from logging.handlers import RotatingFileHandler

#Logger object
logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)

#handler
consoleHandler = logging.StreamHandler()
consoleFormatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                     datefmt="%d-%b-%y %H:%M:%S")
consoleHandler.setFormatter(consoleFormatter)

logger.addHandler(consoleHandler)
logger.info("This is an info message")
logger.warning("This is a warning message")

#Rotating file handler
fileHandler = RotatingFileHandler("process.log")
fileHandler.setFormatter(consoleFormatter)

logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)


#Zip file handler




try:
    x = 1/0
except Exception as e:
    logger.error("Exception occurred", exc_info=e)
    
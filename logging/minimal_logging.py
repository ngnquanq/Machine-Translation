#Short logging example

import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s::%(levelname)s::%(message)s",
                    filename='process.log')
logging.info('This will get logged')

logging.warning('This is a warning')
logging.critical('This is a critical log message')



# Output
# WARNING:root:This will get logged
# CRITICAL:root:This will get logged too
# WARNING:root:This


import sys
from loguru import logger


logger.remove(0) #removes the default handler and allows to add custom handler
# logger.add(sys.stderr,level="TRACE") #setting minimum level to trace and sys.stderr means terminal

# logger.add(sys.stderr,format='<green>{time}</green> | {level} | {message}') #using custom style

logger.info("Hello World")
logger.trace("A trace message")
logger.debug('A debug message')
logger.success('A success message')
logger.warning('A warning message')
logger.error("A error message")
logger.critical('A critical message')


# logger.add(sys.stderr,format="{time}|{level}|{message}|{extra}")


# childlogger=logger.bind(seller_id="001",product_id="123")
# childlogger.info("product page opened")

# alternative method using contextualized function without creaing instance
# def log():
#     logger.info("A user requested a service")


# with logger.contextualize(seller_id="001",product_id="123"):
#     log()


# @logger.catch
# def test(x):
#     return 50/x

# test(0)

# to get output in another file rather than terminal and in json format use seralize=True
logger.add("test.log",serialize=True,format="{time}|{level}|{message}|{extra}")

logger.info("Hi there")


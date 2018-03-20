import logger

if __name__ == "__main__":
    logger.init(level='debug', filename="test.log")

    logger.log('INFO', "This is an INFO message")
    logger.log('DEBUG', "This is an DEBUG message")
    logger.log('WARNING', "This is an WARNING message")
    logger.log('ERROR', "This is an ERROR message")
    logger.log('CRITICAL', "This is an CRITICAL message")
    logger.log('SAUSAGE', "This is a non-existent message")

import logging

#These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[0;%dm"
BOLD_SEQ = "\033[1m"

def formatter_message(message, use_color = True):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message

def to_numeric_level(str_level):
        numeric_level = getattr(logging, str_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: {:s}'.format(str_level))
        return numeric_level

class ColoredFormatter(logging.Formatter):
    RED = 1
    GREEN = 2
    YELLOW = 3
    BLUE = 4
    WHITE = 7

    COLORS = {
        'WARNING'  : YELLOW,
        'INFO'     : GREEN,
        'DEBUG'    : BLUE,
        'CRITICAL' : RED,
        'ERROR'    : RED
    }

    def __init__(self, msg, use_color = True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in self.COLORS:
            levelname_color = COLOR_SEQ % (30 + self.COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)

def init(level='WARNING', filename=None, name='main'):
        logger = logging.getLogger(name)
        FORMAT = "[%(name)s][%(levelname)-8s]  %(message)s" #($BOLD%(filename)s$RESET:%(lineno)d)"
        
        # Check whether log should get redirected to a file
        # only use vt100 colors if logging to stdout
        if filename is not None:
            COLOR_FORMAT = formatter_message(FORMAT, False)
            color_formatter = ColoredFormatter(COLOR_FORMAT, False)
            logfile = logging.FileHandler(filename, delay=True)
            logfile.setFormatter(color_formatter)
            logger.addHandler(logfile)
        else:
            COLOR_FORMAT = formatter_message(FORMAT, True)
            color_formatter = ColoredFormatter(COLOR_FORMAT)
            console = logging.StreamHandler()
            console.setFormatter(color_formatter)
            logger.addHandler(console)

        # Check what level of messages we are interested in
        numeric_level = to_numeric_level(level)
        logger.setLevel(numeric_level)

def log(level, message, name='main'):
    logger = logging.getLogger(name)
    try:
        numeric_level = to_numeric_level(level)
        logger.log(numeric_level, message)
    except ValueError:
        logger.warning("Unknown logging level {:s}".format(level))

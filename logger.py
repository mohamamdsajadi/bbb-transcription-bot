import atexit
import json
import logging.config
import logging.handlers
import os
import pathlib
from stream_pipeline.logger import PipelineLogger

def get_logger():
    return logging.getLogger("live_translation")

def setup_logging():
    logger = logging.getLogger("live_translation")
    
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    
    config_file = pathlib.Path("logging_config.json")
    with open(config_file) as f_in:
        logging_config = json.load(f_in)

    logging.config.dictConfig(logging_config)
        
    pipeline_logger = PipelineLogger()
    pipeline_logger.set_debug(True)
    pipeline_logger.set_info(logger.info)
    pipeline_logger.set_warning(logger.warning)
    pipeline_logger.set_error(logger.error)
    pipeline_logger.set_critical(logger.critical)
    pipeline_logger.set_log(logger.log)
    pipeline_logger.set_exception(logger.exception)
    pipeline_logger.set_excepthook(logger.error)
    pipeline_logger.set_threading_excepthook(logger.error)
    
    return logger



LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "extra",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}

def truncate_dict(d, max_length=0):
    if isinstance(d, dict):
        truncated = {}
        for key, value in d.items():
            truncated[key] = truncate_dict(value, max_length)
        return truncated
    elif isinstance(d, (list, tuple, set)):
        return type(d)(truncate_dict(item, max_length) for item in d)
    elif hasattr(d, 'to_dict'):
        return truncate_dict(d.to_dict(), max_length)
    else:
        return truncate_value(d, max_length)

def truncate_value(value, max_length=0):
    if isinstance(value, BaseException):
        return str(value)  # Simplified for exceptions
    elif hasattr(value, 'to_dict'):
        return value.to_dict()
    elif isinstance(value, (dict, list, tuple, set)):
        return truncate_dict(value, max_length)
    else:
        value_str = str(value)
        if max_length > 0 and len(value_str) > max_length:
            return value_str[:max_length] + '...'
        return value_str

class MyJSONFormatter(logging.Formatter):
    def __init__(self, datefmt='%Y-%m-%dT%H:%M:%S%z', max_length=0, fmt_keys=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datefmt = datefmt
        self.max_length = max_length
        self.fmt_keys = fmt_keys or {}

    def format(self, record):
        # Initialize an empty log record
        log_record = {}

        for key, value in self.fmt_keys.items():  # Use .items() to iterate over key-value pairs
            if value not in LOG_RECORD_BUILTIN_ATTRS:
                raise ValueError(f"Invalid value '{value}' in fmt_keys")
            if key in log_record:
                raise ValueError(f"Duplicate key '{key}' in fmt_keys")
            if value == 'asctime':
                log_record[key] = self.formatTime(record, self.datefmt)
                continue
            log_record[key] = getattr(record, value, None)

        # Add extra attributes
        # Add all atrtibutes which are not in the LOG_RECORD_BUILTIN_ATTRS set
        if 'extra' in self.fmt_keys.values():
            for key, value in record.__dict__.items():
                if key not in LOG_RECORD_BUILTIN_ATTRS:
                    extra_set = {}
                    extra_set[key] = value
                    log_record['extra'] = extra_set

        # Truncate and serialize the final log record
        truncated = truncate_dict(log_record, self.max_length)
        return json.dumps(truncated)
        
class SimpleJSONFormatter(logging.Formatter):
    def __init__(self, max_length=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_length = max_length
    
    def format(self, record):
        # Initialize the log record with the message
        log_record = {"message": record.msg}

        # Include extra fields that are not part of the default log attributes
        extra = {key: value for key, value in record.__dict__.items() 
                 if key not in LOG_RECORD_BUILTIN_ATTRS and not key.startswith('_')}
        
        for key, value in extra.items():
            if hasattr(value, 'to_dict'):
                extra[key] = value.to_dict()
            else:
                # try read json to dict
                try:
                    extra[key] = json.loads(value)
                except:
                    pass
            
        
        if extra:
            log_record["extra"] = extra
            
        log_record = truncate_dict(log_record, self.max_length)

        # Convert to JSON
        return json.dumps(log_record, default=str, indent=4)
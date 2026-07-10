import logging
import sys
import structlog
from app.config.settings import get_settings


def setup_logging():
    settings = get_settings()

    # structlog 处理链
    shared_processors = [
        structlog.contextvars.merge_contextvars,  #  自动合并 request_id 等上下文
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.is_prod:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())


def get_logger(name: str = __name__):
    return structlog.get_logger(name)

#prod needs to use JSON because logs are usually collected by machines, not read manually.
# If logs are extremely high volume, I would optimize by reducing noisy logs, using correct
# log levels, sampling repetitive logs, logging asynchronously or through a queue, batching 
# writes, and avoiding expensive fields like huge request bodies or full stack traces unless
# there is an error.
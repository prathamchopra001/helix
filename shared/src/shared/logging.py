import logging
import sys

import structlog


def configure_logging(service_name: str, level: str = "INFO") -> None:
      """Call once at service startup before any logging occurs."""
      structlog.configure(
          processors=[
              structlog.contextvars.merge_contextvars,
              structlog.stdlib.add_log_level,
              structlog.processors.TimeStamper(fmt="iso", utc=True),
              structlog.processors.StackInfoRenderer(),
              structlog.processors.ExceptionRenderer(),
              structlog.processors.JSONRenderer(),
          ],
          wrapper_class=structlog.make_filtering_bound_logger(
              logging.getLevelName(level)
          ),
          context_class=dict,
          logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
          cache_logger_on_first_use=True,
      )
      structlog.contextvars.bind_contextvars(service=service_name)


def get_logger(name: str) -> structlog.BoundLogger:
      """Get a logger bound to a specific module name."""
      return structlog.get_logger(name)


def bind_request_context(correlation_id: str) -> None:
      """Bind correlation ID to all log lines for the current async context."""
      structlog.contextvars.bind_contextvars(correlation_id=correlation_id)


def clear_request_context() -> None:
      structlog.contextvars.clear_contextvars()

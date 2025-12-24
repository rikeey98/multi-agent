"""
Logging configuration for SOC automation system.

로깅 설정 모듈
- 구조화된 로깅
- 파일 및 콘솔 출력
- 로그 레벨 관리
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter.

    컬러 출력을 지원하는 콘솔 포맷터
    """

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logger(
    name: str = "soc_automation",
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logger with file and console handlers.

    파일 및 콘솔 핸들러로 로거를 설정합니다.

    Args:
        name: Logger name
        log_dir: Directory for log files (default: ./logs)
        log_level: Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        console_output: Enable console output
        file_output: Enable file output
        max_bytes: Max bytes per log file
        backup_count: Number of backup files to keep

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    colored_formatter = ColoredFormatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)

    # File handler
    if file_output:
        if log_dir is None:
            log_dir = Path.cwd() / "logs"

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Main log file
        log_file = log_dir / f"{name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

        # Error log file
        error_log_file = log_dir / f"{name}_error.log"
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)

    return logger


def get_agent_logger(agent_name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Get logger for specific agent.

    특정 Agent를 위한 로거를 가져옵니다.

    Args:
        agent_name: Name of the agent
        log_dir: Directory for log files

    Returns:
        logging.Logger: Agent logger
    """
    logger_name = f"soc_automation.agent.{agent_name}"
    return setup_logger(
        name=logger_name,
        log_dir=log_dir,
        log_level="INFO"
    )


def get_workflow_logger(workflow_id: str, log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Get logger for specific workflow.

    특정 워크플로우를 위한 로거를 가져옵니다.

    Args:
        workflow_id: Workflow ID
        log_dir: Directory for log files

    Returns:
        logging.Logger: Workflow logger
    """
    logger_name = f"soc_automation.workflow.{workflow_id}"
    return setup_logger(
        name=logger_name,
        log_dir=log_dir,
        log_level="INFO"
    )


class AgentLogger:
    """
    Context manager for agent logging.

    Agent 로깅을 위한 컨텍스트 매니저
    """

    def __init__(self, agent_name: str, workflow_id: str, log_dir: Optional[Path] = None):
        self.agent_name = agent_name
        self.workflow_id = workflow_id
        self.log_dir = log_dir
        self.logger = None
        self.start_time = None

    def __enter__(self):
        self.logger = get_agent_logger(self.agent_name, self.log_dir)
        self.start_time = datetime.now()
        self.logger.info(f"Starting agent: {self.agent_name} (workflow: {self.workflow_id})")
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        if exc_type is not None:
            self.logger.error(
                f"Agent {self.agent_name} failed with error: {exc_val}",
                exc_info=True
            )
        else:
            self.logger.info(
                f"Agent {self.agent_name} completed successfully in {duration:.2f}s"
            )

        return False  # Don't suppress exceptions


# Global logger instance
_global_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """
    Get global logger instance.

    전역 로거 인스턴스를 가져옵니다.

    Returns:
        logging.Logger: Global logger
    """
    global _global_logger

    if _global_logger is None:
        _global_logger = setup_logger()

    return _global_logger


if __name__ == "__main__":
    """Test logging configuration"""
    print("=== Testing Logger ===\n")

    # Test global logger
    logger = get_logger()
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    print("\n--- Agent Logger Test ---\n")

    # Test agent logger with context manager
    workflow_id = "test-workflow-123"

    with AgentLogger("error_analyzer", workflow_id) as agent_logger:
        agent_logger.info("Analyzing error from log file")
        agent_logger.debug("Pattern matching in progress")
        agent_logger.info("Error classified as TIMEOUT")

    print("\n--- Workflow Logger Test ---\n")

    # Test workflow logger
    workflow_logger = get_workflow_logger(workflow_id)
    workflow_logger.info("Workflow started")
    workflow_logger.info("Delegating to error_analyzer")
    workflow_logger.info("Error analysis complete")
    workflow_logger.info("Workflow completed")

    print("\nLogs written to ./logs directory")
    print("Check soc_automation.log and soc_automation_error.log")

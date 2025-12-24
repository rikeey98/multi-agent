"""
Global configuration settings for SOC automation system.

전역 설정 관리 모듈
- 환경 변수 로드
- 경로 설정
- DB 연결 정보
- 모델 설정
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator

# Load environment variables
load_dotenv()


class OpenAISettings(BaseModel):
    """OpenAI API 설정"""
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="gpt-4o-mini", description="Default model to use")
    temperature: float = Field(default=0.7, description="Model temperature")
    max_tokens: int = Field(default=2000, description="Max tokens for completion")

    @validator("api_key")
    def validate_api_key(cls, v):
        if not v or v == "":
            raise ValueError("OPENAI_API_KEY must be set")
        return v


class MongoDBSettings(BaseModel):
    """MongoDB 연결 설정"""
    uri: str = Field(default="mongodb://localhost:27017", description="MongoDB connection URI")
    database: str = Field(default="soc_db", description="Database name")
    collection_errors: str = Field(default="errors", description="Errors collection")
    collection_logs: str = Field(default="logs", description="Logs collection")
    collection_notifications: str = Field(default="notifications", description="Notifications collection")


class OracleSettings(BaseModel):
    """Oracle DB 연결 설정"""
    user: str = Field(..., description="Oracle username")
    password: str = Field(..., description="Oracle password")
    dsn: str = Field(..., description="Oracle DSN (host:port/service_name)")

    @validator("user", "password", "dsn")
    def validate_not_empty(cls, v):
        if not v:
            raise ValueError("Oracle connection parameters must be set")
        return v


class PathSettings(BaseModel):
    """경로 설정"""
    log_dir: Path = Field(default=Path("/var/log/sim"), description="Simulation log directory")
    sop_dir: Path = Field(default=Path("/opt/sop"), description="SOP documents directory")
    backup_dir: Path = Field(default=Path("/var/backup/soc"), description="Backup directory")
    config_dir: Path = Field(default=Path("/etc/soc"), description="Configuration directory")

    @validator("log_dir", "sop_dir", "backup_dir", "config_dir")
    def ensure_path(cls, v):
        return Path(v)


class NotificationSettings(BaseModel):
    """알림 설정"""
    enabled: bool = Field(default=True, description="Enable notifications")
    critical_threshold: int = Field(default=8, description="Severity threshold for critical alerts (0-10)")
    notification_table: str = Field(default="SOC_NOTIFICATIONS", description="Oracle table for notifications")


class AgentSettings(BaseModel):
    """Agent 설정"""
    max_iterations: int = Field(default=10, description="Max iterations per agent")
    timeout_seconds: int = Field(default=300, description="Agent timeout in seconds")
    enable_parallel: bool = Field(default=True, description="Enable parallel agent execution")


class Settings(BaseModel):
    """전체 시스템 설정"""
    openai: OpenAISettings
    mongodb: MongoDBSettings
    oracle: OracleSettings
    paths: PathSettings
    notification: NotificationSettings
    agent: AgentSettings

    # MCP server configuration path
    mcp_config_path: Path = Field(
        default=Path(__file__).parent / "mcp_config.json",
        description="MCP server configuration file path"
    )

    # Environment
    environment: str = Field(default="development", description="Environment (development/production)")
    debug: bool = Field(default=False, description="Debug mode")

    class Config:
        arbitrary_types_allowed = True


def load_settings() -> Settings:
    """
    Load settings from environment variables.

    환경 변수에서 설정을 로드합니다.

    Returns:
        Settings: Loaded settings object

    Raises:
        ValueError: If required environment variables are missing
    """
    try:
        # OpenAI settings
        openai_settings = OpenAISettings(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
        )

        # MongoDB settings
        mongodb_settings = MongoDBSettings(
            uri=os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
            database=os.getenv("MONGODB_DB", "soc_db"),
            collection_errors=os.getenv("MONGODB_COLLECTION_ERRORS", "errors"),
            collection_logs=os.getenv("MONGODB_COLLECTION_LOGS", "logs"),
            collection_notifications=os.getenv("MONGODB_COLLECTION_NOTIFICATIONS", "notifications")
        )

        # Oracle settings
        oracle_settings = OracleSettings(
            user=os.getenv("ORACLE_USER", "soc_user"),
            password=os.getenv("ORACLE_PASSWORD", ""),
            dsn=os.getenv("ORACLE_DSN", "localhost:1521/ORCL")
        )

        # Path settings
        path_settings = PathSettings(
            log_dir=Path(os.getenv("LOG_DIR", "/var/log/sim")),
            sop_dir=Path(os.getenv("SOP_DIR", "/opt/sop")),
            backup_dir=Path(os.getenv("BACKUP_DIR", "/var/backup/soc")),
            config_dir=Path(os.getenv("CONFIG_DIR", "/etc/soc"))
        )

        # Notification settings
        notification_settings = NotificationSettings(
            enabled=os.getenv("NOTIFICATION_ENABLED", "true").lower() == "true",
            critical_threshold=int(os.getenv("NOTIFICATION_CRITICAL_THRESHOLD", "8")),
            notification_table=os.getenv("NOTIFICATION_TABLE", "SOC_NOTIFICATIONS")
        )

        # Agent settings
        agent_settings = AgentSettings(
            max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "10")),
            timeout_seconds=int(os.getenv("AGENT_TIMEOUT_SECONDS", "300")),
            enable_parallel=os.getenv("AGENT_ENABLE_PARALLEL", "true").lower() == "true"
        )

        # Create settings object
        settings = Settings(
            openai=openai_settings,
            mongodb=mongodb_settings,
            oracle=oracle_settings,
            paths=path_settings,
            notification=notification_settings,
            agent=agent_settings,
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )

        return settings

    except Exception as e:
        raise ValueError(f"Failed to load settings: {str(e)}")


# Global settings instance
settings = load_settings()


if __name__ == "__main__":
    """Test settings loading"""
    import json

    try:
        settings = load_settings()
        print("Settings loaded successfully!")
        print("\n=== OpenAI Settings ===")
        print(f"Model: {settings.openai.model}")
        print(f"Temperature: {settings.openai.temperature}")
        print(f"Max Tokens: {settings.openai.max_tokens}")

        print("\n=== MongoDB Settings ===")
        print(f"URI: {settings.mongodb.uri}")
        print(f"Database: {settings.mongodb.database}")

        print("\n=== Oracle Settings ===")
        print(f"User: {settings.oracle.user}")
        print(f"DSN: {settings.oracle.dsn}")

        print("\n=== Path Settings ===")
        print(f"Log Directory: {settings.paths.log_dir}")
        print(f"SOP Directory: {settings.paths.sop_dir}")

        print("\n=== Notification Settings ===")
        print(f"Enabled: {settings.notification.enabled}")
        print(f"Critical Threshold: {settings.notification.critical_threshold}")

        print("\n=== Agent Settings ===")
        print(f"Max Iterations: {settings.agent.max_iterations}")
        print(f"Timeout: {settings.agent.timeout_seconds}s")
        print(f"Parallel Execution: {settings.agent.enable_parallel}")

        print("\n=== General Settings ===")
        print(f"Environment: {settings.environment}")
        print(f"Debug: {settings.debug}")

    except Exception as e:
        print(f"Error loading settings: {str(e)}")

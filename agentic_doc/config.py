from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    vision_agent_api_key: str = Field()
    max_workers: int = Field(default=10)
    max_retries: int = Field(default=100)
    max_retry_wait_time: int = Field(default=60)
    retry_logging_style: Literal["none", "log_msg", "inline_blobk"] = Field(
        default="log_msg"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


settings = Settings()

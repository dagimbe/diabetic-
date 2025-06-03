# app/config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    mongo_host: str = "localhost"
    mongo_port: str = "27017"
    database_name: str = "diabetes_monitoring"
    
    class Config:
        env_file = ".env"

settings = Settings()
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MYSQL_HOST = 'YOUR HOSTNAME'
    MYSQL_USER = 'YOUR USER'
    MYSQL_PASSWORD = 'YOUR PASSWORD'
    MYSQL_DB = 'DATABASE NAME'
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')

    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

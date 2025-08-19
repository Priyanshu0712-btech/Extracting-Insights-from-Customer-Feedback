import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MYSQL_HOST = 'localhost'
    MYSQL_USER = 'root'
    MYSQL_PASSWORD = '96082683'
    MYSQL_DB = 'UserDetails'
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')

    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

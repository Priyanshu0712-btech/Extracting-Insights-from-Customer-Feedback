import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', 'Your_Port'))
    MYSQL_USER = os.getenv('MYSQL_USER', 'Your_Username')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'Your_Password')
    MYSQL_DB = os.getenv('MYSQL_DB', 'customer_feedback_db')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')

    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

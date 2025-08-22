from flask import Flask
from flask_mysqldb import MySQL
from flask_jwt_extended import JWTManager
import os
from interninfos import config

mysql = MySQL()
jwt = JWTManager()

def create_app():
    app = Flask(__name__)

    # Load configuration
    app.config['MYSQL_HOST'] = config.Config.MYSQL_HOST
    app.config['MYSQL_USER'] = config.Config.MYSQL_USER
    app.config['MYSQL_PASSWORD'] = config.Config.MYSQL_PASSWORD
    app.config['MYSQL_DB'] = config.Config.MYSQL_DB
    app.config['JWT_SECRET_KEY'] = config.Config.JWT_SECRET_KEY

    # JWT Token stored in cookies
    app.config['JWT_TOKEN_LOCATION'] = ['cookies']
    app.config['JWT_COOKIE_SECURE'] = False   # dev = False
    app.config['JWT_COOKIE_CSRF_PROTECT'] = False
    app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret')
    
    # Initialize extensions
    mysql.init_app(app)
    jwt.init_app(app)

    # Import routes
    from .routes import main
    app.register_blueprint(main)

    return app


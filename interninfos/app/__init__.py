from flask import Flask
from flask_mysqldb import MySQL
import os

mysql = MySQL()

def create_app():
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))
    app.secret_key = "your_secret_key"

    # MySQL Configuration
    app.config['MYSQL_HOST'] = 'localhost'
    app.config['MYSQL_USER'] = 'root'
    app.config['MYSQL_PASSWORD'] = 'Vinay@123'   # 👈 your real MySQL password
    app.config['MYSQL_DB'] = 'reviewsense'

    mysql.init_app(app)

    from . import routes
    app.register_blueprint(routes.bp)

    return app

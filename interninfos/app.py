from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from flask_mysqldb import MySQL
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity,
    set_access_cookies, unset_jwt_cookies
)
from werkzeug.security import generate_password_hash, check_password_hash
import config
import MySQLdb.cursors


# Initialize Flask App
app = Flask(__name__)

# Load configuration
app.config['MYSQL_HOST'] = config.Config.MYSQL_HOST
app.config['MYSQL_USER'] = config.Config.MYSQL_USER
app.config['MYSQL_PASSWORD'] = config.Config.MYSQL_PASSWORD
app.config['MYSQL_DB'] = config.Config.MYSQL_DB
app.config['JWT_SECRET_KEY'] = config.Config.JWT_SECRET_KEY

# JWT Token stored in cookies (for browser auth)
app.config['JWT_TOKEN_LOCATION'] = ['cookies']
app.config['JWT_COOKIE_SECURE'] = False   #  Keep False in development, True in production
app.config['JWT_COOKIE_CSRF_PROTECT'] = False  #  Enable in production
app.secret_key = 'super-secret-key'  # For flash messages


# Initialize extensions
mysql = MySQL(app)
jwt = JWTManager(app)


# Routes

@app.route('/')
def home():
    return render_template('login.html')


# Register Route

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        password_hash = generate_password_hash(password)

        cursor = mysql.connection.cursor()
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
            (username, email, password_hash)
        )
        mysql.connection.commit()
        cursor.close()

        flash("Registration successful! Please login.", "success")
        return redirect(url_for('home'))

    return render_template('register.html')


# Login Route
@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()

    if user and check_password_hash(user['password_hash'], password):
        access_token = create_access_token(identity=user['username'])
        response = redirect(url_for('dashboard'))
        set_access_cookies(response, access_token)
        flash("Login successful!", "success")
        return response
    else:
        flash("Invalid email or password", "danger")
        return redirect(url_for('home'))



# Protected Route
@app.route('/dashboard')
@jwt_required()
def dashboard():
    current_user = get_jwt_identity()
    return render_template("dashboard.html", user=current_user)


# Logout Route
@app.route('/logout')
def logout():
    response = redirect(url_for('home'))
    unset_jwt_cookies(response)
    flash("You have been logged out", "info")
    return response



# Run App
if __name__ == '__main__':
    app.run(debug=True)

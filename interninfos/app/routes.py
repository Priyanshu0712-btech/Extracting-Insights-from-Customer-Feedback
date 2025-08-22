from flask import Blueprint, render_template, request, redirect, session, url_for, flash
from flask_jwt_extended import (
    create_access_token, jwt_required, get_jwt_identity,
    set_access_cookies, unset_jwt_cookies
)
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import MySQLdb.cursors

from . import mysql  # initialized in _init_.py

main = Blueprint('main', _name_, url_prefix="/")

# ---------- Helpers ----------
def dict_cursor():
    return mysql.connection.cursor(MySQLdb.cursors.DictCursor)


# ---------- Home / Login page ----------
@main.route("/")
def home():
    return redirect(url_for("main.login"))


# ---------- Login Page (GET) ----------
@main.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")


# ---------- Login Form (POST) ----------
@main.route("/login", methods=["POST"])
def login():
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    cursor = dict_cursor()
    cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
    user = cursor.fetchone()
    cursor.close()

    if user and check_password_hash(user["password_hash"], password):
        access_token = create_access_token(identity=str(user["user_id"]))
        response = redirect(url_for("main.dashboard"))
        set_access_cookies(response, access_token)
        flash("Login successful!", "success")
        return response

    flash("Invalid email or password.", "danger")
    return redirect(url_for("main.login_page"))


# ---------- Register ----------
@main.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("main.register"))

        cursor = dict_cursor()
        # unique email check
        cursor.execute("SELECT user_id FROM users WHERE email=%s", (email,))
        if cursor.fetchone():
            flash("Email already registered. Please login.", "warning")
            cursor.close()
            return redirect(url_for("main.login_page"))

        password_hash = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, created_at) VALUES (%s, %s, %s, %s)",
            (username, email, password_hash, datetime.utcnow())
        )
        mysql.connection.commit()
        cursor.close()
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("main.login_page"))

    return render_template("register.html")


# ---------- Dashboard (Protected) ----------
@main.route("/dashboard")
@jwt_required()
def dashboard():
    user_id = get_jwt_identity()
    cursor = dict_cursor()
    cursor.execute("SELECT user_id, username, email FROM users WHERE user_id=%s", (user_id,))
    user = cursor.fetchone()
    cursor.close()
    return render_template("dashboard.html", user=user)


# ---------- Logout ----------
@main.route("/logout")
def logout():
    response = redirect(url_for("main.login"))
    unset_jwt_cookies(response)
    flash("You have been logged out.", "info")
    return response


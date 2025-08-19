from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from . import mysql

bp = Blueprint("main", __name__)

# ---------------- HOME ----------------
@bp.route("/")
def home():
    return redirect(url_for("main.login"))

# ---------------- REGISTER ----------------
@bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        try:
            cur = mysql.connection.cursor()
            cur.execute(
                "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                (username, email, password),
            )
            mysql.connection.commit()
            cur.close()

            flash("Registration successful! Please login.", "success")
            return redirect(url_for("main.login"))

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")
            return redirect(url_for("main.register"))

    return render_template("register.html")

# ---------------- LOGIN ----------------
@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()

        if user:
            db_password = user[3]   # password column (hashed)
            if check_password_hash(db_password, password):
                session["user_id"] = user[0]
                session["username"] = user[1]
                flash("Login successful!", "success")
                return redirect(url_for("main.dashboard"))
            else:
                flash("Invalid password!", "danger")
        else:
            flash("Email not found!", "danger")

    return render_template("login.html")

# ---------------- DASHBOARD ----------------
@bp.route("/dashboard")
def dashboard():
    if "user_id" in session:
        return f"Welcome {session['username']}! 🎉"
    return redirect(url_for("main.login"))

# ---------------- LOGOUT ----------------
@bp.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("main.login"))

from flask import Blueprint, render_template, request, redirect, session, url_for, flash, current_app
from flask_jwt_extended import (
    create_access_token, jwt_required, get_jwt_identity,
    set_access_cookies, unset_jwt_cookies
)
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import csv
import io
import os
import MySQLdb.cursors

from . import mysql  # initialized in __init__.py

main = Blueprint('main', __name__, url_prefix="/")

# ---------- Helpers ----------
def dict_cursor():
    return mysql.connection.cursor(MySQLdb.cursors.DictCursor)


# ---------- Home / Login page ----------
@main.route("/")
def home():
    return render_template("login.html")

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
            return redirect(url_for("main.home"))

        password_hash = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, created_at) VALUES (%s, %s, %s, %s)",
            (username, email, password_hash, datetime.utcnow())
        )
        mysql.connection.commit()
        cursor.close()
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("main.home"))

    return render_template("register.html")

# ---------- Login ----------
@main.route("/login", methods=["POST"])
def login():
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    cursor = dict_cursor()
    cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
    user = cursor.fetchone()
    cursor.close()

    if user and check_password_hash(user["password_hash"], password):
        # Store user_id as JWT identity
        access_token = create_access_token(identity=str(user["user_id"]))
        response = redirect(url_for("main.dashboard"))
        set_access_cookies(response, access_token)
        flash("Login successful!", "success")
        return response

    flash("Invalid email or password.", "danger")
    return redirect(url_for("main.home"))


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


# ---------- Profile (view + update) ----------
# ---------- PROFILE ----------
@main.route("/profile", methods=["GET", "POST"])
@jwt_required()
def profile():
    user_id = get_jwt_identity()
    cursor = dict_cursor()

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()

        cursor.execute("SELECT user_id FROM users WHERE email=%s AND user_id<>%s", (email, user_id))
        if cursor.fetchone():
            flash("Email already in use by another account.", "warning")
        else:
            cursor.execute(
                "UPDATE users SET username=%s, email=%s WHERE user_id=%s",
                (username, email, user_id)
            )
            mysql.connection.commit()
            flash("Profile updated successfully!", "success")

    # fetch user
    cursor.execute("SELECT user_id, username, email FROM users WHERE user_id=%s", (user_id,))
    user = cursor.fetchone()

    # ✅ fetch reviews WITH review_id
    cursor.execute("""
        SELECT review_id, review_text, uploaded_at
        FROM reviews
        WHERE user_id=%s
        ORDER BY uploaded_at DESC
        LIMIT 50
    """, (user_id,))
    reviews = cursor.fetchall()
    cursor.close()

    return render_template("profile.html", user=user, reviews=reviews)



# ---------- Upload Reviews (raw text) ----------
@main.route("/upload_review", methods=["GET", "POST"])
@jwt_required()
def upload_review():
    user_id = get_jwt_identity()
    if request.method == "POST":
        # Case 1: Raw review text
        raw_review = (request.form.get("raw_review") or "").strip()
        if raw_review:
            cursor = dict_cursor()
            cursor.execute("""
                INSERT INTO reviews (user_id, review_text, product_id, category, uploaded_at, overall_sentiment, overall_sentiment_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (user_id, raw_review, None, None, datetime.utcnow(), None, None))
            mysql.connection.commit()
            cursor.close()
            flash("Review submitted successfully!", "success")
            return redirect(url_for("main.profile"))

        # # Case 2: CSV upload
        # file = request.files.get("file")
        # if file and file.filename.lower().endswith(".csv"):
        #     try:
        #         stream = io.StringIO(file.stream.read().decode("utf-8"))
        #         reader = csv.DictReader(stream)
        #         if "review_text" not in reader.fieldnames:
        #             flash("CSV must contain a 'review_text' column.", "danger")
        #             return redirect(url_for("main.upload_review"))

        #         rows = []
        #         for row in reader:
        #             text = (row.get("review_text") or "").strip()
        #             if text:
        #                 rows.append((user_id, text, None, None, datetime.utcnow(), None, None))

        #         if not rows:
        #             flash("No valid reviews found in CSV.", "warning")
        #             return redirect(url_for("main.upload_review"))

        #         cursor = mysql.connection.cursor()
        #         cursor.executemany("""
        #             INSERT INTO reviews (user_id, review_text, product_id, category, uploaded_at, overall_sentiment, overall_sentiment_score)
        #             VALUES (%s, %s, %s, %s, %s, %s, %s)
        #         """, rows)
        #         mysql.connection.commit()
        #         cursor.close()
        #         flash(f"Uploaded {len(rows)} reviews from CSV.", "success")
        #     except Exception as e:
        #         flash(f"Failed to process CSV: {e}", "danger")
        #     return redirect(url_for("main.profile"))

        # If neither provided
        flash("Please provide raw review text", "warning")    #Please provide raw review text or upload a CSV file.
        return redirect(url_for("main.upload_review"))

    # GET: show recent uploads for this user
    cursor = dict_cursor()
    cursor.execute("""
        SELECT review_text, uploaded_at FROM reviews
        WHERE user_id=%s ORDER BY uploaded_at DESC LIMIT 20
    """, (user_id,))
    reviews = cursor.fetchall()
    cursor.close()
    return render_template("upload_reviews.html", reviews=reviews)


# -------- Delete Review --------
@main.route("/delete_review/<int:review_id>", methods=["POST"])
@jwt_required()
def delete_review(review_id):
    user_id = get_jwt_identity()
    cursor = mysql.connection.cursor()
    cursor.execute("DELETE FROM reviews WHERE review_id=%s AND user_id=%s", (review_id, user_id))
    deleted = cursor.rowcount
    mysql.connection.commit()
    cursor.close()

    if deleted:
        flash("Review deleted successfully!", "success")
    else:
        flash("Could not delete that review.", "warning")
    return redirect(url_for("main.profile"))



# ---------- Logout ----------
@main.route("/logout")
def logout():
    response = redirect(url_for("main.home"))
    unset_jwt_cookies(response)
    flash("You have been logged out.", "info")
    return response

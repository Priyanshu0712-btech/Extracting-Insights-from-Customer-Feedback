from flask import Blueprint, render_template, request, redirect, session, url_for, flash, current_app, send_file, jsonify
from flask_jwt_extended import (
    create_access_token, jwt_required, get_jwt_identity,
    set_access_cookies, unset_jwt_cookies
)
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timezone
import csv
import io
import os
import MySQLdb.cursors

from . import mysql  # initialized in init.py
from . import nlp_utils  # Import NLP utilities

# Import specific functions from nlp_utils to avoid conflicts
from .nlp_utils import map_sentiment, highlight_keywords, preprocess_text

main = Blueprint('main', __name__, url_prefix="/")

# ---------- Helpers ----------
def dict_cursor():
    return mysql.connection.cursor(MySQLdb.cursors.DictCursor)

# ---------- Home page ----------
@main.route("/")
def home():
    return render_template("home.html")

# ---------- User Login ----------
@main.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
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

            # Flash after setting cookie
            session["_flashes"] = []
            flash("Login successful!", "success")
            return response

        session["_flashes"] = []
        flash("Invalid email or password.", "danger")
        return redirect(url_for("main.login"))

    return render_template("login.html")


# ---------- Admin Login ----------
from flask_jwt_extended import create_access_token, set_access_cookies

@main.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        cursor = dict_cursor()
        cursor.execute("SELECT * FROM admins WHERE username=%s", (username,))
        admin = cursor.fetchone()
        cursor.close()

        if admin and check_password_hash(admin["password_hash"], password):
            access_token = create_access_token(identity=admin["username"], additional_claims={"role": "admin"})
            resp = redirect(url_for("main.admin_dashboard"))
            set_access_cookies(resp, access_token)
            flash("Admin login successful!", "success")
            return resp

        flash("Invalid admin credentials.", "danger")
        return redirect(url_for("main.admin_login"))

    return render_template("admin_login.html")



# ---------- Register ----------
@main.route("/register", methods=["GET", "POST"])
def register():
    error_username = None
    error_email = None
    username = ""
    email = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("main.register"))

        cursor = dict_cursor()
        # unique check
        cursor.execute("SELECT user_id FROM users WHERE username=%s", (username,))
        if cursor.fetchone():
            error_username = "Username already exists"
        cursor.execute("SELECT user_id FROM users WHERE email=%s", (email,))
        if cursor.fetchone():
            error_email = "Email already registered"

        if error_username or error_email:
            cursor.close()
            return render_template("register.html",
                                   error_username=error_username,
                                   error_email=error_email,
                                   username=username,
                                   email=email)

        password_hash = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, created_at) VALUES (%s, %s, %s, %s)",
            (username, email, password_hash, datetime.now(timezone.utc))
        )
        mysql.connection.commit()
        cursor.close()
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("main.home"))

    return render_template("register.html")

# Admin Dashboard (User Details + Review Analysis)
from flask_jwt_extended import get_jwt
from flask import jsonify

@main.route("/admin_dashboard")
@jwt_required()
def admin_dashboard():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    cursor = dict_cursor()
    cursor.execute("SELECT user_id, username, email FROM users ORDER BY user_id")
    users = cursor.fetchall()

    for user in users:
        cursor.execute("""
            SELECT review_text, uploaded_at, overall_sentiment
            FROM reviews
            WHERE user_id=%s
            ORDER BY uploaded_at DESC LIMIT 2
        """, (user["user_id"],))
        user_reviews = cursor.fetchall()
        for r in user_reviews:
            r['highlighted_text'] = highlight_keywords(r['review_text'], r['overall_sentiment'])
        user["reviews"] = user_reviews

    cursor.execute("""
        SELECT r.review_id, r.review_text, r.uploaded_at, r.overall_sentiment,
               r.overall_sentiment_score, u.username
        FROM reviews r
        JOIN users u ON r.user_id = u.user_id
        ORDER BY r.uploaded_at DESC LIMIT 100
    """)
    reviews = cursor.fetchall()

    # Add highlighted text
    for r in reviews:
        r['highlighted_text'] = highlight_keywords(r['review_text'], r['overall_sentiment'])

    # Calculate sentiment counts for all reviews
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    for r in reviews:
        sent = r['overall_sentiment'].lower() if r['overall_sentiment'] else 'neutral'
        if sent in sentiment_counts:
            sentiment_counts[sent] += 1

    cursor.close()

    return render_template("admin.html", users=users, reviews=reviews, sentiment_counts=sentiment_counts)

@main.route("/sentiment_trends")
@jwt_required()
def sentiment_trends():
    # Aggregate sentiment counts by date
    cursor = dict_cursor()
    cursor.execute("""
        SELECT DATE(uploaded_at) as date,
               SUM(CASE WHEN overall_sentiment = 'positive' THEN 1 ELSE 0 END) as positive_count,
               SUM(CASE WHEN overall_sentiment = 'negative' THEN 1 ELSE 0 END) as negative_count,
               SUM(CASE WHEN overall_sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_count
        FROM reviews
        GROUP BY DATE(uploaded_at)
        ORDER BY DATE(uploaded_at) ASC
    """)
    rows = cursor.fetchall()
    cursor.close()

    dates = [row['date'].strftime('%Y-%m-%d') for row in rows]
    positive_counts = [row['positive_count'] for row in rows]
    negative_counts = [row['negative_count'] for row in rows]
    neutral_counts = [row['neutral_count'] for row in rows]

    return render_template("sentiment_trends.html",
                           dates=dates,
                           positive_counts=positive_counts,
                           negative_counts=negative_counts,
                           neutral_counts=neutral_counts)




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
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        # Handle profile update
        if username or email:
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

        # Handle password change
        if current_password and new_password and confirm_password:
            # Get current user password hash
            cursor.execute("SELECT password_hash FROM users WHERE user_id=%s", (user_id,))
            user_data = cursor.fetchone()
            if not user_data or not check_password_hash(user_data["password_hash"], current_password):
                flash("Current password is incorrect.", "danger")
            elif new_password != confirm_password:
                flash("New passwords do not match.", "danger")
            elif len(new_password) < 6:
                flash("New password must be at least 6 characters long.", "warning")
            else:
                new_password_hash = generate_password_hash(new_password)
                cursor.execute(
                    "UPDATE users SET password_hash=%s WHERE user_id=%s",
                    (new_password_hash, user_id)
                )
                mysql.connection.commit()
                flash("Password changed successfully!", "success")

    # fetch user
    cursor.execute("SELECT user_id, username, email FROM users WHERE user_id=%s", (user_id,))
    user = cursor.fetchone()

    # fetch reviews WITH review_id
    cursor.execute("""
        SELECT review_id, review_text, uploaded_at, overall_sentiment, overall_sentiment_score
        FROM reviews
        WHERE user_id=%s
        ORDER BY uploaded_at DESC
        LIMIT 50
    """, (user_id,))
    reviews = cursor.fetchall()
    cursor.close()

    # Add highlighted text
    for r in reviews:
        r['highlighted_text'] = highlight_keywords(r['review_text'], r['overall_sentiment'])

    # Calculate sentiment counts for chart
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    for r in reviews:
        sent = r['overall_sentiment'].lower() if r['overall_sentiment'] else 'neutral'
        if sent in sentiment_counts:
            sentiment_counts[sent] += 1

    return render_template("profile.html", user=user, reviews=reviews, sentiment_counts=sentiment_counts)



# ---------- Upload Reviews (raw text) ----------
@main.route("/upload_review", methods=["GET", "POST"])
@jwt_required()
def upload_review():
    user_id = get_jwt_identity()
    if request.method == "POST":
        raw_review = (request.form.get("raw_review") or "").strip()
        file = request.files.get("file")
        rows = []

        # Case 1: raw text
        if raw_review:
            cursor = mysql.connection.cursor()
            # Analyze sentiment and irony
            sentiment_analyzer = nlp_utils.get_sentiment_analyzer()
            irony_analyzer = nlp_utils.get_irony_analyzer()
            # Use enhanced sentiment analysis instead of basic
            sentiment_result = nlp_utils.enhanced_sentiment_analysis(raw_review)
            sentiment_label = sentiment_result['sentiment']
            sent_score = sentiment_result['confidence']
            cursor.execute("""
                INSERT INTO reviews (user_id, review_text, product_id, category, uploaded_at, overall_sentiment, overall_sentiment_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (user_id, raw_review, None, None, datetime.now(timezone.utc), sentiment_label, sent_score))
            mysql.connection.commit()
            cursor.close()
            flash("Review uploaded with sentiment!", "success")
            return redirect(url_for("main.profile"))

        # Case 2: CSV
        elif file and file.filename.lower().endswith(".csv"):
            stream = io.StringIO(file.stream.read().decode("utf-8"))
            reader = csv.DictReader(stream)
            if "review_text" not in reader.fieldnames:
                flash("CSV must contain a 'review_text' column.", "danger")
                return redirect(url_for("main.upload_review"))
            for row in reader:
                text = (row.get("review_text") or "").strip()
                if text:
                    # Analyze sentiment and irony
                    sentiment_analyzer = nlp_utils.get_sentiment_analyzer()
                    irony_analyzer = nlp_utils.get_irony_analyzer()
                    sent_result = sentiment_analyzer(text[:512])[0]
                    sent_label, sent_score = sent_result["label"], float(sent_result["score"])
                    irony_result = irony_analyzer(text[:512])[0]
                    irony_label, irony_score = irony_result["label"], float(irony_result["score"])
                    sentiment_label = map_sentiment(sent_label, irony_label, irony_score)
                    rows.append((user_id, text, None, None, datetime.now(timezone.utc), sentiment_label, sent_score))

            if rows:
                cursor = mysql.connection.cursor()  # Define cursor here before use
                cursor.executemany("""
                    INSERT INTO reviews (user_id, review_text, product_id, category, uploaded_at, overall_sentiment, overall_sentiment_score)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                """, rows)
                mysql.connection.commit()
                cursor.close()
                flash(f"Uploaded {len(rows)} review(s) with sentiment!", "success")
                return redirect(url_for("main.profile"))

        # If neither provided
        flash("Please provide raw review text or upload a CSV.", "warning")
        return redirect(url_for("main.upload_review"))

    # GET: show recent uploads for this user
    cursor = dict_cursor()
    cursor.execute("""
        SELECT review_text, uploaded_at, overall_sentiment, overall_sentiment_score 
        FROM reviews WHERE user_id=%s ORDER BY uploaded_at DESC LIMIT 20
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
    # flash("You have been logged out.", "info")
    return response

# ---------- Detailed Review Analysis ----------
@main.route("/review_analysis/<int:review_id>")
@jwt_required()
def review_analysis(review_id):
    """Return detailed analysis of a specific review."""
    user_id = get_jwt_identity()
    cursor = dict_cursor()

    # Fetch the review and verify ownership
    cursor.execute("""
        SELECT review_id, review_text, overall_sentiment, overall_sentiment_score
        FROM reviews
        WHERE review_id=%s AND user_id=%s
    """, (review_id, user_id))

    review = cursor.fetchone()
    cursor.close()

    if not review:
        return {"error": "Review not found or access denied"}, 404

    # Perform detailed analysis using NLP utilities
    analysis_result = nlp_utils.analyze_review_detailed(
        review['review_text'],
        review['overall_sentiment'],
        review['overall_sentiment_score'] or 0.0
    )

    return {
        'review_id': review_id,
        'original_text': analysis_result['original_text'],
        'highlighted_text': analysis_result['highlighted_text'],
        'aspects': analysis_result['aspects'],
        'aspect_sentiments': analysis_result['aspect_sentiments'],
        'summary': analysis_result['summary']
    }

# ---------- Admin API Routes ----------
@main.route("/admin/api/admin_stats", methods=["GET"])
@jwt_required()
def admin_api_stats():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    cursor = dict_cursor()
    cursor.execute("SELECT COUNT(*) as count FROM reviews")
    reviews_count = cursor.fetchone()['count']
    cursor.execute("SELECT COUNT(*) as count FROM aspect_categories")
    aspects_count = cursor.fetchone()['count']
    cursor.close()

    # Placeholder accuracy
    accuracy = 95

    return jsonify({"reviews": reviews_count, "aspects": aspects_count, "accuracy": accuracy})

# New endpoint for analytics data
@main.route("/admin/api/analytics_data", methods=["GET"])
@jwt_required()
def admin_api_analytics_data():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    cursor = dict_cursor()

    # Get overall sentiment distribution counts
    cursor.execute("""
        SELECT overall_sentiment, COUNT(*) as count
        FROM reviews
        GROUP BY overall_sentiment
    """)
    sentiment_counts_raw = cursor.fetchall()
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for row in sentiment_counts_raw:
        sentiment = row['overall_sentiment'].lower() if row['overall_sentiment'] else 'neutral'
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] = row['count']

    # Get total count of all reviews
    cursor.execute("SELECT COUNT(*) as total FROM reviews")
    total_reviews = cursor.fetchone()['total']

    # Get recent reviews with username, sentiment, confidence (basic data only)
    cursor.execute("""
        SELECT r.review_id, r.review_text, r.overall_sentiment, r.overall_sentiment_score, u.username
        FROM reviews r
        JOIN users u ON r.user_id = u.user_id
        ORDER BY r.uploaded_at DESC
        LIMIT 10
    """)
    reviews = cursor.fetchall()
    # Convert datetime objects to strings for JSON serialization
    for review in reviews:
        if 'uploaded_at' in review and review['uploaded_at']:
            review['uploaded_at'] = review['uploaded_at'].isoformat()

    cursor.close()

    return jsonify({
        "sentiment_counts": sentiment_counts,
        "reviews": reviews,
        "total_reviews": total_reviews
    })

# New endpoint for detailed review analysis (admin)
@main.route("/admin/api/review_analysis/<int:review_id>", methods=["GET"])
@jwt_required()
def admin_api_review_analysis(review_id):
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    cursor = dict_cursor()

    # Fetch the review
    cursor.execute("""
        SELECT r.review_id, r.review_text, r.overall_sentiment, r.overall_sentiment_score, u.username
        FROM reviews r
        JOIN users u ON r.user_id = u.user_id
        WHERE r.review_id=%s
    """, (review_id,))

    review = cursor.fetchone()
    cursor.close()

    if not review:
        return {"error": "Review not found"}, 404

    # Perform detailed analysis using NLP utilities
    from .nlp_utils import analyze_review_detailed
    analysis_result = analyze_review_detailed(
        review['review_text'],
        review['overall_sentiment'],
        review['overall_sentiment_score'] or 0.0
    )

    return jsonify({
        'review_id': review_id,
        'username': review['username'],
        'original_text': analysis_result['original_text'],
        'clean_text': analysis_result['clean_text'],
        'highlighted_text': analysis_result['highlighted_text'],
        'aspects': analysis_result['aspects'],
        'aspect_sentiments': analysis_result['aspect_sentiments'],
        'summary': analysis_result['summary']
    })

@main.route("/admin/api/aspect_categories", methods=["GET"])
@jwt_required()
def admin_api_aspect_categories():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    cursor = dict_cursor()
    cursor.execute("SELECT id, name, description FROM aspect_categories ORDER BY name")
    categories = cursor.fetchall()
    cursor.close()

    return jsonify(categories)

@main.route("/admin/aspect_categories", methods=["POST"])
@jwt_required()
def admin_add_aspect_category():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    data = request.get_json()
    name = data.get('name', '').strip()
    description = data.get('description', '').strip()

    if not name:
        return {"error": "Name is required"}, 400

    cursor = mysql.connection.cursor()
    try:
        cursor.execute("INSERT INTO aspect_categories (name, description) VALUES (%s, %s)", (name, description))
        mysql.connection.commit()
        return {"message": "Aspect category added"}, 201
    except Exception as e:
        mysql.connection.rollback()
        return {"error": str(e)}, 500
    finally:
        cursor.close()

@main.route("/admin/api/sentiment_trends", methods=["GET"])
@jwt_required()
def admin_api_sentiment_trends():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    category = request.args.get('category', 'all')
    time_range = int(request.args.get('time_range', 30))
    sentiment_filter = request.args.get('sentiment', 'all')

    cursor = dict_cursor()

    # Build query
    query = """
        SELECT DATE(uploaded_at) as date,
               SUM(CASE WHEN overall_sentiment = 'positive' THEN 1 ELSE 0 END) as positive_count,
               SUM(CASE WHEN overall_sentiment = 'negative' THEN 1 ELSE 0 END) as negative_count,
               SUM(CASE WHEN overall_sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_count
        FROM reviews
        WHERE uploaded_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
    """
    params = [time_range]

    if category != 'all':
        query += " AND category = %s"
        params.append(category)

    if sentiment_filter != 'all':
        query += " AND overall_sentiment = %s"
        params.append(sentiment_filter)

    query += " GROUP BY DATE(uploaded_at) ORDER BY DATE(uploaded_at) ASC"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    cursor.close()

    dates = [row['date'].strftime('%Y-%m-%d') for row in rows]
    positive = [row['positive_count'] for row in rows]
    negative = [row['negative_count'] for row in rows]
    neutral = [row['neutral_count'] for row in rows]

    return jsonify({"dates": dates, "positive": positive, "negative": negative, "neutral": neutral})

@main.route("/admin/api/aspect_sentiment_distribution", methods=["GET"])
@jwt_required()
def admin_api_aspect_sentiment_distribution():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    category = request.args.get('category', 'all')
    time_range = int(request.args.get('time_range', 30))
    sentiment_filter = request.args.get('sentiment', 'all')

    cursor = dict_cursor()

    # Fetch last 100 reviews for analysis
    query = """
        SELECT review_text, overall_sentiment
        FROM reviews
        WHERE uploaded_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
    """
    params = [time_range]

    if category != 'all':
        query += " AND category = %s"
        params.append(category)

    if sentiment_filter != 'all':
        query += " AND overall_sentiment = %s"
        params.append(sentiment_filter)

    query += " ORDER BY uploaded_at DESC LIMIT 100"

    cursor.execute(query, params)
    reviews = cursor.fetchall()
    cursor.close()

    # Analyze aspects and collect detailed data
    positive_aspects = {}
    negative_aspects = {}

    for review in reviews:
        analysis = nlp_utils.analyze_review_detailed(review['review_text'], review['overall_sentiment'], 0.0)
        for aspect, sent_info in analysis['aspect_sentiments'].items():
            sentiment = sent_info['sentiment']
            confidence = sent_info['confidence']

            if sentiment.lower() == 'positive':
                if aspect not in positive_aspects:
                    positive_aspects[aspect] = {'count': 0, 'total_confidence': 0.0}
                positive_aspects[aspect]['count'] += 1
                positive_aspects[aspect]['total_confidence'] += confidence
            elif sentiment.lower() == 'negative':
                if aspect not in negative_aspects:
                    negative_aspects[aspect] = {'count': 0, 'total_confidence': 0.0}
                negative_aspects[aspect]['count'] += 1
                negative_aspects[aspect]['total_confidence'] += confidence

    # Calculate average confidence and sort
    def process_aspects(aspects_dict):
        processed = []
        for aspect, data in aspects_dict.items():
            avg_confidence = data['total_confidence'] / data['count'] if data['count'] > 0 else 0.0
            processed.append({
                'aspect': aspect,
                'count': data['count'],
                'avg_confidence': round(avg_confidence, 2)
            })
        # Sort by count descending, then by avg_confidence descending
        processed.sort(key=lambda x: (x['count'], x['avg_confidence']), reverse=True)
        return processed[:10]  # Top 10

    top_positive = process_aspects(positive_aspects)
    top_negative = process_aspects(negative_aspects)

    return jsonify({
        'top_positive_aspects': top_positive,
        'top_negative_aspects': top_negative
    })

# --------- Admin API: Get all users ---------
@main.route("/admin/api/users", methods=["GET"])
@jwt_required()
def admin_api_get_users():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    cursor = dict_cursor()
    cursor.execute("SELECT user_id, username, email, created_at FROM users ORDER BY user_id")
    users = cursor.fetchall()
    cursor.close()

    return jsonify({"users": users})

# --------- Admin API: Delete user ---------
@main.route("/admin/api/users/<int:user_id>", methods=["DELETE"])
@jwt_required()
def admin_api_delete_user(user_id):
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    cursor = mysql.connection.cursor()
    cursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
    deleted = cursor.rowcount
    mysql.connection.commit()
    cursor.close()

    if deleted:
        return jsonify({"message": "User deleted successfully"})
    else:
        return jsonify({"error": "User not found"}), 404

@main.route("/admin/change_password", methods=["POST"])
@jwt_required()
def admin_change_password():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    current_password = request.form.get("current_password", "")
    new_password = request.form.get("new_password", "")
    confirm_password = request.form.get("confirm_password", "")

    if not current_password or not new_password or not confirm_password:
        return jsonify({"error": "All password fields are required."}), 400

    username = get_jwt_identity()  # admin username
    cursor = dict_cursor()
    cursor.execute("SELECT password_hash FROM admins WHERE username=%s", (username,))
    admin_data = cursor.fetchone()

    if not admin_data or not check_password_hash(admin_data["password_hash"], current_password):
        cursor.close()
        return jsonify({"error": "Current password is incorrect."}), 400
    elif new_password != confirm_password:
        cursor.close()
        return jsonify({"error": "New passwords do not match."}), 400
    elif len(new_password) < 6:
        cursor.close()
        return jsonify({"error": "New password must be at least 6 characters long."}), 400
    else:
        new_password_hash = generate_password_hash(new_password)
        cursor.execute("UPDATE admins SET password_hash=%s WHERE username=%s", (new_password_hash, username))
        mysql.connection.commit()
        cursor.close()
        return jsonify({"message": "Password changed successfully!"}), 200

@main.route("/admin/export_data", methods=["GET"])
@jwt_required()
def admin_export_data():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    cursor = dict_cursor()
    cursor.execute("""
        SELECT r.review_text, r.uploaded_at, r.overall_sentiment, r.overall_sentiment_score, u.username
        FROM reviews r
        JOIN users u ON r.user_id = u.user_id
        ORDER BY r.uploaded_at DESC
    """)
    reviews = cursor.fetchall()
    cursor.close()

    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.append(['Username', 'Review Text', 'Uploaded At', 'Sentiment', 'Score'])
    for r in reviews:
        ws.append([r['username'], r['review_text'], r['uploaded_at'], r['overall_sentiment'], r['overall_sentiment_score']])
    from io import BytesIO
    bio = BytesIO()
    wb.save(bio)
    bio.seek(0)
    return send_file(bio, as_attachment=True, download_name='reviews.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
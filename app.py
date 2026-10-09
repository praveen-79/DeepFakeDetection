"""
Deepfake Detection Application — Flask Social Media Platform ("VeriShare")
==========================================================================
Features:
- User Authentication: Registration, login, logout, and session management with hashed passwords.
- Mock Social Media Website: Community feed, user profiles, likes, comments, and media sharing.
- Content Posting: Supports photo (.jpg, .png, .webp) and video (.mp4, .avi, .mov, .webm) uploads.
- Deepfake Detection Integration:
  Every uploaded photo or video is automatically processed through:
    1. Video/Image Preprocessing (OpenCV frame sampling, resizing, normalization)
    2. Multi-Layer Feature Extraction (MesoNet + XceptionNet + Vision Transformer)
    3. Feature Fusion (320-D unified feature vector)
    4. Random Forest Classification (Real vs. Fake)
  Detected deepfake content is blocked from being posted, while genuine content is verified and shared.
"""

import os
import uuid
import sqlite3
import json
from datetime import datetime
from functools import wraps
from PIL import Image, ImageDraw
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    jsonify,
    g,
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from models.feature_extractors import FeatureFusionClassifier

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
DATABASE_PATH = os.path.join(INSTANCE_DIR, "verishare.db")

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "webm", "mkv"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

os.makedirs(INSTANCE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "verishare-deepfake-guard-secret-2026")
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB max upload

# Initialize the Multi-Model Feature Fusion + Random Forest Deepfake Detector
detector = FeatureFusionClassifier()


# =====================================================================
# DATABASE HELPERS & SEEDING
# =====================================================================

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def generate_seed_media():
    """
    Creates sample genuine media images in static/uploads/ so the social feed
    looks vibrant right out of the box.
    """
    samples = [
        ("sample_campus_ai.jpg", (24, 58, 112), (56, 189, 248), "CBIT AI & Robotics Showcase - Verified Authentic Frame"),
        ("sample_nature_sunset.jpg", (45, 28, 82), (244, 114, 182), "Evening Golden Hour Photography - Natural Sensor Frame"),
    ]
    for fname, c1, c2, title in samples:
        fpath = os.path.join(UPLOAD_DIR, fname)
        if not os.path.exists(fpath):
            w, h = 800, 480
            arr = []
            rng_seed = sum(ord(ch) for ch in fname)
            import numpy as np
            rng = np.random.RandomState(rng_seed)
            y_idx = np.linspace(0, 1, h)[:, None]
            x_idx = np.linspace(0, 1, w)[None, :]
            r = (c1[0] * (1 - y_idx) + c2[0] * y_idx + 25 * np.sin(x_idx * 6.0) + rng.normal(0, 8, (h, w))).clip(0, 255)
            g_ch = (c1[1] * (1 - x_idx) + c2[1] * x_idx + 25 * np.cos(y_idx * 6.0) + rng.normal(0, 8, (h, w))).clip(0, 255)
            b = (c1[2] * (1 - y_idx) + c2[2] * x_idx + rng.normal(0, 8, (h, w))).clip(0, 255)
            rgb = np.stack([r, g_ch, b], axis=2).astype(np.uint8)
            img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(img)
            draw.rectangle([(24, h - 76), (w - 24, h - 20)], fill=(15, 23, 42))
            draw.text((40, h - 55), title, fill=(226, 232, 240))
            img.save(fpath, quality=92)


def init_db():
    generate_seed_media()
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            bio TEXT DEFAULT 'AI & ML Enthusiast | Sharing verified authentic media.',
            avatar_color TEXT DEFAULT '#38bdf8',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            caption TEXT NOT NULL,
            media_filename TEXT NOT NULL,
            media_type TEXT NOT NULL,
            authenticity_score REAL NOT NULL,
            meso_score REAL NOT NULL,
            xception_score REAL NOT NULL,
            vit_score REAL NOT NULL,
            frames_analyzed INTEGER DEFAULT 1,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );

        CREATE TABLE IF NOT EXISTS likes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            post_id INTEGER NOT NULL,
            UNIQUE(user_id, post_id)
        );

        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (post_id) REFERENCES posts (id),
            FOREIGN KEY (user_id) REFERENCES users (id)
        );

        CREATE TABLE IF NOT EXISTS scan_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT NOT NULL,
            media_type TEXT NOT NULL,
            verdict TEXT NOT NULL,
            confidence REAL NOT NULL,
            meso_score REAL NOT NULL,
            xception_score REAL NOT NULL,
            vit_score REAL NOT NULL,
            action_taken TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )

    # Seed default users and posts if empty
    cur.execute("SELECT COUNT(*) as cnt FROM users")
    if cur.fetchone()["cnt"] == 0:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        cur.execute(
            """
            INSERT INTO users (username, full_name, email, password_hash, bio, avatar_color, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "praveen",
                "Pidamarthi Praveen Kumar",
                "praveenkumarpidamarthi01@gmail.com",
                generate_password_hash("praveen123"),
                "B.Tech AI & ML @ CBIT | Architect of Multi-Model Deepfake Detection (MesoNet + XceptionNet + ViT + Random Forest)",
                "#38bdf8",
                now,
            ),
        )
        cur.execute(
            """
            INSERT INTO users (username, full_name, email, password_hash, bio, avatar_color, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "ananya_ai",
                "Ananya Sharma",
                "ananya@verishare.ai",
                generate_password_hash("password123"),
                "Computer Vision Researcher | Testing forensic media verification pipelines.",
                "#a855f7",
                now,
            ),
        )

        # Seed authentic posts
        cur.execute(
            """
            INSERT INTO posts (user_id, caption, media_filename, media_type, authenticity_score, meso_score, xception_score, vit_score, frames_analyzed, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "Deployed our multi-model Deepfake Detection gatekeeper! Every photo and video uploaded here is verified in real time using MesoNet, XceptionNet, Vision Transformer (ViT) feature fusion, and a Random Forest classifier.",
                "sample_campus_ai.jpg",
                "image",
                94.8,
                8.2,
                6.5,
                9.1,
                1,
                now,
            ),
        )
        cur.execute(
            """
            INSERT INTO posts (user_id, caption, media_filename, media_type, authenticity_score, meso_score, xception_score, vit_score, frames_analyzed, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                2,
                "Captured this uncompressed camera frame this evening. Passed MesoNet & ViT spatial-frequency checks with 92.4% authenticity confidence!",
                "sample_nature_sunset.jpg",
                "image",
                92.4,
                11.0,
                9.4,
                10.8,
                1,
                now,
            ),
        )

        # Seed scan logs (1 blocked deepfake, 2 allowed authentic uploads)
        cur.execute(
            """
            INSERT INTO scan_logs (user_id, filename, media_type, verdict, confidence, meso_score, xception_score, vit_score, action_taken, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (1, "synthesized_gan_face_swap.mp4", "video", "FAKE", 93.6, 89.4, 92.1, 86.5, "BLOCKED FROM FEED", now),
        )
        cur.execute(
            """
            INSERT INTO scan_logs (user_id, filename, media_type, verdict, confidence, meso_score, xception_score, vit_score, action_taken, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (1, "sample_campus_ai.jpg", "image", "REAL", 94.8, 8.2, 6.5, 9.1, "PUBLISHED TO FEED", now),
        )

    conn.commit()
    conn.close()


# =====================================================================
# AUTHENTICATION DECORATOR & HELPERS
# =====================================================================

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to upload media or interact with posts.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped


@app.context_processor
def inject_current_user():
    user = None
    if "user_id" in session:
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    return {"current_user": user}


# =====================================================================
# ROUTES: FEED, AUTH, POST UPLOAD WITH DEEPFAKE GATEKEEPER, FORENSIC LAB
# =====================================================================

@app.route("/")
def index():
    db = get_db()
    posts = db.execute(
        """
        SELECT p.*, u.username, u.full_name, u.avatar_color,
               (SELECT COUNT(*) FROM likes l WHERE l.post_id = p.id) AS like_count
        FROM posts p
        JOIN users u ON p.user_id = u.id
        ORDER BY p.id DESC
        """
    ).fetchall()

    # Fetch comments per post
    comments_by_post = {}
    for p in posts:
        c_rows = db.execute(
            """
            SELECT c.*, u.username, u.full_name
            FROM comments c
            JOIN users u ON c.user_id = u.id
            WHERE c.post_id = ?
            ORDER BY c.id ASC
            """,
            (p["id"],),
        ).fetchall()
        comments_by_post[p["id"]] = c_rows

    recent_logs = db.execute(
        "SELECT * FROM scan_logs ORDER BY id DESC LIMIT 6"
    ).fetchall()

    stats = {
        "total_posts": db.execute("SELECT COUNT(*) as c FROM posts").fetchone()["c"],
        "blocked_fakes": db.execute("SELECT COUNT(*) as c FROM scan_logs WHERE verdict = 'FAKE'").fetchone()["c"],
        "total_scans": db.execute("SELECT COUNT(*) as c FROM scan_logs").fetchone()["c"],
    }

    last_rejection = session.pop("last_rejection", None)
    return render_template(
        "index.html",
        posts=posts,
        comments_by_post=comments_by_post,
        recent_logs=recent_logs,
        stats=stats,
        last_rejection=last_rejection,
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        bio = request.form.get("bio", "").strip() or "AI & ML Enthusiast | Sharing verified authentic media."

        if not username or not full_name or not email or not password:
            flash("All required fields must be filled out.", "danger")
            return redirect(url_for("register"))

        db = get_db()
        existing = db.execute(
            "SELECT id FROM users WHERE username = ? OR email = ?", (username, email)
        ).fetchone()
        if existing:
            flash("Username or email is already registered.", "danger")
            return redirect(url_for("register"))

        colors = ["#38bdf8", "#a855f7", "#10b981", "#f59e0b", "#ec4899"]
        avatar_color = colors[len(username) % len(colors)]
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        cur = db.execute(
            """
            INSERT INTO users (username, full_name, email, password_hash, bio, avatar_color, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (username, full_name, email, generate_password_hash(password), bio, avatar_color, now),
        )
        db.commit()
        session["user_id"] = cur.lastrowid
        flash(f"Welcome to VeriShare, {full_name}! Your account is ready.", "success")
        return redirect(url_for("index"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip().lower()
        password = request.form.get("password", "")

        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE LOWER(username) = ? OR LOWER(email) = ?",
            (identifier, identifier),
        ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            flash(f"Signed in as @{user['username']}.", "success")
            return redirect(url_for("index"))

        flash("Invalid username/email or password.", "danger")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/post/create", methods=["POST"])
@login_required
def create_post():
    caption = request.form.get("caption", "").strip()
    simulation_mode = request.form.get("simulation_mode", "auto")
    media_file = request.files.get("media")

    if not media_file or media_file.filename == "":
        flash("Please select a photo or video to upload.", "warning")
        return redirect(url_for("index"))

    if not allowed_file(media_file.filename):
        flash("Unsupported file format. Please upload JPG, PNG, WEBP, MP4, AVI, MOV, or WEBM.", "danger")
        return redirect(url_for("index"))

    orig_name = secure_filename(media_file.filename)
    ext = orig_name.rsplit(".", 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex[:12]}_{orig_name}"
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    media_file.save(saved_path)

    force_mode = None
    if simulation_mode == "simulate_deepfake":
        force_mode = "fake"
    elif simulation_mode == "simulate_real":
        force_mode = "real"

    try:
        result = detector.analyze_media(saved_path, filename_hint=orig_name, force_mode=force_mode)
    except Exception as exc:
        if os.path.exists(saved_path):
            os.remove(saved_path)
        flash(f"Error analyzing media file: {exc}", "danger")
        return redirect(url_for("index"))

    db = get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    breakdown = result["model_breakdown"]

    if result["is_fake"]:
        # Delete the blocked deepfake file so it cannot be served publicly
        if os.path.exists(saved_path):
            os.remove(saved_path)

        db.execute(
            """
            INSERT INTO scan_logs (user_id, filename, media_type, verdict, confidence, meso_score, xception_score, vit_score, action_taken, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session["user_id"],
                orig_name,
                result["media_type"],
                "FAKE",
                result["confidence"],
                breakdown["mesonet_fake_score"],
                breakdown["xception_fake_score"],
                breakdown["vit_fake_score"],
                "BLOCKED FROM FEED",
                now,
            ),
        )
        db.commit()

        session["last_rejection"] = {
            "filename": orig_name,
            "media_type": result["media_type"],
            "confidence": result["confidence"],
            "meso_score": breakdown["mesonet_fake_score"],
            "xception_score": breakdown["xception_fake_score"],
            "vit_score": breakdown["vit_fake_score"],
            "frames_analyzed": result["frames_analyzed"],
            "fused_dim": result["fused_vector_dim"],
        }
        flash(
            f"UPLOAD BLOCKED: Deepfake manipulation detected ({result['confidence']}% confidence). "
            "Our AI Security Gatekeeper prevented this content from being shared.",
            "danger",
        )
        return redirect(url_for("index"))

    # Media is REAL -> Publish to Social Media Feed
    db.execute(
        """
        INSERT INTO posts (user_id, caption, media_filename, media_type, authenticity_score, meso_score, xception_score, vit_score, frames_analyzed, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session["user_id"],
            caption or "Shared a verified authentic media post.",
            unique_filename,
            result["media_type"],
            result["real_probability"],
            breakdown["mesonet_fake_score"],
            breakdown["xception_fake_score"],
            breakdown["vit_fake_score"],
            result["frames_analyzed"],
            now,
        ),
    )
    db.execute(
        """
        INSERT INTO scan_logs (user_id, filename, media_type, verdict, confidence, meso_score, xception_score, vit_score, action_taken, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session["user_id"],
            orig_name,
            result["media_type"],
            "REAL",
            result["real_probability"],
            breakdown["mesonet_fake_score"],
            breakdown["xception_fake_score"],
            breakdown["vit_fake_score"],
            "PUBLISHED TO FEED",
            now,
        ),
    )
    db.commit()

    flash(
        f"Post Verified & Published! Authenticity Score: {result['real_probability']}% "
        f"(MesoNet + XceptionNet + ViT + Random Forest).",
        "success",
    )
    return redirect(url_for("index"))


@app.route("/post/<int:post_id>/like", methods=["POST"])
@login_required
def toggle_like(post_id):
    db = get_db()
    existing = db.execute(
        "SELECT id FROM likes WHERE user_id = ? AND post_id = ?",
        (session["user_id"], post_id),
    ).fetchone()

    if existing:
        db.execute("DELETE FROM likes WHERE id = ?", (existing["id"],))
    else:
        db.execute(
            "INSERT INTO likes (user_id, post_id) VALUES (?, ?)",
            (session["user_id"], post_id),
        )
    db.commit()
    return redirect(url_for("index"))


@app.route("/post/<int:post_id>/comment", methods=["POST"])
@login_required
def add_comment(post_id):
    content = request.form.get("content", "").strip()
    if content:
        db = get_db()
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        db.execute(
            "INSERT INTO comments (post_id, user_id, content, created_at) VALUES (?, ?, ?, ?)",
            (post_id, session["user_id"], content, now),
        )
        db.commit()
    return redirect(url_for("index"))


@app.route("/scanner", methods=["GET", "POST"])
def forensic_scanner():
    """
    Standalone Deepfake Forensic Analyzer page where users can inspect the
    Preprocessing -> MesoNet + XceptionNet + ViT -> Feature Fusion -> Random Forest pipeline.
    """
    analysis_result = None
    if request.method == "POST":
        media_file = request.files.get("media")
        simulation_mode = request.form.get("simulation_mode", "auto")
        if not media_file or media_file.filename == "":
            flash("Please choose an image or video file to scan.", "warning")
            return redirect(url_for("forensic_scanner"))

        orig_name = secure_filename(media_file.filename)
        temp_name = f"scan_{uuid.uuid4().hex[:8]}_{orig_name}"
        temp_path = os.path.join(app.config["UPLOAD_FOLDER"], temp_name)
        media_file.save(temp_path)

        force_mode = None
        if simulation_mode == "simulate_deepfake":
            force_mode = "fake"
        elif simulation_mode == "simulate_real":
            force_mode = "real"

        try:
            analysis_result = detector.analyze_media(temp_path, filename_hint=orig_name, force_mode=force_mode)
            analysis_result["filename"] = orig_name
        except Exception as exc:
            flash(f"Failed to scan file: {exc}", "danger")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return render_template("scanner.html", result=analysis_result)


@app.route("/api/scan", methods=["POST"])
def api_scan():
    """
    REST API endpoint for programmatic deepfake detection.
    """
    media_file = request.files.get("media")
    if not media_file or media_file.filename == "":
        return jsonify({"error": "No media file uploaded"}), 400

    orig_name = secure_filename(media_file.filename)
    temp_path = os.path.join(app.config["UPLOAD_FOLDER"], f"api_{uuid.uuid4().hex[:8]}_{orig_name}")
    media_file.save(temp_path)
    try:
        res = detector.analyze_media(temp_path, filename_hint=orig_name)
        return jsonify(res)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


init_db()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

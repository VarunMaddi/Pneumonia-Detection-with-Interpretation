import os
import io
import uuid
import logging
import base64
import numpy as np
from PIL import Image

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
    session,
    jsonify,
)

from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from dotenv import load_dotenv
import requests

# ------------------------------------------------------------------
# ENV & APP SETUP
# ------------------------------------------------------------------
load_dotenv(override=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "replace-this-with-a-secret")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.logger.setLevel(logging.INFO)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dcm"}

# ------------------------------------------------------------------
# OPTIONAL PNEUMONIA MODEL (NOT USED BY CHATBOT)
# ------------------------------------------------------------------
PNEUMO_MODEL_PATH = os.path.join(BASE_DIR, "pneuno_ensemble.keras")

try:
    pneumo_model = load_model(PNEUMO_MODEL_PATH)
    app.logger.info("Pneumonia model loaded successfully.")
except Exception as e:
    pneumo_model = None
    app.logger.warning("Pneumonia model not available. Chatbot will still work.")

# ------------------------------------------------------------------
# PERPLEXITY CONFIG
# ------------------------------------------------------------------
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_dcm_to_png(dcm_path, out_png_path):
    import pydicom

    dcm = pydicom.dcmread(dcm_path)
    arr = dcm.pixel_array.astype(np.float32)

    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    amin, amax = arr.min(), arr.max()
    arr = (arr - amin) / (amax - amin + 1e-6)

    img = Image.fromarray((arr * 255).astype(np.uint8)).convert("L")
    img.save(out_png_path)
    return out_png_path


def save_upload_and_prepare(file_storage):
    filename = secure_filename(file_storage.filename)
    uid = uuid.uuid4().hex
    _, ext = os.path.splitext(filename)

    saved_name = f"{uid}{ext}"
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_name)
    file_storage.save(saved_path)

    if ext.lower() == ".dcm":
        png_name = f"{uid}.png"
        png_path = os.path.join(app.config["UPLOAD_FOLDER"], png_name)
        convert_dcm_to_png(saved_path, png_path)
        return png_name, png_path, filename

    return saved_name, saved_path, filename

# ------------------------------------------------------------------
# MODEL-INDEPENDENT CT CHATBOT (CORE FEATURE)
# ------------------------------------------------------------------
def perplexity_ct_chat(user_message):
    if not PERPLEXITY_API_KEY:
        return "Perplexity API key not configured."

    image_path = session.get("ct_image_path")
    chat_history = session.get("chat_history", [])

    if not image_path:
        return "No CT scan uploaded yet."

    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    image_uri = f"data:image/png;base64,{base64_image}"

    system_prompt = """
You are an expert thoracic radiologist.

You are analyzing a chest CT scan visually.

Rules:
- Base answers only on what is visible in the CT scan
- Identify lung regions (upper, middle, lower lobes; left/right)
- Describe patterns (ground-glass opacity, consolidation, reticulation)
- Estimate severity qualitatively (mild / moderate / severe)
- If unsure, say so clearly
- Do NOT reference any AI model or predictions
- Use simple, calm medical language

text format: no bold/italic only plain text
"""

    messages = [{"role": "system", "content": system_prompt}]

    for msg in chat_history:
        messages.append(msg)

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_message},
            {"type": "image_url", "image_url": {"url": image_uri}}
        ]
    })

    payload = {
        "model": "sonar-pro",
        "messages": messages,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "accept": "application/json",
        "content-type": "application/json"
    }

    response = requests.post(
        PERPLEXITY_API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()

    reply = response.json()["choices"][0]["message"]["content"]

    session["chat_history"].append({"role": "user", "content": user_message})
    session["chat_history"].append({"role": "assistant", "content": reply})

    return reply
def get_model_input_hw(model):
    try:
        in_shape = model.inputs[0].shape
        h = int(in_shape[1]) if in_shape[1] is not None else 224
        w = int(in_shape[2]) if in_shape[2] is not None else 224
        return h, w
    except Exception:
        return 224, 224


def predict_pneumonia(image_path: str):
    input_h, input_w = get_model_input_hw(pneumo_model)

    img = load_img(
        image_path,
        color_mode="rgb",
        target_size=(input_h, input_w)
    )

    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = pneumo_model.predict(x)
    out = np.squeeze(preds)

    app.logger.info(
        f"Raw pneumonia prediction output: shape={np.shape(out)}, value={out}"
    )

    # -------------------------------
    # Output handling (UNCHANGED)
    # -------------------------------
    if np.ndim(out) == 0:
        prob_pneumonia = float(out) * 100.0
        prob_normal = 100.0 - prob_pneumonia

    elif np.ndim(out) == 1:
        if out.size == 1:
            prob_pneumonia = float(out[0]) * 100.0
            prob_normal = 100.0 - prob_pneumonia

        elif out.size == 2:
            prob_normal = float(out[0]) * 100.0
            prob_pneumonia = float(out[1]) * 100.0

        else:
            flat = np.asarray(out, dtype=np.float64)
            e = np.exp(flat - np.max(flat))
            probs = e / np.sum(e)
            idx = int(np.argmax(probs))
            probs_pct = (probs * 100.0).tolist()

            classes = [
                {"label": f"Class_{i}", "prob": round(float(probs_pct[i]), 2)}
                for i in range(len(probs_pct))
            ]

            return (
                f"Class_{idx}",
                round(float(probs_pct[idx]), 2),
                classes,
            )

    else:
        prob_pneumonia = float(np.ravel(out)[-1]) * 100.0
        prob_normal = 100.0 - prob_pneumonia

    if prob_pneumonia > prob_normal:
        label = "Pneumonia"
        confidence = round(prob_pneumonia, 2)
    else:
        label = "Normal"
        confidence = round(prob_normal, 2)

    classes = [
        {"label": "Normal", "prob": round(prob_normal, 2)},
        {"label": "Pneumonia", "prob": round(prob_pneumonia, 2)},
    ]

    app.logger.info(
        f"Pneumonia prediction mapped: label={label}, confidence={confidence}"
    )

    return label, confidence, classes

# ------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "ctfile" not in request.files:
        flash("No file part in request.")
        return redirect(url_for("index"))

    file = request.files["ctfile"]

    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Allowed: png, jpg, jpeg, dcm")
        return redirect(url_for("index"))

    try:
        display_filename, feed_image_path, uploaded_name = save_upload_and_prepare(file)
    except Exception as e:
        app.logger.error(f"Upload error: {e}")
        flash("Failed to process uploaded file.")
        return redirect(url_for("index"))

    # -------------------------------------------------
    # MODEL PREDICTION (UNCHANGED LOGIC)
    # -------------------------------------------------
    try:
        if pneumo_model is not None:
            pred_label, confidence, classes = predict_pneumonia(feed_image_path)
        else:
            pred_label = "Model unavailable"
            confidence = 0.0
            classes = []
    except Exception as e:
        app.logger.error(f"Model prediction failed: {e}")
        pred_label = "Prediction failed"
        confidence = 0.0
        classes = []

    # -------------------------------------------------
    # STORE CT FOR CHATBOT (MODEL-INDEPENDENT)
    # -------------------------------------------------
    session["ct_image_path"] = feed_image_path
    session["chat_history"] = []

    display_url = url_for("uploaded_file", filename=display_filename)

    return render_template(
        "result.html",
        pred_label=pred_label,
        confidence=confidence,
        classes=classes,
        uploaded_name=uploaded_name,
        original_img=display_url,
    )


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"reply": "Please ask a question about the CT scan."}), 400

    try:
        reply = perplexity_ct_chat(message)
        return jsonify({"reply": reply})
    except Exception as e:
        app.logger.error(f"Chat error: {e}")
        return jsonify({"reply": "Unable to analyze the CT scan right now."}), 500


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ------------------------------------------------------------------
# RUN
# ------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

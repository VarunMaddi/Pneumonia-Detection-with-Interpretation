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
)
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv(override=True)

# Flask app setup
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "replace-this-with-a-secret")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.logger.setLevel(logging.INFO)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dcm"}

# Load Pneumonia model
PNEUMO_MODEL_PATH = os.path.join(BASE_DIR, "pneuno_ensemble.keras")

try:
    pneumo_model = load_model(PNEUMO_MODEL_PATH)
    app.logger.info(f"Loaded pneumonia model from: {PNEUMO_MODEL_PATH}")
except Exception as e:
    app.logger.exception(f"Failed to load pneumonia model: {e}")
    raise

# Perplexity API config
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model_input_hw(model):
    try:
        in_shape = model.inputs[0].shape
        h = int(in_shape[1]) if in_shape[1] is not None else 224
        w = int(in_shape[2]) if in_shape[2] is not None else 224
        return h, w
    except Exception:
        return 224, 224


def convert_dcm_to_png(dcm_path: str, out_png_path: str) -> str:
    import pydicom
    dcm = pydicom.dcmread(dcm_path)
    arr = dcm.pixel_array.astype(np.float32)
    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept
    amin, amax = arr.min(), arr.max()
    if amax - amin > 0:
        arr_norm = (arr - amin) / (amax - amin)
    else:
        arr_norm = np.zeros_like(arr)
    arr_uint8 = (arr_norm * 255.0).astype(np.uint8)
    img = Image.fromarray(arr_uint8).convert("L")
    img.save(out_png_path)
    return out_png_path


def save_upload_and_prepare(file_storage):
    orig_filename = secure_filename(file_storage.filename)
    uid = uuid.uuid4().hex
    _, ext = os.path.splitext(orig_filename)
    ext = ext.lower()
    saved_name = f"{uid}{ext}"
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_name)
    file_storage.save(saved_path)
    if ext == ".dcm":
        png_name = f"{uid}.png"
        png_path = os.path.join(app.config["UPLOAD_FOLDER"], png_name)
        converted = convert_dcm_to_png(saved_path, png_path)
        return png_name, converted, orig_filename
    else:
        return saved_name, saved_path, orig_filename


def predict_pneumonia(image_path: str):
    input_h, input_w = get_model_input_hw(pneumo_model)
    img = load_img(image_path, color_mode="rgb", target_size=(input_h, input_w))
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = pneumo_model.predict(x)
    out = np.squeeze(preds)
    app.logger.info(f"Raw pneumonia prediction output: shape={np.shape(out)}, value={out}")
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
            if len(probs_pct) >= 2:
                classes = [{"label": f"Class_{i}", "prob": round(float(probs_pct[i]),2)} for i in range(len(probs_pct))]
                label = f"Class_{idx}"
                confidence = round(float(probs_pct[idx]), 2)
                return label, confidence, classes
            else:
                prob_pneumonia = float(probs[0] * 100.0)
                prob_normal = 100.0 - prob_pneumonia
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
    app.logger.info(f"Pneumonia prediction mapped: label={label}, confidence={confidence}")
    return label, confidence, classes


def get_perplexity_multimodal_interpretation(image_path: str):
    if not PERPLEXITY_API_KEY:
        return "Perplexity API key not configured."
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        image_data_uri = f"data:image/png;base64,{base64_image}"
    except Exception as e:
        app.logger.error(f"Failed to encode image for perplexity API: {e}")
        return "Failed to process image for interpretation."

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "accept": "application/json",
        "content-type": "application/json"
    }
    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are an medical imaging expert, analyze the uploaded CT scan image and interpret pneumonia (if detected) in simple words 5-6lines"},
                    {"type": "image_url", "image_url": {"url": image_data_uri}}
                ]
            }
        ],
        "stream": False
    }
    try:
        response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Perplexity API request failed: {e}")
        return f"Interpretation API request failed: {e}"

# Routes
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
        app.logger.error(f"Error processing uploaded file: {e}")
        flash(f"Error processing upload: {e}")
        return redirect(url_for("index"))
    try:
        pred_label, confidence, classes = predict_pneumonia(feed_image_path)
        interpretation = get_perplexity_multimodal_interpretation(feed_image_path)
        display_url = url_for("uploaded_file", filename=display_filename)
        return render_template(
            "result.html",
            pred_label=pred_label,
            confidence=confidence,
            classes=classes,
            uploaded_name=uploaded_name,
            original_img=display_url,
            interpretation=interpretation,
        )
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        flash(f"Error while processing: {e}")
        return redirect(url_for("index"))


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

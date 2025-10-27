import os
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import io

# --- Suppress TensorFlow informational messages ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)

# --- Build Absolute Paths (Guarantees it finds the files) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Point to your specific model files ---
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'bmi_image.pkl')
NUMERIC_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'LassoRegression_bmi.pkl')
NUMERIC_SCALER_PATH = os.path.join(BASE_DIR, 'models', 'numeric_scaler.pkl')
FEATURE_SCALER_PATH = os.path.join(BASE_DIR, 'models', 'feature_scaler.pkl')

# --- Initialize Application ---
app = Flask(__name__)
CORS(app)

# --- Load Models and Scalers ---
try:
    print("--- Loading scikit-learn models and scalers...")
    image_model = joblib.load(IMAGE_MODEL_PATH)
    numeric_model = joblib.load(NUMERIC_MODEL_PATH)
    numeric_scaler = joblib.load(NUMERIC_SCALER_PATH)
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    print("✅ Scikit-learn models loaded successfully.")

    # --- Robust EfficientNetB0 Loading ---
    print("\n--- Loading TensorFlow EfficientNetB0 model...")
    try:
        print("Trying to load with pre-trained 'imagenet' weights...")
        base_model = EfficientNetB0(
            weights='imagenet', 
            include_top=False, 
            pooling='avg', 
            input_shape=(224, 224, 3)
        )
        print("✅ EfficientNetB0 with 'imagenet' weights loaded successfully.")
    except ValueError as e:
        print(f"⚠️ Primary attempt failed with error: {e}")
        print("Falling back to loading EfficientNetB0 with NO pre-trained weights...")
        base_model = EfficientNetB0(
            weights=None,  
            include_top=False, 
            pooling='avg', 
            input_shape=(224, 224, 3)
        )
        print("✅ EfficientNetB0 loaded successfully (without weights).")

except FileNotFoundError as e:
    print(f"❌ CRITICAL ERROR: Missing model file: {e}")
    print("Ensure your 'models' folder is in the same directory as app.py.")
    exit()
except Exception as e:
    print(f"❌ Unexpected critical error during model loading: {e}")
    exit()


# --- Helper Functions ---
def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    elif 30 <= bmi < 35:
        return "Obese (Moderate)"
    elif 35 <= bmi < 40:
        return "Obese (Severe)"
    else:
        return "Obese (Very Severe)"

def get_image_embedding(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((224, 224))
    if img.mode != "RGB":
        img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    embedding = base_model.predict(x)
    return embedding

# --- Routes ---
@app.route('/')
def home():
    return jsonify({"message": "🏥 BMI Prediction API is running successfully!"})

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image_embedding = get_image_embedding(file.read())
        padded_features = np.zeros((1, 1283))
        padded_features[0, :1280] = image_embedding
        scaled_features = feature_scaler.transform(padded_features)
        prediction = image_model.predict(scaled_features[:, :1280])
        bmi = float(prediction[0])
        category = get_bmi_category(bmi)
        return jsonify({'bmi': bmi, 'category': category})
    except Exception as e:
        return jsonify({'error': f'Image prediction failed: {str(e)}'}), 500

@app.route('/predict_numerical', methods=['POST'])
def predict_numerical():
    data = request.get_json()
    height, weight, age = data.get('height'), data.get('weight'), data.get('age')
    if not all([height, weight, age]):
        return jsonify({'error': 'Missing numerical data.'}), 400

    try:
        numerical_features = np.array([[float(height), float(weight), float(age)]])
        scaled_numerical_features = numeric_scaler.transform(numerical_features)
        padded_features = np.hstack([np.zeros((1, 1280)), scaled_numerical_features])
        prediction = numeric_model.predict(padded_features)
        bmi = float(prediction[0])
        category = get_bmi_category(bmi)
        return jsonify({'bmi': bmi, 'category': category})
    except Exception as e:
        return jsonify({'error': f'Numerical prediction failed: {str(e)}'}), 500

@app.route('/calorie_calculator', methods=['POST'])
def calorie_calculator():
    data = request.get_json()
    gender = data.get('gender')
    height = data.get('height')
    weight = data.get('weight')
    age = data.get('age')
    activity = data.get('activity')

    if not all([gender, height, weight, age, activity]):
        return jsonify({'error': 'Missing data for calorie calculation.'}), 400

    try:
        if gender.lower() == 'male':
            bmr = 10 * float(weight) + 6.25 * float(height) - 5 * float(age) + 5
        elif gender.lower() == 'female':
            bmr = 10 * float(weight) + 6.25 * float(height) - 5 * float(age) - 161
        else:
            return jsonify({'error': 'Invalid gender'}), 400

        activity_factors = {
            'sedentary': 1.2,
            'lightly active': 1.375,
            'moderately active': 1.55,
            'very active': 1.725,
            'extra active': 1.9
        }
        factor = activity_factors.get(activity.lower())
        if not factor:
            return jsonify({'error': 'Invalid activity level'}), 400

        daily_calories = bmr * factor
        return jsonify({'bmr': bmr, 'daily_calories': daily_calories})
    except Exception as e:
        return jsonify({'error': f'Calorie calculator failed: {str(e)}'}), 500


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    print("\n--- Starting Flask Server ---")
    port = int(os.environ.get("PORT", 5000))  # Render uses dynamic port
    app.run(host='0.0.0.0', port=port, debug=False)

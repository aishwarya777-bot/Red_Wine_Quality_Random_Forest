import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the model safely
MODEL_PATH = "random_forest.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "pickle" in MODEL_PATH and "rb" or "r") as f:
        model = pickle.load(f)
else:
    model = None

# Integrated Attractive HTML/CSS Layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Prediction Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --card-bg: rgba(255, 255, 255, 0.03);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --success-color: #10b981;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-gradient);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 800px;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            animation: fadeIn 0.6s ease-out;
        }

        h1 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(to right, #fff, #a5b4fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 0.95rem;
            text-align: center;
            margin-bottom: 2.5rem;
        }

        /* Result Banner Styling */
        .result-container {
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            text-align: center;
            animation: slideDown 0.4s ease-out;
        }
        .result-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 0.25rem;
        }
        .result-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--success-color);
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
        }

        @media (max-width: 600px) {
            .grid-form {
                grid-template-columns: 1fr;
            }
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-secondary);
            text-transform: capitalize;
        }

        input {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.95rem;
            transition: all 0.2s ease;
        }

        input:focus {
            outline: none;
            border-color: var(--accent-color);
            background: rgba(255, 255, 255, 0.08);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        .btn-submit {
            grid-column: 1 / -1;
            background: var(--accent-color);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 1rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            margin-top: 1rem;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }

        .btn-submit:hover {
            background: var(--accent-hover);
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(99, 102, 241, 0.4);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container">
    <h1>Model Prediction Dashboard</h1>
    <p class="subtitle">Enter the characteristic metrics below to compute model inference</p>

    {% if prediction is not none %}
    <div class="result-container">
        <div class="result-title">Prediction Output</div>
        <div class="result-value">{{ prediction }}</div>
    </div>
    {% endif %}

    <form action="/" method="POST" class="grid-form">
        {% for feature in features %}
        <div class="input-group">
            <label for="{{ feature }}">{{ feature }}</label>
            <input 
                type="number" 
                step="any" 
                name="{{ feature }}" 
                id="{{ feature }}" 
                value="{{ last_inputs.get(feature, '') }}" 
                required
            >
        </div>
        {% endfor %}
        <button type="submit" class="btn-submit">Generate Prediction</button>
    </form>
</div>

</body>
</html>
"""

# Ordered list of features parsed from the provided pickle file
FEATURE_NAMES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    last_inputs = {}
    
    if request.method == "POST":
        try:
            # Gather features in exact order needed by the model
            input_data = []
            for feature in FEATURE_NAMES:
                val = float(request.form.get(feature, 0.0))
                input_data.append(val)
                last_inputs[feature] = val # retain input value in form UI
            
            if model:
                # Convert to shape (1, n_features) and predict
                array_data = np.array([input_data])
                pred_out = model.predict(array_data)
                prediction = str(pred_out[0])
            else:
                prediction = "Error: model file 'random_forest.pkl' not found on server context."
        except Exception as e:
            prediction = f"Inference Error: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE, 
        features=FEATURE_NAMES, 
        prediction=prediction,
        last_inputs=last_inputs
    )

if __name__ == "__main__":
    # Render binds dynamic port configuration variables
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

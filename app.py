from flask import Flask, request, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

# ==============================
# Load XGBoost Model
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_model.pkl")

model = joblib.load(MODEL_PATH)


# ==============================
# HTML + CSS
# ==============================
HTML = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Insurance Cost Predictor</title>

    <style>

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: Arial, sans-serif;
        }

        body {
            min-height: 100vh;
            background: linear-gradient(135deg, #0f172a, #1e3a8a);
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 30px;
        }

        .container {
            width: 100%;
            max-width: 850px;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 25px;
        }

        .header h1 {
            font-size: 38px;
            margin-bottom: 10px;
        }

        .header p {
            color: #cbd5e1;
            font-size: 16px;
        }

        .card {
            background: rgba(255,255,255,0.97);
            border-radius: 22px;
            padding: 35px;
            box-shadow: 0 20px 50px rgba(0,0,0,0.3);
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-weight: bold;
            margin-bottom: 8px;
            color: #1e293b;
        }

        input,
        select {
            padding: 14px;
            border-radius: 10px;
            border: 1px solid #cbd5e1;
            font-size: 15px;
            outline: none;
            background: #f8fafc;
        }

        input:focus,
        select:focus {
            border-color: #2563eb;
            box-shadow: 0 0 0 3px rgba(37,99,235,0.1);
        }

        .full {
            grid-column: 1 / -1;
        }

        button {
            width: 100%;
            margin-top: 25px;
            padding: 16px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: white;
            font-size: 17px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(37,99,235,0.3);
        }

        .result {
            margin-top: 25px;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            background: linear-gradient(135deg, #dcfce7, #dbeafe);
        }

        .result-title {
            color: #475569;
            font-size: 15px;
            margin-bottom: 8px;
        }

        .result-value {
            font-size: 36px;
            font-weight: bold;
            color: #166534;
        }

        .error {
            margin-top: 20px;
            padding: 15px;
            border-radius: 10px;
            background: #fee2e2;
            color: #991b1b;
            text-align: center;
        }

        .footer {
            text-align: center;
            color: #94a3b8;
            margin-top: 20px;
            font-size: 13px;
        }

        @media(max-width: 650px) {

            .form-grid {
                grid-template-columns: 1fr;
            }

            .full {
                grid-column: auto;
            }

            .card {
                padding: 22px;
            }

            .header h1 {
                font-size: 28px;
            }
        }

    </style>
</head>


<body>

<div class="container">

    <div class="header">

        <h1>💳 Insurance Cost Predictor</h1>

        <p>
            XGBoost Machine Learning Prediction System
        </p>

    </div>


    <div class="card">

        <form method="POST">

            <div class="form-grid">


                <!-- Age -->
                <div class="form-group">

                    <label>Age</label>

                    <input
                        type="number"
                        name="age"
                        placeholder="Enter age"
                        min="1"
                        max="120"
                        required
                    >

                </div>


                <!-- BMI -->
                <div class="form-group">

                    <label>BMI</label>

                    <input
                        type="number"
                        name="bmi"
                        placeholder="Enter BMI"
                        step="0.01"
                        min="1"
                        required
                    >

                </div>


                <!-- Children -->
                <div class="form-group">

                    <label>Children</label>

                    <input
                        type="number"
                        name="children"
                        placeholder="Number of children"
                        min="0"
                        required
                    >

                </div>


                <!-- Sex -->
                <div class="form-group">

                    <label>Sex</label>

                    <select name="sex" required>

                        <option value="">Select Sex</option>

                        <option value="female">
                            Female
                        </option>

                        <option value="male">
                            Male
                        </option>

                    </select>

                </div>


                <!-- Smoker -->
                <div class="form-group">

                    <label>Smoker</label>

                    <select name="smoker" required>

                        <option value="">
                            Select Option
                        </option>

                        <option value="no">
                            No
                        </option>

                        <option value="yes">
                            Yes
                        </option>

                    </select>

                </div>


                <!-- Region -->
                <div class="form-group">

                    <label>Region</label>

                    <select name="region" required>

                        <option value="">
                            Select Region
                        </option>

                        <option value="northeast">
                            Northeast
                        </option>

                        <option value="northwest">
                            Northwest
                        </option>

                        <option value="southeast">
                            Southeast
                        </option>

                        <option value="southwest">
                            Southwest
                        </option>

                    </select>

                </div>

            </div>


            <button type="submit">
                🔮 Predict Insurance Cost
            </button>

        </form>


        {% if prediction %}

        <div class="result">

            <div class="result-title">
                Estimated Insurance Cost
            </div>

            <div class="result-value">
                ₹ {{ prediction }}
            </div>

        </div>

        {% endif %}


        {% if error %}

        <div class="error">

            ❌ {{ error }}

        </div>

        {% endif %}

    </div>


    <div class="footer">

        Powered by Flask + XGBoost

    </div>

</div>

</body>

</html>
"""


# ==============================
# Encoding
# ==============================

SEX = {
    "female": 0,
    "male": 1
}

SMOKER = {
    "no": 0,
    "yes": 1
}

REGION = {
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
}


# ==============================
# Home Route
# ==============================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":

        try:

            age = float(request.form["age"])

            bmi = float(request.form["bmi"])

            children = int(request.form["children"])

            sex = request.form["sex"]

            smoker = request.form["smoker"]

            region = request.form["region"]


            # ==============================
            # Prepare Input
            # ==============================

            data = np.array([[
                age,
                SEX[sex],
                bmi,
                children,
                SMOKER[smoker],
                REGION[region]
            ]])


            # ==============================
            # Prediction
            # ==============================

            result = model.predict(data)

            prediction = f"{float(result[0]):,.2f}"


        except Exception as e:

            error = str(e)


    return render_template_string(
        HTML,
        prediction=prediction,
        error=error
    )


# ==============================
# Run Application
# ==============================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        age = float(request.form["age"])
        gender = int(request.form["gender"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        ap_hi = float(request.form["ap_hi"])
        ap_lo = float(request.form["ap_lo"])
        cholesterol = int(request.form["cholesterol"])
        gluc = int(request.form["gluc"])
        smoke = int(request.form["smoke"])
        alco = int(request.form["alco"])
        active = int(request.form["active"])

        # Create feature array
        features = np.array([[
            age,
            gender,
            height,
            weight,
            ap_hi,
            ap_lo,
            cholesterol,
            gluc,
            smoke,
            alco,
            active
        ]])

        # Scale features
        features = scaler.transform(features)

        # Predict
        prediction = model.predict(features)[0]

        # Render result page with all details
        return render_template(
            "result.html",
            prediction=prediction,
            age=age,
            gender="Male" if gender == 2 else "Female",
            height=height,
            weight=weight,
            ap_hi=ap_hi,
            ap_lo=ap_lo,
            cholesterol={
                1: "Normal",
                2: "Above Normal",
                3: "Well Above Normal"
            }[cholesterol],
            gluc={
                1: "Normal",
                2: "Above Normal",
                3: "Well Above Normal"
            }[gluc],
            smoke="Yes" if smoke == 1 else "No",
            alco="Yes" if alco == 1 else "No",
            active="Active" if active == 1 else "Not Active"
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
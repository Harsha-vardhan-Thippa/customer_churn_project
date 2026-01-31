from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
with open("customer_churn.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        # ---------- Numeric ----------
        tenure = float(request.form["tenure"])
        total_charges = float(request.form["total_charges"])

        # ---------- Binary ----------
        gender_male = 1 if request.form["gender"] == "Male" else 0
        partner_yes = 1 if request.form["partner"] == "Yes" else 0
        dependents_yes = 1 if request.form["dependents"] == "Yes" else 0
        phone_yes = 1 if request.form["phone"] == "Yes" else 0
        paperless_yes = 1 if request.form["paperless"] == "Yes" else 0

        # ---------- Multiple lines ----------
        ml_no_phone = 1 if request.form["multiple_lines"] == "No phone service" else 0
        ml_yes = 1 if request.form["multiple_lines"] == "Yes" else 0

        # ---------- Internet ----------
        net_fiber = 1 if request.form["internet"] == "Fiber optic" else 0
        net_no = 1 if request.form["internet"] == "No" else 0

        # ---------- Payment ----------
        pay_cc = 1 if request.form["payment"] == "Credit card (automatic)" else 0
        pay_ec = 1 if request.form["payment"] == "Electronic check" else 0
        pay_mc = 1 if request.form["payment"] == "Mailed check" else 0

        # ---------- Ordinal ----------
        ordinal_map = {
            "No internet service": 0,
            "No": 1,
            "Yes": 2
        }

        contract_map = {
            "Month-to-month": 0,
            "One year": 1,
            "Two year": 2
        }

        online_sec = ordinal_map[request.form["online_security"]]
        online_backup = ordinal_map[request.form["online_backup"]]
        device_protect = ordinal_map[request.form["device_protection"]]
        tech_support = ordinal_map[request.form["tech_support"]]
        stream_tv = ordinal_map[request.form["stream_tv"]]
        stream_movies = ordinal_map[request.form["stream_movies"]]
        contract = contract_map[request.form["contract"]]

        # ---------- Final feature vector ----------
        input_data = np.array([[
            tenure,
            total_charges,
            gender_male,
            partner_yes,
            dependents_yes,
            phone_yes,
            ml_no_phone,
            ml_yes,
            net_fiber,
            net_no,
            paperless_yes,
            pay_cc,
            pay_ec,
            pay_mc,
            online_sec,
            online_backup,
            device_protect,
            tech_support,
            stream_tv,
            stream_movies,
            contract
        ]])

        # ---------- SCALE ----------
        input_scaled = scaler.transform(input_data)

        # ---------- PREDICT ----------
        result = model.predict(input_scaled)[0]
        prediction = "CHURN" if result == 1 else "NOT CHURN"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import sklearn

app = Flask(__name__)
model  = pickle.load(open("model.pkl","rb"))




@app.route("/predict",methods=["GET"])
def predict_form():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # Extract values from the JSON data
    gender = data.get("Gender")
    married = data.get("Married")
    dependents = data.get("Dependents")
    education = data.get("Education")
    self_employed = data.get("Self_Employed")
    applicant_income = float(data.get("ApplicantIncome", 0))
    coapplicant_income = float(data.get("CoapplicantIncome", 0))
    loan_amount = float(data.get("LoanAmount", 0))
    loan_amount_term = int(data.get("Loan_Amount_Term", 0))
    credit_history = int(data.get("Credit_History", 0))
    property_area = data.get("Property_Area")

    # --- Model prediction logic: match notebook preprocessing ---
    # Gender (OneHot)
    gender_male = 1 if gender == "Male" else 0
    gender_female = 1 if gender == "Female" else 0
    # Married (LabelEncoder)
    married_val = 1 if married == "Yes" else 0
    # Dependents (OrdinalEncoder)
    dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
    dependents_val = dependents_map.get(dependents, 0)
    # Education (OrdinalEncoder)
    education_val = 1 if education == "Graduate" else 0
    # Self_Employed (OneHot)
    self_employed_yes = 1 if self_employed == "Yes" else 0
    self_employed_no = 1 if self_employed == "No" else 0
    # Property_Area (OneHot)
    property_area_urban = 1 if property_area == "Urban" else 0
    property_area_semiurban = 1 if property_area == "Semiurban" else 0
    property_area_rural = 1 if property_area == "Rural" else 0
    # Credit_History (LabelEncoder)
    credit_history_val = int(credit_history)

    # Prepare features for the model (order must match training)
    features = [
        gender_male, gender_female,
        married_val,
        dependents_val,
        education_val,
        self_employed_yes, self_employed_no,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history_val,
        property_area_urban, property_area_semiurban, property_area_rural
    ]
    # Use predict_proba to get probability of approval (class 1) if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba([features])[0][1]  # Probability of class 1 (Approved)
        output = round(float(proba), 2)
        if output > 0.5:
            result = f"Loan Approved (Probability: {output})"
        else:
            result = f"Loan Not Approved (Probability: {output})"
    else:
        prediction = model.predict([features])
        output = round(float(prediction[0]), 2)
        if output > 0.5:
            result = f"Loan Approved (Score: {output})"
        else:
            result = f"Loan Not Approved (Score: {output})"
    return jsonify({"message": result})


if __name__ == "__main__":
    app.run(debug=True)

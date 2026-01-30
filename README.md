# Loan Approval Prediction Web App

This project is a Flask-based web application for predicting loan approval using a machine learning model trained in a Jupyter Notebook. The app takes user input from a web form, encodes features as per the model's training pipeline, and returns the probability of loan approval.

## Prerequisites
- Python 3.7+
- pip (Python package manager)
- Required Python packages (see below)
- Trained model file: `model.pkl` (should be present in the project directory)

## Installation Steps
1. **Clone or Download the Repository**
   - Place all files in a single folder (e.g., `Datamites_proc_1`).

2. **Navigate to the Project Directory**
   ```sh
   cd "D:\data mites\Datamites_proc_1"
   ```

3. **(Optional) Create a Virtual Environment**
   ```sh
   python -m venv venv
   venv\Scripts\activate
   ```

4. **Install Required Packages**
   ```sh
   pip install flask scikit-learn pandas numpy
   ```
   - If you used a Jupyter Notebook for training, you may also need:
     ```sh
     pip install jupyter
     ```

5. **Ensure `model.pkl` is Present**
   - The trained model file (`model.pkl`) should be in the same directory as `main.py`.

## Running the Application
1. **Start the Flask App**
   ```sh
   python main.py
   ```
   - Or, if using Flask CLI:
     ```sh
     flask run
     ```

2. **Access the Web App**
   - Open your browser and go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

3. **Use the Form**
   - Fill in all required fields and submit to get the loan approval probability.

## Troubleshooting
- If you get import errors, ensure all required packages are installed.
- If you see feature mismatch errors, verify that the form fields and backend encoding match the model's expected input.
- If the app does not start, check for errors in the terminal and resolve missing dependencies.

## File Structure
- `main.py` — Flask backend
- `index.html` — Frontend form (if present)
- `model.pkl` — Trained machine learning model
- `README.md` — This file

## License
This project is for educational purposes.

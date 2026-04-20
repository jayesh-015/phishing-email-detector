from flask import Flask, request, render_template
import joblib
import pandas as pd
from utils import clean_text

app = Flask(__name__)

# Load the trained model globally
try:
    model = joblib.load('model/phishing_model.pkl')
except FileNotFoundError:
    model = None
    print("Warning: Model not found. Please run train.py first.")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    email_content = ""
    
    if request.method == 'POST':
        if not model:
            return "Model not loaded. Train the model first.", 500
            
        email_content = request.form['email_content']
        
        # Prepare data exactly as the pipeline expects
        input_data = pd.DataFrame({
            'text': [email_content],
            'clean_text': [clean_text(email_content)]
        })
        
        # Predict (0 = Safe, 1 = Phishing)
        result = model.predict(input_data)[0]
        prediction = "🚨 Phishing Detected!" if result == 1 else "✅ Safe Email"

    return render_template('index.html', prediction=prediction, email_content=email_content)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Load model and tokenizer
MODEL_PATH = "kabirj25/distilbert_fraud_revs"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode

# Label mapping
label_map = {1: "Genuine", 0: "Fraud"}



@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    original_review = ""  
    review_text = ""

    if request.method == "POST":
        original_review = request.form["review"]
        review_text="Classify the app as fraud or genuine given the review: " + original_review
        
        # Preprocess input
        inputs = tokenizer(review_text, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            print(predicted_class)
        
        prediction = label_map[predicted_class]

    return render_template("index.html", prediction=prediction, review_text=original_review)

if __name__ == "__main__":
    app.run(debug=True)
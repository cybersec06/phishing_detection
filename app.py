import re
import joblib
import string
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

app = Flask(__name__)

# ---------- Load and Train ML Model (for demo) ----------
print("[INFO] Training model on sample data...")

# Sample dataset (phishing=1, legitimate=0)
data = {
    "subject": [
        "Urgent: Account Suspended",
        "Congratulations! You've won",
        "Meeting Agenda for Tomorrow",
        "Verify your login credentials",
        "Lunch Invitation"
    ],
    "body": [
        "Please verify your account to avoid suspension.",
        "Click here to claim your prize now!",
        "Attached is the agenda for tomorrow's meeting.",
        "Login now to secure your account.",
        "Let's have lunch at 1 PM."
    ],
    "label": [1, 1, 0, 1, 0]
}
df = pd.DataFrame(data)
df['text'] = df['subject'] + ' ' + df['body']

# Text preprocessing
def preprocess_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower()

df['clean_text'] = df['text'].apply(preprocess_text)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("[INFO] Model trained. Classification report:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ---------- Rule-Based Detection ----------
SUSPICIOUS_KEYWORDS = ["urgent", "verify", "account suspended", "click here", "password expired"]

def rule_based_detection(subject, body):
    text = f"{subject} {body}".lower()
    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in text:
            return 1  # phishing
    return 0

# ---------- ML Prediction ----------
def ml_prediction(text):
    processed_text = preprocess_text(text)
    features = vectorizer.transform([processed_text])
    return model.predict(features)[0]

# ---------- Combined Detection ----------
def classify_email(subject, body):
    rule_score = rule_based_detection(subject, body)
    ml_score = ml_prediction(f"{subject} {body}")
    final = 1 if (rule_score + ml_score) > 0 else 0
    return {"rule_based": rule_score, "ml_based": int(ml_score), "final_decision": final}

# ---------- Email Fetching & Auto Detection ----------
IMAP_SERVER = "imap.gmail.com"
EMAIL_ACCOUNT = "your.email@gmail.com"  # replace with your Gmail
EMAIL_PASSWORD = "your-app-password"     # use an App Password from Gmail settings

def fetch_and_scan_emails():
    print("[INFO] Checking inbox for new emails...")
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
        mail.select("inbox")
        result, data = mail.search(None, 'UNSEEN')
        email_ids = data[0].split()

        for e_id in email_ids:
            result, msg_data = mail.fetch(e_id, "(RFC822)")
            raw_email = msg_data[0][1]
            message = email.message_from_bytes(raw_email)

            subject = message["subject"]
            body = ""

            if message.is_multipart():
                for part in message.walk():
                    if part.get_content_type() == "text/plain":
                        body += part.get_payload(decode=True).decode()
            else:
                body = message.get_payload(decode=True).decode()

            result = classify_email(subject, body)
            print(f"\n[EMAIL] Subject: {subject}")
            print(f"[RESULT] {result}")

            if result['final_decision'] == 1:
                with open("phishing_log.txt", "a") as log:
                    log.write(f"PHISHING DETECTED:\nSubject: {subject}\nBody: {body}\n\n")

    except Exception as e:
        print(f"[ERROR] {e}")

# ---------- Flask Endpoint ----------
@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    subject = data.get('subject', '')
    body = data.get('body', '')
    result = classify_email(subject, body)
    return jsonify(result)

if __name__ == '__main__':
    print("[INFO] APDT Flask API running at http://127.0.0.1:5000/detect")
    app.run(debug=True)

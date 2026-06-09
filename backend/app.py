from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

RISK_DB = {
    "sugar": {"type": "health", "msg": "High sugar increases diabetes risk"},
    "oil": {"type": "health", "msg": "Excess oil leads to heart issues"},
    "interest": {"type": "finance", "msg": "High interest increases repayment"},
    "penalty": {"type": "finance", "msg": "Hidden charges possible"},
}

def extract_keywords(text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([text])
    words = vectorizer.get_feature_names_out()
    scores = tfidf.toarray()[0]

    word_scores = dict(zip(words, scores))
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

    return [w for w, _ in sorted_words[:10]]
@app.route('/')
def home():
    return "Backend is running successfully!"


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get("text", "")

    keywords = extract_keywords(text)

    risks = []
    for word in keywords:
        if word in RISK_DB:
            risks.append({
                "word": word,
                "type": RISK_DB[word]["type"],
                "message": RISK_DB[word]["msg"]
            })

    return jsonify({
        "keywords": keywords,
        "risks": risks,
        "risk_score": len(risks)
    })

if __name__ == '__main__':
    app.run(debug=True)

   
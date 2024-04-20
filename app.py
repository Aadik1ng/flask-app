from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import numpy as np

app = Flask(__name__)
nlp = spacy.load("en_core_web_lg")

def preprocess_text(text):
    doc = nlp(text)
    cleaned_text = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    return cleaned_text

def calculate_similarity(text1, text2):
    text1_cleaned = preprocess_text(text1)
    text2_cleaned = preprocess_text(text2)
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1_cleaned, text2_cleaned])
    
    similarity_score_tfidf = cosine_similarity(tfidf_matrix)[0, 1]
    similarity_score_embeddings = embed(text1_cleaned, text2_cleaned)
    
    # Convert similarity scores to standard Python floats
    similarity_score_tfidf = float(similarity_score_tfidf)
    similarity_score_embeddings = float(similarity_score_embeddings)
    
    return similarity_score_tfidf, similarity_score_embeddings

def embed(text1_cleaned, text2_cleaned):
    vector1 = nlp(text1_cleaned).vector
    vector2 = nlp(text2_cleaned).vector
    
    similarity_score_embeddings = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0, 0]
    
    return similarity_score_embeddings

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home', methods=['POST'])
def home():
    if not request.json or 'text1' not in request.json or 'text2' not in request.json:
        return jsonify({'error': 'Invalid request. JSON data with "text1" and "text2" keys is required.'}), 400

    text1 = request.json['text1']
    text2 = request.json['text2']

    try:
        similarity_score_tfidf, similarity_score_embeddings = calculate_similarity(text1, text2)

        response = {
            'similarity_score_using_TFIDF': similarity_score_tfidf,
            'similarity_score_using_embeddings': similarity_score_embeddings
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

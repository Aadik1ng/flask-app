
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text)
    cleaned_text = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    return cleaned_text

def calculate_similarity(text1, text2):
    text1_cleaned = preprocess_text(text1)
    text2_cleaned = preprocess_text(text2)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1_cleaned, text2_cleaned])

    similarity_score = cosine_similarity(tfidf_matrix)[0, 1]
    similarity_score = np.float64(similarity_score)

    return similarity_score

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/home/")
async def home(request: Request):
    data = await request.json()

    if not data or 'text1' not in data or 'text2' not in data:
        raise HTTPException(status_code=400, detail='Invalid request. JSON data with "text1" and "text2" keys is required.')

    text1 = data['text1']
    text2 = data['text2']

    try:
        similarity_score_tfidf = calculate_similarity(text1, text2)
        similarity_score_embeddings = embed(text1, text2)

        response = {
            'similarity_score_using_TFIDF': similarity_score_tfidf,
            'similarity_score_using_embeddings': similarity_score_embeddings
        }

        return JSONResponse(content=response, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def embed(text1, text2):
    text1_cleaned = preprocess_text(text1)
    text2_cleaned = preprocess_text(text2)

    vector1 = nlp(text1_cleaned).vector
    vector2 = nlp(text2_cleaned).vector

    similarity_score = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0, 0]
    similarity_score = np.float64(similarity_score)

    return similarity_score

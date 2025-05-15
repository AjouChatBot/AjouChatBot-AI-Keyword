from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from konlpy.tag import Okt
from typing import List
import numpy as np

keyword_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
kw_model = KeyBERT(keyword_model)
okt = Okt()

from transformers import XLMRobertaTokenizer, AutoModelForSequenceClassification
from transformers.pipelines import ZeroShotClassificationPipeline

tokenizer = XLMRobertaTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
classifier_model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")

classifier = ZeroShotClassificationPipeline(model=classifier_model, tokenizer=tokenizer)

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    noun_candidates = okt.nouns(text)
    candidates = list(set(noun_candidates))
    print("후보 명사들:", candidates)

    keywords = kw_model.extract_keywords(
        text,
        candidates=candidates,
        keyphrase_ngram_range=(1, 2),
        stop_words=None,
        top_n=top_n
    )
    for word, score in keywords:
        print(f"{word}: {score:.4f}")

    result = [word for word, _ in keywords]
    return result

ENTITY_CANDIDATES = [
    "편의점", "식당", "도서관", "카페", "강의실", "기숙사", "체육관", "서점", "학생회관", "전공필수", "과목", "사이버보안학과", "장학금"
]

def extract_target_entity(text: str):
    text_embedding = keyword_model.encode(text, convert_to_numpy=True)
    candidate_embeddings = keyword_model.encode(ENTITY_CANDIDATES, convert_to_numpy=True)
    similarities = np.dot(candidate_embeddings, text_embedding)

    best_idx = int(np.argmax(similarities))
    best_entity = ENTITY_CANDIDATES[best_idx]

    print(f"[DEBUG] 선택된 주제어: {best_entity} (유사도: {similarities[best_idx]:.4f})")
    return best_entity

CATEGORY_LABELS = ["입학", "학적", "학칙", "장학"]

def classify_categories(text: str) -> str:
    result = classifier(text, CATEGORY_LABELS)
    return result["labels"][0]
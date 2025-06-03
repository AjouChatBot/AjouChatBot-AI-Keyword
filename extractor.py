from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from konlpy.tag import Okt
from typing import List
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

keyword_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
kw_model = KeyBERT(keyword_model)
okt = Okt()

from transformers import XLMRobertaTokenizer, AutoModelForSequenceClassification
from transformers.pipelines import ZeroShotClassificationPipeline

tokenizer = XLMRobertaTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
classifier_model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")

classifier = ZeroShotClassificationPipeline(model=classifier_model, tokenizer=tokenizer)

from pymilvus import connections, Collection

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")

connections.connect(
    host=MILVUS_HOST,
    port=MILVUS_PORT
)

collection_name = "a_mate_keywords"
collection = Collection(collection_name)
collection.load()

def get_ada_embedding(text: str):
    return client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    ).data[0].embedding

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    noun_candidates = okt.nouns(text)
    candidates = list(set(noun_candidates))

    keywords = kw_model.extract_keywords(
        text,
        candidates=candidates,
        keyphrase_ngram_range=(1, 2),
        stop_words=None,
        top_n=top_n
    )

    result = [word for word, _ in keywords]
    return result

def extract_target_entity(text: str):
    text_embedding = get_ada_embedding(text)

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = collection.search(
        data=[text_embedding],
        anns_field="vector",
        param=search_params,
        limit=5,  # ← 유사 키워드 5개 뽑기
        output_fields=["keyword"]
    )

    top_keywords = [hit.entity.get("keyword") for hit in results[0]]

    return top_keywords[0] if top_keywords else ""

CATEGORY_LABELS = ["대학 소개", "연구", "일정", "학사정보", "강의", "등록", "장학", "교수", "시설", "서비스", "학생활동 및 동아리", "공지사항", "입학"]

def classify_categories(text: str) -> str:
    result = classifier(text, CATEGORY_LABELS)
    return result["labels"][0]
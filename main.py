from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from extractor import extract_keywords, extract_target_entity, classify_categories
import asyncio

app = FastAPI()

# 요청 바디
class TextInput(BaseModel):
    text: str

# 응답 바디
class KeywordResponse(BaseModel):
    keywords: List[str]

class CombinedResponse(BaseModel):
    keywords: List[str]
    target: str
    category: str

# @app.post("/keyword", response_model=KeywordResponse)
# async def extract(text_input: TextInput):
#     keywords = extract_keywords(text_input.text)
#     return {"keywords": keywords}
#
# @app.post("/target", response_model=KeywordResponse)
# async def extract(text_input: TextInput):
#     keywords = extract_target_entity(text_input.text)
#     return {"keywords": [keywords]}
#
# @app.post("/category", response_model=KeywordResponse)
# async def extract(text_input: TextInput):
#     keywords = classify_categories(text_input.text)
#     return {"keywords": [keywords]}

@app.post("/keyword", response_model=CombinedResponse)
async def extract_all(text_input: TextInput):
    text = text_input.text

    # 세 작업을 병렬 실행
    keywords_task = asyncio.to_thread(extract_keywords, text)
    target_task = asyncio.to_thread(extract_target_entity, text)
    category_task = asyncio.to_thread(classify_categories, text)

    keywords, target, category = await asyncio.gather(
        keywords_task,
        target_task,
        category_task
    )

    return {
        "keywords": keywords,
        "target": target,
        "category": category
    }

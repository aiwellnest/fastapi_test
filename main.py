from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

class Item(BaseModel):
    content: str

@app.post("/ask-question/")
async def example(item: Item):
    return {"content": item.content}

from fastapi import FastAPI

# Initialize FastAPI app
app = FastAPI()

class Item(BaseModel):
    content: str

@app.post("/ask-question/")
async def example(item: Item):
    return {"content": item.content}
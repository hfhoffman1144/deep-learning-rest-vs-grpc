from fastapi import FastAPI
import asyncio
from schemas.model_inputs import EmbeddingRequest, EmbeddingResponse
from sentence_transformers import SentenceTransformer

REQUESTED_DEIVCE = "mps"
MODEL = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2",
                            device=REQUESTED_DEIVCE)


# Instantiate FastAPI app
app = FastAPI(title="Embedding REST API Server")

# Create the event loop
loop = asyncio.get_event_loop()


@app.get("/")
async def get_base_route():
    return "FastAPI is running"


@app.post("/text-embeddings/")
async def create_text_embeddings(request: EmbeddingRequest):

    embeddings_list = MODEL.encode(request.texts).tolist()
    response = EmbeddingResponse(embeddings=embeddings_list)

    return response

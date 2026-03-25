import os
import uuid
from dotenv import load_dotenv
import torch
from PIL import Image
import requests
from io import BytesIO
import threading
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import uvicorn

load_dotenv()

class CLIPEmbeddingStore:
    def __init__(self, qdrant_url, qdrant_api_key, collection_name="image_embeddings_512"):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name
        self._create_collection()

    def _create_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
            print(f"[OK] Created collection: {self.collection_name}")
        else:
            print(f"[OK] Using existing collection: {self.collection_name}")

    def _load_image(self, image_path_or_url):
        if image_path_or_url.startswith("http"):
            response = requests.get(image_path_or_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        else:
            if not os.path.exists(image_path_or_url):
                raise FileNotFoundError(f"Image not found: {image_path_or_url}")
            image = Image.open(image_path_or_url)
        return image.convert("RGB")

    def _get_image_embedding(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        # transformers versions may return either a Tensor or an object with pooler_output.
        pooled = image_features.pooler_output if hasattr(image_features, "pooler_output") else image_features
        
        # L2 normalize
        pooled = pooled / torch.norm(pooled, dim=-1, keepdim=True)
        
        # Extract [512]
        embedding = pooled[0].cpu().numpy().tolist()
        
        print(f"Debug: embedding len={len(embedding)}, first3={embedding[:3]}")  # CONFIRM 512
        
        if len(embedding) != 512:
            raise ValueError(f"Expected 512, got {len(embedding)}")
        
        return embedding

    def get_image_embedding(self, image_path_or_url):
        image = self._load_image(image_path_or_url)
        return self._get_image_embedding(image)

    def get_image_embedding_from_pil(self, image):
        return self._get_image_embedding(image.convert("RGB"))

    def store_embedding(self, image_id, embedding, metadata=None):
        metadata = metadata or {}
        # Qdrant point IDs must be uint64 or UUID.
        point_id = image_id
        if isinstance(image_id, str):
            try:
                uuid.UUID(image_id)
            except ValueError:
                if not image_id.isdigit():
                    # Deterministic UUID so the same image name maps to same point.
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, image_id))

        self.client.upsert(
            collection_name=self.collection_name,
            points=[{"id": point_id, "vector": embedding, "payload": metadata}]
        )
        print(f"[OK] Stored: {image_id} -> {point_id}")

    def search_similar(self, query_embedding, limit=5):
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit
        )
        return response.points

def index_folder_images(store, image_dir="images"):
    if not os.path.exists(image_dir):
        raise ValueError(f"Directory '{image_dir}' does not exist.")

    processed_count = 0
    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            img_path = os.path.join(image_dir, filename)
            try:
                embedding = store.get_image_embedding(img_path)
                store.store_embedding(
                    image_id=filename,
                    embedding=embedding,
                    metadata={"file_name": filename, "path": img_path}
                )
                processed_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to process {filename}: {e}")
    return processed_count


IMAGES_DIR = os.getenv("IMAGES_DIR", "images")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "image_embeddings_512")

app = FastAPI(title="Qdrant Image Presence API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_store_instance = None
_store_lock = threading.Lock()


def get_store() -> CLIPEmbeddingStore:
    global _store_instance
    if _store_instance is not None:
        return _store_instance
    with _store_lock:
        if _store_instance is not None:
            return _store_instance

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_url or not qdrant_api_key:
            raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY in your .env file.")

        _store_instance = CLIPEmbeddingStore(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_name=COLLECTION_NAME,
        )
        return _store_instance


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/collection/count")
def collection_count():
    store = get_store()
    count = store.client.count(collection_name=store.collection_name, exact=True).count
    return {"collection": store.collection_name, "count": count}


@app.get("/api/image")
def get_indexed_image(file_name: str = Query(..., min_length=1)):
    # Serve local indexed images so the React UI can show thumbnails.
    safe_name = os.path.basename(file_name)
    file_path = os.path.join(IMAGES_DIR, safe_name)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Image not found on disk.")
    return FileResponse(file_path)


@app.post("/api/index")
def index_images(image_dir: str = Query(default=IMAGES_DIR)):
    store = get_store()
    processed = index_folder_images(store, image_dir=image_dir)
    return {"indexed": processed, "collection": store.collection_name}


@app.post("/api/presence")
async def presence_check(
    image: UploadFile = File(...),
    threshold: float = Query(default=0.85, ge=0.0, le=1.0),
    top_k: int = Query(default=5, ge=1, le=20),
):
    store = get_store()

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        query_image = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse image: {e}")

    query_embedding = store.get_image_embedding_from_pil(query_image)
    results = store.search_similar(query_embedding, limit=top_k)

    if not results:
        return {
            "present": False,
            "best_score": None,
            "threshold": threshold,
            "results": [],
        }

    best_match = results[0]
    best_score = float(best_match.score)
    is_present = best_score >= threshold

    formatted = []
    for r in results:
        payload = r.payload or {}
        formatted.append(
            {
                "id": str(r.id),
                "score": float(r.score),
                "file_name": payload.get("file_name"),
            }
        )

    return {
        "present": is_present,
        "best_score": best_score,
        "threshold": threshold,
        "results": formatted,
    }


if __name__ == "__main__":
    parser = None
    import argparse

    parser = argparse.ArgumentParser(description="Qdrant image embedding service")
    parser.add_argument("--index", action="store_true", help="Index images from IMAGES_DIR and exit")
    parser.add_argument("--image-dir", default=IMAGES_DIR, help="Folder containing images to index")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.index:
        store = get_store()
        processed = index_folder_images(store, image_dir=args.image_dir)
        print(f"[OK] Indexed {processed} image(s) into `{store.collection_name}`.")
    else:
        uvicorn.run(app, host=args.host, port=args.port)

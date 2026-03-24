import os
import uuid
from dotenv import load_dotenv
import torch
from PIL import Image
import requests
from io import BytesIO
import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

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
                st.warning(f"Failed to process {filename}: {e}")
    return processed_count


def run_streamlit_app():
    st.set_page_config(page_title="Qdrant Image Presence Check", layout="wide")
    st.title("Qdrant Image Presence Check")
    st.write("Upload a query image and check if a similar image already exists in Qdrant.")

    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    if not QDRANT_URL or not QDRANT_API_KEY:
        st.error("Set QDRANT_URL and QDRANT_API_KEY in your .env file.")
        st.stop()

    store = CLIPEmbeddingStore(QDRANT_URL, QDRANT_API_KEY)

    st.subheader("Query image and check presence")
    count = store.client.count(collection_name=store.collection_name, exact=True).count
    st.caption(f"Collection `{store.collection_name}` currently has {count} vector(s).")
    threshold = st.slider(
        "Presence threshold (cosine similarity)",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.01
    )
    top_k = st.slider("Top K matches", min_value=1, max_value=10, value=5, step=1)
    uploaded = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded is not None:
        query_image = Image.open(uploaded).convert("RGB")
        st.image(query_image, caption="Query Image", use_container_width=True)

        query_embedding = store.get_image_embedding_from_pil(query_image)
        results = store.search_similar(query_embedding, limit=top_k)

        if not results:
            st.warning("No results found in the collection.")
            return

        best_match = results[0]
        best_score = float(best_match.score)
        is_present = best_score >= threshold

        if is_present:
            st.success(f"Image is present (best similarity: {best_score:.3f} >= {threshold:.2f})")
        else:
            st.error(f"Image is NOT present (best similarity: {best_score:.3f} < {threshold:.2f})")

        st.markdown("### Top Matches")
        for r in results:
            payload = r.payload or {}
            file_name = payload.get("file_name", "unknown")
            path = payload.get("path")
            st.write(f"- id: `{r.id}`, score: `{float(r.score):.3f}`, file: `{file_name}`")
            if path and os.path.exists(path):
                st.image(path, caption=f"Match: {file_name} ({float(r.score):.3f})", width=220)


if __name__ == "__main__":
    run_streamlit_app()

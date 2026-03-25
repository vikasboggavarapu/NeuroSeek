# CLIP Image Embedding Store with Qdrant Cloud

This application uses OpenAI's CLIP model to generate embeddings for images and stores them in Qdrant Cloud for similarity search.

## Setup

1. Install dependencies:
   ``bash
   pip install -r requirements.txt
   ``

2. Set your Qdrant Cloud credentials as environment variables:
   ```bash
   export QDRANT_URL="your-qdrant-cloud-url"
   export QDRANT_API_KEY="your-api-key"
   ```

## Usage
### Run the FastAPI backend

1. Start the API server:
   ```bash
   python app.py
   ```
   It serves the endpoints under `http://localhost:8000/api/*`.

2. (Optional) Index images from the local `images/` folder:
   ```bash
   python app.py --index
   ```

### Run the React frontend

1. Install frontend deps (once):
   ```bash
   cd frontend
   npm install
   ```
2. Start the dev server:
   ```bash
   npm run dev
   ```
3. Open `http://127.0.0.1:5173/` in your browser.

### Basic Example

```python
from app import CLIPEmbeddingStore

# Initialize with your credentials
store = CLIPEmbeddingStore(
    qdrant_url="your-url",
    qdrant_api_key="your-key"
)

# Generate and store embedding for an image
embedding = store.get_image_embedding("path/to/image.jpg")
store.store_embedding(
    image_id="unique_id",
    embedding=embedding,
    metadata={"description": "My image"}
)
```

### Search Similar Images

```python
# Get embedding for query image
query_embedding = store.get_image_embedding("query_image.jpg")

# Search for similar images
results = store.search_similar(query_embedding, limit=5)
for result in results:
    print(f"ID: {result.id}, Score: {result.score}")
```

## Features

- Generate CLIP embeddings for images (local files or URLs)
- Store embeddings in Qdrant Cloud
- Search for similar images using cosine similarity
- Metadata support for additional information

## Requirements

- Python 3.7+
- Qdrant Cloud account
- Internet connection for downloading CLIP model

## Troubleshooting

- Ensure your Qdrant URL and API key are correct
- Check that images are valid and accessible
- For large images, consider resizing for faster processing
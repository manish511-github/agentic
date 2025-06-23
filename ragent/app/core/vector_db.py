import os
import structlog
from qdrant_client import QdrantClient

logger = structlog.get_logger()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

_qdrant_client = None

def get_collection_name(agent_name: str, prefix: str) -> str:
    """Generate collection name for a specific agent and prefix."""
    clean_name = agent_name.replace(' ', '_').replace('-', '_').lower()
    clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
    return f"{prefix}_{clean_name}"

def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client instance."""
    global _qdrant_client
    if _qdrant_client is None:
        try:
            _qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY
            )
            logger.info(f"Initialized Qdrant client with URL: {QDRANT_URL}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    return _qdrant_client

def ensure_collection_exists(agent_name: str, prefix: str):
    """Ensure the collection exists in Qdrant for the specific agent and prefix."""
    try:
        client = get_qdrant_client()
        from qdrant_client.http import models
        collection_name = get_collection_name(agent_name, prefix)
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
            logger.info(f"Created Qdrant collection: {collection_name}")
        else:
            logger.info(f"Qdrant collection '{collection_name}' already exists")
    except Exception as e:
        logger.error(f"Failed to ensure collection exists for agent '{agent_name}' and prefix '{prefix}': {e}")
        raise 
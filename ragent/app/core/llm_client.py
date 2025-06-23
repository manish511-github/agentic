import os
import structlog
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

logger = structlog.get_logger()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

_embedding_model = None
_llm = None

def get_embedding_model():
    """Get or create embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        try:
            _embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
            logger.info("Initialized Google Generative AI embedding model")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    return _embedding_model

def get_llm():
    """Get or create LLM instance."""
    global _llm
    if _llm is None:
        try:
            _llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.7,
                max_output_tokens=2048
            )
            logger.info("Initialized Google Generative AI LLM")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    return _llm 
from celery import Celery
import os

# Get Redis URL from environment variable
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "ragent",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.tasks"]
)

# Optional configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# Optional: Configure periodic tasks
celery_app.conf.beat_schedule = {
    "run-agent-tasks": {
        "task": "app.tasks.run_agent_tasks",
        "schedule": 300.0,  # Run every 5 minutes
    },
} 
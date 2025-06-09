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
    broker_connection_retry_on_startup=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour
    worker_prefetch_multiplier=1,  # Disable prefetching
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks
    worker_max_memory_per_child=200000  # Restart worker after using 200MB memory
)

# Optional: Configure periodic tasks
celery_app.conf.beat_schedule = {
    "run-agent-tasks": {
        "task": "app.tasks.run_agent_tasks",
        "schedule": 300.0,  # Run every 5 minutes
    },
} 
from logging.config import fileConfig
import os
import sys
from pathlib import Path
from sqlalchemy import engine_from_config, pool
from sqlalchemy.ext.asyncio import AsyncEngine
from alembic import context
from sqlalchemy import text

# Add the parent directory to sys.path to ensure 'app' module is importable
sys.path.append(str(Path(__file__).parent.parent))
from app.models import Base

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
fileConfig(config.config_file_name)

# Add your model's MetaData object here
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py
connectable = AsyncEngine(
    engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True
    )
)

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

async def run_migrations_online():
    """Run migrations in 'online' mode."""
    async with connectable.connect() as connection:
        await connection.execute(text("SELECT 1"))  # Test connection
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            connection.run_sync(context.run_migrations)

if context.is_offline_mode():
    run_migrations_offline()
else:
    import asyncio
    asyncio.run(run_migrations_online())
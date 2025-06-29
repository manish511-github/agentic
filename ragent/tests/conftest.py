from datetime import datetime
import sys
import os
from typing import Generator

import pytest
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
from starlette.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.models import Base, UserModel
from app.database import get_sync_db
from app.auth.security import hash_password

USER_NAME = "Test"
USER_EMAIL = "test@test.com"
USER_PASSWORD = "123#DTest"

# Test database configuration
TEST_DB_NAME = "test_agent"
TEST_SYNC_POSTGRES_DSN = f"postgresql+psycopg2://test:test@postgres:5432/{TEST_DB_NAME}"

def create_test_database():
    """Create test database if it doesn't exist"""
    # Connect to default postgres database to create test database
    default_engine = create_engine("postgresql+psycopg2://test:test@postgres:5432/postgres", isolation_level="AUTOCOMMIT")
    
    with default_engine.connect() as conn:
        # Check if test database exists
        result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{TEST_DB_NAME}'"))
        if not result.fetchone():
            # Create test database
            conn.execute(text(f"CREATE DATABASE {TEST_DB_NAME}"))
    
    default_engine.dispose()

# Create test database
create_test_database()

# Create test engine
test_engine = create_engine(TEST_SYNC_POSTGRES_DSN, echo=False, pool_pre_ping=True)
SessionTesting = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture(scope="function")
def test_session() -> Generator:
    session = SessionTesting()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture(scope="function")
def app_test():
    # Create all tables for testing
    Base.metadata.create_all(bind=test_engine)
    yield app
    # Clean up after tests
    Base.metadata.drop_all(bind=test_engine)

@pytest.fixture(scope="function")
def client(app_test, test_session):
    def _test_db():
        try:
            yield test_session
        finally:
            pass

    app_test.dependency_overrides[get_sync_db] = _test_db
    return TestClient(app_test)

@pytest.fixture(scope="function")
def user(test_session):
    model = UserModel()
    model.username = USER_NAME
    model.email = USER_EMAIL
    model.hashed_password = hash_password(USER_PASSWORD)
    model.updated_at = datetime.utcnow()
    model.verified_at = datetime.utcnow()
    model.is_active = True
    test_session.add(model)
    test_session.commit()
    test_session.refresh(model)
    return model

@pytest.fixture(scope="function")
def inactive_user(test_session):
    model = UserModel()
    model.username = USER_NAME
    model.email = USER_EMAIL
    model.hashed_password = hash_password(USER_PASSWORD)
    model.updated_at = datetime.utcnow()
    model.is_active = False
    test_session.add(model)
    test_session.commit()
    test_session.refresh(model)
    return model

@pytest.fixture(scope="function")
def unverified_user(test_session):
    model = UserModel()
    model.username = USER_NAME
    model.email = USER_EMAIL
    model.hashed_password = hash_password(USER_PASSWORD)
    model.updated_at = datetime.utcnow()
    model.is_active = True
    model.verified_at = None
    test_session.add(model)
    test_session.commit()
    test_session.refresh(model)
    return model
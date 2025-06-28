from datetime import datetime
import sys
import os
from typing import Generator

import pytest
from sqlalchemy.orm import sessionmaker
from starlette.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.models import Base, UserModel
from app.database import sync_engine, get_sync_db
from app.api.users.security import hash_password

USER_NAME = "Test"
USER_EMAIL = "test@test.com"
USER_PASSWORD = "123#DTest"

SessionTesting = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)

@pytest.fixture(scope="function")
def test_session() -> Generator:
    session = SessionTesting()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture(scope="function")
def app_test():
    Base.metadata.create_all(bind=sync_engine)
    yield app
    Base.metadata.drop_all(bind=sync_engine)

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
import logging
from tests.conftest import USER_NAME, USER_EMAIL, USER_PASSWORD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_create_user(client):
    logger.info("Starting test_create_user")
    data = {
        "name": USER_NAME,
        "email": USER_EMAIL,
        "password": USER_PASSWORD
    }
    response = client.post('/users', json=data)
    logger.info(f"Response status: {response.status_code}, body: {response.json()}")
    assert response.status_code == 201
    resp_json = response.json()
    assert "password" not in resp_json
    assert resp_json["name"] == USER_NAME
    assert resp_json["email"] == USER_EMAIL
    assert resp_json["is_active"] is not None
    

def test_create_user_with_existing_email(client, inactive_user):
    logger.info("Starting test_create_user_with_existing_email")
    data = {
        "name": "Keshari Nandan",
        "email": inactive_user.email,
        "password": USER_PASSWORD
    }
    response = client.post("/users", json=data)
    logger.info(f"Response status: {response.status_code}, body: {response.json()}")
    assert response.status_code != 201


def test_create_user_with_invalid_email(client):
    logger.info("Starting test_create_user_with_invalid_email")
    data = {
        "name": "Keshari Nandan",
        "email": "keshari.com",
        "password": USER_PASSWORD
    }
    response = client.post("/users", json=data)
    logger.info(f"Response status: {response.status_code}, body: {response.json()}")
    assert response.status_code != 201


def test_create_user_with_empty_password(client):
    logger.info("Starting test_create_user_with_empty_password")
    data = {
        "name": "Keshari Nandan",
        "email": USER_EMAIL,
        "password": ""
    }
    response = client.post("/users", json=data)
    logger.info(f"Response status: {response.status_code}, body: {response.json()}")
    assert response.status_code != 201


def test_create_user_with_numeric_password(client):
    logger.info("Starting test_create_user_with_numeric_password")
    data = {
        "name": "Keshari Nandan",
        "email": USER_EMAIL,
        "password": "1232382318763"
    }
    response = client.post("/users", json=data)
    logger.info(f"Response status: {response.status_code}, body: {response.json()}")
    assert response.status_code != 201


def test_create_user_with_char_password(client):
    logger.info("Starting test_create_user_with_char_password")
    data = {
        "name": "Keshari Nandan",
        "email": USER_EMAIL,
        "password": "asjhgahAdF"
    }
    response = client.post("/users", json=data)
    logger.info(f"Response status: {response.status_code}, body: {response.json()}")
    assert response.status_code != 201


def test_create_user_with_alphanumeric_password(client):
    logger.info("Starting test_create_user_with_alphanumeric_password")
    data = {
        "name": "Keshari Nandan",
        "email": USER_EMAIL,
        "password": "sjdgajhGG27862"
    }
    response = client.post("/users", json=data)
    logger.info(f"Response status: {response.status_code}, body: {response.json()}")
    assert response.status_code != 201
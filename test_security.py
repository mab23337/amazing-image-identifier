import io
import pytest
from production_app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_no_file(client):
    response = client.post('/upload')
    assert response.status_code == 400

def test_invalid_extension(client):
    data = {
        'file': (io.BytesIO(b"notanimage"), "test.txt")
    }
    response = client.post('/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 400

def test_invalid_image_content(client):
    # JPG extension but not actually an image
    data = {
        "file": (io.BytesIO(b"notanimage"), "fake.jpg")
    }
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    
def test_invalid_magic_bytes(client):
    fake_jpg = b"NOTREALIMAGECONTENT"
    data = {
        "file": (io.BytesIO(fake_jpg), "fake.jpg")
    }
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"Invalid image header" in response.data

def test_file_too_large(client):
    large_content = b"\xff\xd8\xff" + b"\x00" * (11 * 1024 * 1024)
    data = {
        "file": (io.BytesIO(large_content), "big.jpg")
    }
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 413

import os

def test_file_deleted_after_upload(client):
    png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    data = {
        "file": (io.BytesIO(png_header), "test.png")
    }

    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code in [200, 400]

    upload_path = os.path.join("uploads", "test.png")
    assert not os.path.exists(upload_path)

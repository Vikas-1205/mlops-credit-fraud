import pytest
from unittest.mock import MagicMock
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
def mock_model():
    """Mock ML model that returns 0 (not fraud)."""
    model = MagicMock()
    model.predict.return_value = [0]
    return model


@pytest.fixture
def mock_fraud_model():
    """Mock ML model that returns 1 (fraud)."""
    model = MagicMock()
    model.predict.return_value = [1]
    return model


@pytest.fixture
async def client():
    """Async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

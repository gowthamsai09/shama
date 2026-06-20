import pytest
import pytest_asyncio
from shama.client import ShamaClient

@pytest_asyncio.fixture(scope="session")
async def client():
    """Initializes a single ShamaClient instance for all integration tests,
    and handles safe cleanup after tests finish.
    """
    client_instance = ShamaClient()
    await client_instance.initialize()  
    
    yield client_instance
    
    if hasattr(client_instance, "close"):
        await client_instance.close()
    elif hasattr(client_instance, "disconnect"):
        await client_instance.disconnect()


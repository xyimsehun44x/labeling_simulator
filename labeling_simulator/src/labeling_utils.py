import asyncio
import aiohttp
import json
import logging
import time
import os
import ssl
import certifi
from aiolimiter import AsyncLimiter
import google.auth
from google.auth.transport.requests import Request as AuthRequest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Vertex AI Configuration - DYNAMIC FROM ENV
PROJECT_ID = os.environ.get("PROJECT_ID", "slcc-buzz-ai") # Default fallback
LOCATION = os.environ.get("LOCATION", "us-central1")
MODEL_ID = os.environ.get("MODEL_ID", "gemini-3-pro-preview") # Using stable model ID as default

# Vertex AI Endpoint (OAuth)
VERTEX_ENDPOINT = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:generateContent"
# AI Studio Endpoint (API Key)
AI_STUDIO_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent"

class AsyncProcessor:
    def __init__(self, rpm_limit: int = 30000, max_concurrent: int = 500, api_key: str = None):
        """
        Args:
            rpm_limit: Requests per minute limit (Quota).
            max_concurrent: Maximum concurrent in-flight requests.
            api_key: Optional Google AI Studio API Key. If provided, uses AI Studio instead of Vertex AI.
        """
        self.rpm_limit = rpm_limit
        # aiolimiter expects per-second or arbitrary interval. 
        # 30k RPM = 500 RPS. We'll set it slightly lower to be safe.
        safe_rps = (rpm_limit / 60) * 0.95 
        self.limiter = AsyncLimiter(max_rate=safe_rps, time_period=1.0)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.token = None
        self.api_key = api_key
        self.headers = {}
        
        # Determine endpoint
        if self.api_key:
            self.endpoint = f"{AI_STUDIO_ENDPOINT}?key={self.api_key}"
            logger.info("🔧 Using AI Studio API Endpoint (API Key configured)")
        else:
            self.endpoint = VERTEX_ENDPOINT
            logger.info(f"🔧 Using Vertex AI Endpoint: {self.endpoint}")
        
    def authenticate(self):
        """Refreshes credentials if using Vertex AI. No-op for API Key."""
        if self.api_key:
            # API Key auth is handled in the URL query param, just set Content-Type
            self.headers = {
                "Content-Type": "application/json"
            }
            return

        logger.info("Authenticating with Google Cloud...")
        credentials, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(AuthRequest())
        self.token = credentials.token
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        logger.info(f"Authentication successful. Token acquired.")

    async def call_vertex_api(self, session: aiohttp.ClientSession, row: dict) -> dict:
        """
        Calls Vertex AI API for a single row.
        Returns the row with 'response' or 'error' appended.
        """
        request_body = row.get("request")
        row_key = row.get("key")

        if not request_body:
            return {"key": row_key, "error": "Missing request body"}

        max_retries = 5
        base_delay = 1.0

        ssl_context = ssl.create_default_context(cafile=certifi.where())
        for attempt in range(max_retries + 1):
            async with self.semaphore:
                async with self.limiter:
                    try:
                        async with session.post(
                            self.endpoint,
                            json=request_body,
                            headers=self.headers,
                            ssl=ssl_context,
                        ) as resp:
                            if resp.status == 200:
                                result = await resp.json()
                                return {
                                    "key": row_key,
                                    "response": result,
                                    "status": "success"
                                }
                            elif resp.status == 429 or resp.status == 503:
                                # Quota exceeded (429) or Service Unavailable (503)
                                if attempt < max_retries:
                                    delay = base_delay * (2 ** attempt) + (0.1 * (attempt + 1)) # Add jitter
                                    logger.warning(f"Status {resp.status} for {row_key} (Attempt {attempt+1}/{max_retries}). Sleeping {delay:.2f}s...")
                                    await asyncio.sleep(delay)
                                    continue # Retry loop
                                else:
                                    # Out of retries
                                    raise aiohttp.ClientResponseError(resp.request_info, resp.history, status=resp.status, message=f"Max retries exceeded for {resp.status}")
                            else:
                                text = await resp.text()
                                logger.error(f"API Error {resp.status} for {row_key}: {text}")
                                return {
                                    "key": row_key,
                                    "error": f"HTTP {resp.status}: {text}",
                                    "status": "failed"
                                }
                    except Exception as e:
                        if attempt < max_retries and ('429' in str(e) or '503' in str(e) or isinstance(e, aiohttp.ClientConnectorError)):
                             # Catch network blips, re-raised 429s/503s, or connection errors
                             delay = base_delay * (2 ** attempt)
                             logger.warning(f"Request failed for {row_key}: {e}. Retrying in {delay}s...")
                             await asyncio.sleep(delay)
                             continue
                        
                        logger.error(f"Request failed for {row_key}: {e}")
                        return {
                            "key": row_key,
                            "error": str(e),
                            "status": "failed"
                        }

    async def process_batch(self, input_data: list) -> list:
        """
        Process a list of input rows concurrently.
        """
        self.authenticate() # Ensure fresh token
        
        timeout = aiohttp.ClientTimeout(total=60) # 60s timeout per request
        connector = aiohttp.TCPConnector(limit=0) # No limit on connector, controlled by semaphore
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            for row in input_data:
                task = asyncio.create_task(self.call_vertex_api(session, row))
                tasks.append(task)
            
            # Use tqdm if available for progress tracking, or just gather
            results = []
            try:
                from tqdm.asyncio import tqdm
                for f in tqdm.as_completed(tasks, total=len(tasks), desc="Processing Batch"):
                    results.append(await f)
            except ImportError:
                 results = await asyncio.gather(*tasks)

            return results

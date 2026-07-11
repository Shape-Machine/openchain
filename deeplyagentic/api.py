"""Shared HTTP client configuration for the RiffDesk API."""

import os

import httpx

API_BASE_URL = os.environ.get("RIFFDESK_API_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("RIFFDESK_API_KEY", "demo-secret-key")


def api_client() -> httpx.Client:
    """Return a configured client for the RiffDesk API."""
    return httpx.Client(
        base_url=API_BASE_URL,
        headers={"X-API-Key": API_KEY},
        timeout=10.0,
    )

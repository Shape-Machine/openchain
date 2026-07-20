"""Shared RiffDesk tools and agent construction.

UI adapters decide how to persist and resume the graph: the terminal app uses
an in-memory checkpointer, while LangGraph Agent Server supplies persistence
for the Studio variant.
"""

import os
from typing import Any

import httpx
from deepagents import create_deep_agent
from langchain.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import interrupt
from pydantic import BaseModel, EmailStr, ValidationError

API_BASE_URL = os.environ.get("RIFFDESK_API_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("RIFFDESK_API_KEY", "demo-secret-key")


class IdentityInput(BaseModel):
    """Authoritative schema for identity data supplied after an interrupt."""

    customer_id: int
    email: EmailStr


def _parse_identity_response(response: object) -> IdentityInput:
    """Accept native CLI data and JSON text submitted by Studio."""
    if isinstance(response, str):
        return IdentityInput.model_validate_json(response)
    return IdentityInput.model_validate(response)


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=API_BASE_URL,
        headers={"X-API-Key": API_KEY},
        timeout=10.0,
    )


@tool
def collect_customer_identity() -> str:
    """Call this FIRST, before any other tool. Pauses and asks the customer
    for their customer ID and email, then verifies identity. Returns the
    verified customer_id to use in later tool calls."""
    error = None

    while True:
        payload = {
            "type": "identity_request",
            "message": "Please provide your customer ID and email to verify your identity.",
        }
        if error:
            payload["error"] = error

        response = interrupt(payload)

        try:
            identity = _parse_identity_response(response)
            break
        except ValidationError as exc:
            error = "; ".join(
                f"{e['loc'][0] if e['loc'] else 'input'}: {e['msg']}"
                for e in exc.errors()
            )

    with _client() as client:
        resp = client.get(f"/customers/{identity.customer_id}")
    if resp.status_code == 404:
        return f"No customer found with ID {identity.customer_id}. Ask the customer to try again."
    resp.raise_for_status()
    customer = resp.json()

    if customer["email"].strip().lower() != identity.email.strip().lower():
        return "Identity verification FAILED: email does not match our records. Do not proceed."

    return (
        f"Identity verified for {customer['first_name']} {customer['last_name']} "
        f"(customer_id={identity.customer_id}). Use this customer_id for all further lookups."
    )


@tool
def query_invoices(customer_id: int, limit: int = 10) -> str:
    """Look up a verified customer's recent invoices (id, date, total)."""
    with _client() as client:
        resp = client.get(f"/customers/{customer_id}/invoices", params={"limit": limit})
    if resp.status_code == 404:
        return f"No customer found with ID {customer_id}."
    resp.raise_for_status()
    return str(resp.json())


@tool
def get_invoice_detail(customer_id: int, invoice_id: int) -> str:
    """Get line-item detail (tracks, price, quantity) for a specific invoice.
    The invoice must belong to the verified customer. Use this to confirm
    exactly what's being refunded before filing a refund."""
    with _client() as client:
        resp = client.get(f"/customers/{customer_id}/invoices/{invoice_id}")
    if resp.status_code == 404:
        return f"Invoice {invoice_id} was not found for customer {customer_id}."
    resp.raise_for_status()
    return str(resp.json())


@tool
def request_refund(customer_id: int, invoice_id: int, reason: str) -> str:
    """File a refund request for a given invoice. Requires human approval
    before it takes effect. The invoice must belong to the verified customer."""
    with _client() as client:
        resp = client.post(
            "/refunds",
            json={
                "customer_id": customer_id,
                "invoice_id": invoice_id,
                "reason": reason,
            },
        )
    if resp.status_code == 404:
        return (
            f"Invoice {invoice_id} was not found for customer {customer_id}; "
            "refund not filed."
        )
    resp.raise_for_status()
    return str(resp.json())


def build_agent(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    *,
    refund_tool=request_refund,
    require_refund_approval: bool = True,
):
    """Build the RiffDesk graph for a UI or Agent Server adapter."""
    interrupt_on = {
        "query_invoices": False,
        "get_invoice_detail": False,
        refund_tool.name: (
            {"allowed_decisions": ["approve", "reject"]}
            if require_refund_approval
            else False
        ),
    }
    return create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=[
            collect_customer_identity,
            query_invoices,
            get_invoice_detail,
            refund_tool,
        ],
        system_prompt=(
            "You are a music store support agent for RiffDesk. "
            "Always call collect_customer_identity FIRST, before anything else, "
            "and only proceed to query_invoices, get_invoice_detail, or "
            "request_refund once identity has been verified successfully. "
            "Always pass the verified customer_id to every invoice or refund tool. "
            "Confirm the exact invoice with the customer before filing a refund."
        ),
        interrupt_on=interrupt_on,
        checkpointer=checkpointer,
    )

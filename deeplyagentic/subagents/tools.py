"""API-backed tools shared by the RiffDesk specialist sub-agents."""

from langchain.tools import tool

from deeplyagentic.api import api_client


@tool
def query_invoices(customer_id: int, limit: int = 10) -> str:
    """Look up a verified customer's recent invoices (id, date, total)."""
    with api_client() as client:
        response = client.get(
            f"/customers/{customer_id}/invoices", params={"limit": limit}
        )
    if response.status_code == 404:
        return f"No customer found with ID {customer_id}."
    response.raise_for_status()
    return str(response.json())


@tool
def get_invoice_detail(customer_id: int, invoice_id: int) -> str:
    """Return track-level detail for an invoice owned by the customer."""
    with api_client() as client:
        response = client.get(
            f"/invoices/{invoice_id}", params={"customer_id": customer_id}
        )
    if response.status_code == 404:
        return f"Invoice {invoice_id} does not belong to customer {customer_id}."
    response.raise_for_status()
    return str(response.json())


@tool
def request_refund(customer_id: int, invoice_id: int, reason: str) -> str:
    """File an approval-gated refund request for an owned invoice."""
    with api_client() as client:
        response = client.post(
            "/refunds",
            json={
                "customer_id": customer_id,
                "invoice_id": invoice_id,
                "reason": reason,
            },
        )
    if response.status_code == 404:
        return (
            f"Invoice {invoice_id} does not belong to customer {customer_id}; "
            "refund not filed."
        )
    response.raise_for_status()
    return str(response.json())

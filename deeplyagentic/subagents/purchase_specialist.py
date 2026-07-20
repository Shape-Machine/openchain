"""Read-only purchase-history specialist definition."""

from deeplyagentic.subagents.tools import get_invoice_detail, query_invoices

purchase_specialist = {
    "name": "purchase-specialist",
    "description": (
        "Read-only specialist for listing a verified customer's invoices and "
        "explaining the tracks, prices, and quantities on a specific invoice."
    ),
    "system_prompt": (
        "You are RiffDesk's purchase-history specialist. Use only the verified "
        "customer_id supplied by the supervisor. Handle recent-purchase and "
        "invoice-detail questions with the available read-only tools. Never "
        "perform or promise a refund. Return a concise factual answer to the "
        "supervisor."
    ),
    "tools": [query_invoices, get_invoice_detail],
}

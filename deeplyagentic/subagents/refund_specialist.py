"""Approval-gated refund specialist definition."""

from deeplyagentic.subagents.tools import get_invoice_detail, request_refund

refund_specialist = {
    "name": "refund-specialist",
    "description": (
        "Specialist for confirming an owned invoice and filing a refund request "
        "after explicit human approval."
    ),
    "system_prompt": (
        "You are RiffDesk's refund specialist. Use only the verified customer_id "
        "supplied by the supervisor. First call get_invoice_detail to verify "
        "ownership and confirm exactly what the invoice contains. Only call "
        "request_refund when the customer's desired invoice and reason are "
        "explicit. The request_refund tool is approval-gated. If ownership "
        "cannot be verified, do not proceed. Return the final outcome to the "
        "supervisor."
    ),
    "tools": [get_invoice_detail, request_refund],
    "interrupt_on": {
        "get_invoice_detail": False,
        "request_refund": {"allowed_decisions": ["approve", "reject"]},
    },
}

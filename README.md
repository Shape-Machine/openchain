# OpenChain / RiffDesk

A human-in-the-loop customer support demo for the Chinook music store data.

The support supervisor verifies the customer's identity, then delegates to:

- a read-only purchase specialist for invoice history and line-item details;
- an approval-gated refund specialist for refund requests.

Run the API first:

    uv run python -m riffdesk.main

Then run the interactive support agent in another terminal:

    uv run python -m deeplyagentic.main

The API defaults to `http://127.0.0.1:8000` and the demo API key
`demo-secret-key`. Set `RIFFDESK_API_URL`, `RIFFDESK_API_KEY`, or `API_KEY` to
override these values.

customer id: 1
email: luisg@embraer.com.br

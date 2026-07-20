Run the RiffDesk API first:

```bash
uv run riffdesk/main.py
```

Run the terminal agent:

```bash
uv run python -m deeplyagentic.main
```

## LangSmith Studio

Run the local LangGraph Agent Server and open the LangSmith Studio URL it
prints:

```bash
uv run langgraph dev
```

This is the command used to run this project with LangSmith Studio. It starts
a local development server with hot reload; it does not create a hosted
LangSmith deployment.

To create a hosted deployment instead, authenticate and run:

```bash
uv run langgraph deploy
```

Hosted deployment may require selecting a LangSmith workspace and deployment
configuration interactively.

In Studio, start the run in Chat mode by entering a message. In Graph mode,
use an input with a non-empty `messages` array:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What did I buy recently?"
    }
  ]
}
```

The agent uses `ANTHROPIC_API_KEY` for Claude. Optional configuration:

```bash
export RIFFDESK_API_URL=http://127.0.0.1:8000
export RIFFDESK_API_KEY=demo-secret-key
export LANGSMITH_API_KEY=your-langsmith-key
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=openchain
```

When Studio pauses for identity verification, resume with:

```json
{
  "customer_id": 1,
  "email": "luisg@embraer.com.br"
}
```

When Studio pauses for refund approval, replace the resume editor's default
value with one of these payloads.

Approve:

```json
{
  "decisions": [
    {"type": "approve"}
  ]
}
```

Reject:

```json
{
  "decisions": [
    {"type": "reject"}
  ]
}
```

customer id: 1
email: luisg@embraer.com.br

"""
riffdesk - a minimal REST API exposing the Chinook music store database
to a customer support agent.

Assumes chinook.sql sits next to this file: riffdesk/chinook.sql

Run with:  uv run riffdesk/main.py

customer id: 1
email: luisg@embraer.com.br
"""

import os
import sqlite3
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

# --- paths & config -------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
SQL_FILE = BASE_DIR / "chinook.sql"
DB_FILE = BASE_DIR / "chinook.db"

API_KEY = os.environ.get("API_KEY", "demo-secret-key")


def init_db() -> None:
    """Materialize chinook.sql into a real SQLite file (once), and add a
    refund_requests table for writes so we never touch the original data."""
    if not DB_FILE.exists():
        with closing(sqlite3.connect(DB_FILE)) as conn:
            conn.executescript(SQL_FILE.read_text())

    with closing(sqlite3.connect(DB_FILE)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS refund_requests (
                refund_id INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_id INTEGER NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def get_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def require_api_key(x_api_key: str = Header(...)) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


# --- schemas ---------------------------------------------------------------

class Customer(BaseModel):
    customer_id: int
    first_name: str
    last_name: str
    email: str
    support_rep_id: Optional[int] = None


class Invoice(BaseModel):
    invoice_id: int
    invoice_date: str
    total: float


class InvoiceLine(BaseModel):
    track_name: str
    unit_price: float
    quantity: int


class InvoiceDetail(Invoice):
    lines: list[InvoiceLine]


class RefundRequestIn(BaseModel):
    invoice_id: int
    reason: str


class RefundRequestOut(BaseModel):
    refund_id: int
    invoice_id: int
    reason: str
    status: str
    created_at: str


# --- app ---------------------------------------------------------------

app = FastAPI(title="riffdesk", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get(
    "/customers/{customer_id}",
    response_model=Customer,
    dependencies=[Depends(require_api_key)],
)
def get_customer(customer_id: int, db: sqlite3.Connection = Depends(get_db)) -> Customer:
    row = db.execute(
        "SELECT CustomerId, FirstName, LastName, Email, SupportRepId "
        "FROM Customer WHERE CustomerId = ?",
        (customer_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return Customer(
        customer_id=row["CustomerId"],
        first_name=row["FirstName"],
        last_name=row["LastName"],
        email=row["Email"],
        support_rep_id=row["SupportRepId"],
    )


@app.get(
    "/customers/{customer_id}/invoices",
    response_model=list[Invoice],
    dependencies=[Depends(require_api_key)],
)
def list_invoices(
    customer_id: int,
    limit: int = Query(default=20, le=100),
    db: sqlite3.Connection = Depends(get_db),
) -> list[Invoice]:
    exists = db.execute(
        "SELECT 1 FROM Customer WHERE CustomerId = ?", (customer_id,)
    ).fetchone()
    if exists is None:
        raise HTTPException(status_code=404, detail="Customer not found")

    rows = db.execute(
        "SELECT InvoiceId, InvoiceDate, Total FROM Invoice "
        "WHERE CustomerId = ? ORDER BY InvoiceDate DESC LIMIT ?",
        (customer_id, limit),
    ).fetchall()
    return [
        Invoice(invoice_id=r["InvoiceId"], invoice_date=r["InvoiceDate"], total=r["Total"])
        for r in rows
    ]


@app.get(
    "/invoices/{invoice_id}",
    response_model=InvoiceDetail,
    dependencies=[Depends(require_api_key)],
)
def get_invoice(invoice_id: int, db: sqlite3.Connection = Depends(get_db)) -> InvoiceDetail:
    invoice = db.execute(
        "SELECT InvoiceId, InvoiceDate, Total FROM Invoice WHERE InvoiceId = ?",
        (invoice_id,),
    ).fetchone()
    if invoice is None:
        raise HTTPException(status_code=404, detail="Invoice not found")

    lines = db.execute(
        "SELECT t.Name AS TrackName, il.UnitPrice, il.Quantity "
        "FROM InvoiceLine il JOIN Track t ON t.TrackId = il.TrackId "
        "WHERE il.InvoiceId = ?",
        (invoice_id,),
    ).fetchall()

    return InvoiceDetail(
        invoice_id=invoice["InvoiceId"],
        invoice_date=invoice["InvoiceDate"],
        total=invoice["Total"],
        lines=[
            InvoiceLine(
                track_name=l["TrackName"], unit_price=l["UnitPrice"], quantity=l["Quantity"]
            )
            for l in lines
        ],
    )


@app.post(
    "/refunds",
    response_model=RefundRequestOut,
    status_code=201,
    dependencies=[Depends(require_api_key)],
)
def create_refund(
    body: RefundRequestIn, db: sqlite3.Connection = Depends(get_db)
) -> RefundRequestOut:
    invoice = db.execute(
        "SELECT 1 FROM Invoice WHERE InvoiceId = ?", (body.invoice_id,)
    ).fetchone()
    if invoice is None:
        raise HTTPException(status_code=404, detail="Invoice not found")

    created_at = datetime.now(timezone.utc).isoformat()
    cur = db.execute(
        "INSERT INTO refund_requests (invoice_id, reason, status, created_at) "
        "VALUES (?, ?, 'pending', ?)",
        (body.invoice_id, body.reason, created_at),
    )
    db.commit()

    return RefundRequestOut(
        refund_id=cur.lastrowid,
        invoice_id=body.invoice_id,
        reason=body.reason,
        status="pending",
        created_at=created_at,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

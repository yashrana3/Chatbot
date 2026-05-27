"""
RAG Engine — FAISS vector store with AirBnB support knowledge base.
Uses Ollama embeddings (Mistral) for semantic search.
"""

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# ---------------------------------------------------------------------------
# AirBnB Support Knowledge Base
# ---------------------------------------------------------------------------
AIRBNB_KNOWLEDGE = [
    # Refund & Cancellation
    "Guests can receive a full refund if they cancel within 48 hours of booking, provided the check-in date is at least 14 days away.",
    "Under the Flexible cancellation policy, guests get a full refund if they cancel at least 24 hours before check-in.",
    "Under the Moderate cancellation policy, guests get a full refund if they cancel at least 5 days before check-in.",
    "Under the Strict cancellation policy, guests get a 50% refund if they cancel at least 7 days before check-in. No refund after that.",
    "If a host cancels a reservation, the guest receives a full refund automatically. Hosts may face penalties including fees and calendar blocking.",
    "Service fees are non-refundable if the guest cancels after the free cancellation window has passed.",
    "Refunds are processed within 5-7 business days to the original payment method.",
    "Airbnb's extenuating circumstances policy allows penalty-free cancellation for emergencies like natural disasters, serious illness, or government travel restrictions.",

    # Check-in & Check-out
    "Standard check-in time is 3:00 PM and check-out time is 11:00 AM unless the host specifies otherwise in the listing.",
    "Self check-in options include smart locks, lockboxes, and building staff. Hosts should provide clear instructions at least 24 hours before arrival.",
    "Early check-in or late check-out must be arranged directly with the host and is not guaranteed. Some hosts may charge extra.",
    "If a guest cannot check in due to host issues (wrong code, no access), they should contact Airbnb support immediately for rebooking assistance.",

    # Payments & Pricing
    "Airbnb charges guests at the time of booking. For reservations over $500 or 28+ nights, a payment plan with 50% upfront may be available.",
    "The total price includes the nightly rate, cleaning fee, Airbnb service fee, and applicable local taxes.",
    "Airbnb service fee for guests is typically 14-16% of the booking subtotal.",
    "Hosts receive payouts 24 hours after the guest's scheduled check-in time via their chosen payout method.",
    "Accepted payment methods include credit/debit cards, PayPal, Apple Pay, and Google Pay depending on the region.",
    "Hosts can set custom pricing for specific dates, weekly discounts, and monthly discounts through their calendar settings.",

    # Host Guidelines
    "Hosts must maintain accurate listing descriptions and photos. Misleading listings may result in penalties or removal.",
    "Hosts are expected to respond to booking inquiries within 24 hours. Response rate affects search ranking.",
    "Superhost status requires: 4.8+ overall rating, <1% cancellation rate, 90%+ response rate, and 10+ completed stays per year.",
    "Hosts can set house rules including no smoking, no pets, no parties, quiet hours, and maximum guest limits.",
    "Hosts are responsible for ensuring the property meets local safety regulations including smoke detectors, fire extinguishers, and carbon monoxide detectors.",

    # Safety & Trust
    "Airbnb provides Host Protection Insurance covering up to $1 million in liability per incident.",
    "AirCover for Hosts includes up to $3 million in damage protection for property damage caused by guests.",
    "Guests and hosts can report safety concerns 24/7 through the Airbnb app's Safety Center.",
    "All users must verify their identity with government-issued ID and a selfie before their first booking.",
    "Airbnb has a strict no-party and no-event policy at all listings globally.",

    # Guest Support
    "If a listing doesn't match the description, guests can report the issue within 72 hours of check-in for a full or partial refund.",
    "For issues during a stay (broken amenities, cleanliness), guests should message the host first, then contact Airbnb if unresolved within 24 hours.",
    "Airbnb customer support is available 24/7 via phone, chat, and email in the Help Center.",
    "Guests can modify reservation dates or number of guests through the app, subject to host approval and price adjustments.",
    "Long-term stays (28+ nights) may have different cancellation terms. Guests should review the specific long-term cancellation policy.",

    # Reviews & Ratings
    "Both guests and hosts have 14 days after check-out to leave a review. Reviews are published once both parties submit or after the 14-day window.",
    "Reviews cannot be edited after submission. They can only be removed if they violate Airbnb's content policy.",
    "Hosts can publicly respond to guest reviews within 30 days of the review being posted.",

    # Experiences & Services
    "Airbnb Experiences are activities hosted by locals. They have separate cancellation and refund policies from stays.",
    "Guests can book Experiences up to 2 hours before start time, subject to availability.",
]

# ---------------------------------------------------------------------------
# Vector Store (lazy init)
# ---------------------------------------------------------------------------
_db = None


def _get_db():
    """Initialize the FAISS vector store lazily on first query."""
    global _db
    if _db is None:
        print("🔧 Building FAISS index with Ollama embeddings...")
        embeddings = OllamaEmbeddings(model="mistral", base_url="http://localhost:11434")
        _db = FAISS.from_texts(AIRBNB_KNOWLEDGE, embeddings)
        print("✅ FAISS index ready!")
    return _db


def get_context(query: str, k: int = 2) -> str:
    """
    Retrieve the top-k most relevant knowledge snippets for a user query.

    Args:
        query: The user's question.
        k: Number of documents to retrieve.

    Returns:
        A newline-separated string of relevant context.
    """
    db = _get_db()
    docs = db.similarity_search(query, k=k)
    return "\n".join(f"• {doc.page_content}" for doc in docs)

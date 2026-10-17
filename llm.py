"""Groq chat-model factory, model listing and friendly error messages."""
import os

import groq
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]
# Non-chat models returned by /models that can't do this job.
_NON_CHAT = ("whisper", "guard", "tts", "playai", "orpheus", "compound", "allam")


def api_key() -> str:
    load_dotenv(override=True)  # pick up a key pasted into .env without restarting the app
    key = (os.getenv("GROQ_API_KEY") or "").strip().strip("'\"")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Create a key at https://console.groq.com/keys and put it in "
            "the .env file next to app.py as:  GROQ_API_KEY=gsk_..."
        )
    return key


def _is_gpt_oss(model: str) -> bool:
    return model.startswith("openai/gpt-oss")


def make_llm(model: str = DEFAULT_MODEL, temperature: float = 0.0, max_tokens: int = 2000,
             reasoning: str = "low") -> ChatGroq:
    # gpt-oss models reason before answering; "low" keeps that cheap on the free tier's token/minute cap.
    extra = {"reasoning_effort": reasoning} if _is_gpt_oss(model) else {}
    return ChatGroq(
        model=model,
        api_key=api_key(),
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=6,  # the Groq SDK backs off and honours retry-after on 429s
        timeout=120,
        **extra,
    )


def structured(llm: ChatGroq, schema, model: str):
    """Structured output: strict JSON schema where Groq supports it, tool calling elsewhere."""
    if _is_gpt_oss(model):
        return llm.with_structured_output(schema, method="json_schema", strict=True)
    return llm.with_structured_output(schema, method="function_calling")


# Models that score jobs reliably with structured output (checked 2026-10-09).
SCORING_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b")


def scoring_pool(selected: str, available: list[str]) -> list[str]:
    """Groq's free-tier limits (8K tokens/min, 1K requests/day) are per model, so scoring with every model the
    key can use, side by side, multiplies throughput. The selected model always comes first."""
    return [selected] + [m for m in SCORING_MODELS if m != selected and m in available]


def with_fallbacks(build, model: str):
    """build(model) -> chain. If Groq rate-limits the call even after its retries (e.g. the model's daily cap is
    used up), the same call runs on the other scoring models instead of failing."""
    others = [build(m) for m in SCORING_MODELS if m != model]
    return build(model).with_fallbacks(others, exceptions_to_handle=(groq.RateLimitError,))


def list_models() -> list[str]:
    """Chat models this key can use, default first. Falls back to a known list if Groq can't be reached."""
    try:
        client = groq.Groq(api_key=api_key(), max_retries=1, timeout=15)
        ids = [
            m.id for m in client.models.list().data
            if getattr(m, "active", True) and not any(x in m.id for x in _NON_CHAT)
        ]
    except Exception:
        ids = []
    ids = ids or list(FALLBACK_MODELS)
    if DEFAULT_MODEL not in ids:
        ids.append(DEFAULT_MODEL)
    return sorted(set(ids), key=lambda m: (m != DEFAULT_MODEL, m))


def is_daily_limit(e: Exception) -> bool:
    """Groq's per-day caps (e.g. 200K tokens/day per model on the free tier) don't clear for hours."""
    return isinstance(e, groq.RateLimitError) and "per day" in str(e)


def is_fatal(e: Exception) -> bool:
    """Errors that will repeat on every call, so a batch should stop instead of retrying each job."""
    return isinstance(e, (groq.AuthenticationError, groq.PermissionDeniedError, groq.NotFoundError, RuntimeError))


def explain(e: Exception) -> str:
    if isinstance(e, RuntimeError):
        return str(e)
    if isinstance(e, groq.AuthenticationError):
        return (
            "Groq rejected the API key (401 Invalid API Key). Create a new key at "
            "https://console.groq.com/keys, paste it into .env as GROQ_API_KEY=gsk_..., then restart the app."
        )
    if isinstance(e, groq.PermissionDeniedError):
        return (
            "Groq returned 403 Forbidden: the key is recognised but this model or your organisation is "
            "blocked, or your network/VPN region is not allowed. Try another model or network."
        )
    if is_daily_limit(e):
        return ("This model's daily Groq limit is used up (free tier: about 200K tokens per model per day, roughly "
                "100 scored jobs). Pick another model in the sidebar; the limit refills over the next 24 hours.")
    if isinstance(e, groq.RateLimitError):
        return (
            "Groq rate limit reached (free tier on gpt-oss-120b is about 8K tokens/min and 1K requests/day). "
            "Wait a minute, lower 'AI-score the top N', or pick a smaller model."
        )
    if isinstance(e, groq.NotFoundError):
        return "That model isn't available to this key. Pick another model in the sidebar."
    if isinstance(e, groq.APIConnectionError):
        return "Could not reach Groq. Check your internet connection."
    return f"AI call failed ({type(e).__name__}): {str(e)[:300]}"

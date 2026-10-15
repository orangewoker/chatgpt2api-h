from __future__ import annotations

from typing import Any

from services.account_service import account_service
from services.model_service import model_catalog_service
from utils.helper import (
    AGNES_IMAGE_MODEL,
    CODEX_IMAGE_MODEL,
    GEMINI_IMAGE_MODEL,
    GEMINI_IMAGE_MODELS,
    WEB_IMAGE_MODELS,
)


def list_models() -> dict[str, Any]:
    result = model_catalog_service.list_models()
    data = result.get("data")
    if not isinstance(data, list):
        return result
    seen = {str(item.get("id") or "").strip() for item in data if isinstance(item, dict)}
    dynamic_models: set[str] = set()
    accounts = account_service.list_accounts()
    chatgpt_image_accounts = [
        account
        for account in accounts
        if isinstance(account, dict)
           and account_service.account_provider(account) == "chatgpt"
           and account.get("status") not in {"禁用", "异常"}
           and account_service._normalize_source_type(account.get("source_type")) != "codex"
    ]
    agnes_image_accounts = [
        account
        for account in accounts
        if isinstance(account, dict)
           and account_service.account_provider(account) == "agnes"
           and account_service._is_image_account_available(account)
    ]
    gemini_image_accounts = [
        account
        for account in accounts
        if isinstance(account, dict)
           and account_service.account_provider(account) == "gemini"
           and account_service._is_image_account_available(account)
    ]
    codex_types = {
        normalized
        for account in accounts
        if isinstance(account, dict)
           and account_service.account_provider(account) == "chatgpt"
           and account.get("status") not in {"禁用", "异常"}
           and account_service._normalize_source_type(account.get("source_type")) == "codex"
           and (normalized := account_service._normalize_account_type(account.get("type")))
    }

    if chatgpt_image_accounts:
        dynamic_models.update(WEB_IMAGE_MODELS)
    if agnes_image_accounts:
        dynamic_models.add(AGNES_IMAGE_MODEL)
    if gemini_image_accounts:
        dynamic_models.update(GEMINI_IMAGE_MODELS)
    if codex_types & {"Plus", "Team", "Pro"}:
        dynamic_models.add(CODEX_IMAGE_MODEL)
    if "Plus" in codex_types:
        dynamic_models.add(f"plus-{CODEX_IMAGE_MODEL}")
    if "Team" in codex_types:
        dynamic_models.add(f"team-{CODEX_IMAGE_MODEL}")
    if "Pro" in codex_types:
        dynamic_models.add(f"pro-{CODEX_IMAGE_MODEL}")

    for model in sorted(dynamic_models):
        if model not in seen:
            data.append({
                "id": model,
                "object": "model",
                "created": 0,
                "owned_by": "chatgpt2api",
                "permission": [],
                "root": model,
                "parent": None,
            })
    return result

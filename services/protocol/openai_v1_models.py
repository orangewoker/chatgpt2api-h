from __future__ import annotations

from typing import Any

from services.account_service import account_service
from services.model_service import model_catalog_service
from utils.helper import CODEX_IMAGE_MODEL, WEB_IMAGE_MODELS


def list_models() -> dict[str, Any]:
    result = model_catalog_service.list_models()
    data = result.get("data")
    if not isinstance(data, list):
        return result
    seen = {str(item.get("id") or "").strip() for item in data if isinstance(item, dict)}
    dynamic_models: set[str] = set()
    accounts = account_service.list_accounts()
    active_accounts = [
        account
        for account in accounts
        if isinstance(account, dict) and account.get("status") not in {"禁用", "异常"}
    ]
    web_image_accounts = [
        account
        for account in active_accounts
        # Web accounts can be imported by token, password, refresh-token and
        # other login flows. Only Codex-source accounts are incompatible with
        # the ChatGPT Web image path; treating "password" as a separate product
        # source accidentally hid all Web image models in v1.8.2.
        if account_service._normalize_source_type(account.get("source_type")) != "codex"
    ]
    codex_types = {
        normalized
        for account in active_accounts
        if account_service._normalize_source_type(account.get("source_type")) == "codex"
           and (normalized := account_service._normalize_account_type(account.get("type")))
    }

    if web_image_accounts:
        dynamic_models.update(WEB_IMAGE_MODELS)
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

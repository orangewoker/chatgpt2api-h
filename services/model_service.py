from __future__ import annotations

import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock
from typing import Any

from services.account_service import AccountService, account_service
from services.openai_backend_api import OpenAIBackendAPI
from utils.log import logger


# The ChatGPT web catalog also exposes rollout/workflow slugs that are not
# compatible with the plain conversation endpoint used by this service.
# Advertising those through /v1/models makes OpenAI-compatible clients select
# a model that can finish with no assistant message.  Keep the filter narrow:
# normal upstream model slugs continue to be discovered dynamically.
INCOMPATIBLE_MODEL_IDS = frozenset({"research"})
INCOMPATIBLE_MODEL_SUFFIXES = ("-wm",)

# Local routing aliases for the shared ChatGPT Web slug. These are not new
# upstream slugs: send the advertised slug after selecting the matching plan.
GPT6_VARIANT_MODELS = frozenset({"gpt-6-sol", "gpt-6-luna"})
# Known successor slugs used only when the configured image upstream disappears
# from the live ChatGPT catalog (for example gpt-5-5 -> gpt-5-6 -> gpt-6).
IMAGE_UPSTREAM_FALLBACK_MODELS = ("gpt-5-6", "gpt-6")

GPT6_VARIANT_BY_PLAN = {
    "free": "luna",
    "go": "luna",
    "plus": "sol",
    "pro": "sol",
    "prolite": "sol",
    "team": "sol",
    "enterprise": "sol",
    "edu": "sol",
}


def _with_gpt6_variants(
    models: dict[str, dict[str, Any]], account_type: str,
) -> dict[str, dict[str, Any]]:
    """Preserve native model IDs and split an ambiguous gpt-6 by account plan.

    An explicit upstream title takes precedence over the plan-based fallback.
    Existing explicit Sol/Luna slugs also take precedence over local aliases.
    """
    item = models.get("gpt-6")
    if item is None:
        return models
    title = " ".join(str(item.get(field) or "") for field in ("title", "display_name"))
    named_variants = set(re.findall(r"\b(sol|luna)\b", title.lower()))
    if len(named_variants) == 1:
        variant = named_variants.pop()
        basis = "upstream_title"
    else:
        # Do not assume that an unrecognised/combined title names one model.
        if named_variants:
            return models
        variant = GPT6_VARIANT_BY_PLAN.get(account_type.lower())
        basis = "account_plan"
    if not variant:
        return models
    alias = f"gpt-6-{variant}"
    if alias in models:
        return models
    result = dict(models)
    result[alias] = {
        **item,
        "id": alias,
        "root": alias,
        "parent": "gpt-6",
        "owned_by": "chatgpt2api",
        "display_name": f"GPT-6 {variant.title()}",
        "upstream_model": "gpt-6",
        "routing_basis": basis,
    }
    return result


def is_compatible_conversation_model(model_id: object) -> bool:
    normalized = str(model_id or "").strip().lower()
    return bool(
        normalized
        and normalized not in INCOMPATIBLE_MODEL_IDS
        and not normalized.endswith(INCOMPATIBLE_MODEL_SUFFIXES)
    )


@dataclass(frozen=True)
class ModelRoute:
    account_types: frozenset[str]
    allow_anonymous: bool = False


class ModelUnavailableError(RuntimeError):
    pass


class ModelCatalogService:
    """Caches the model catalogs advertised to each active account type."""

    def __init__(
        self,
        accounts: AccountService,
        *,
        backend_factory: Callable[..., Any] = OpenAIBackendAPI,
        cache_ttl_seconds: float = 300,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._accounts = accounts
        self._backend_factory = backend_factory
        self._cache_ttl_seconds = max(1.0, float(cache_ttl_seconds))
        self._clock = clock
        self._lock = RLock()
        self._expires_at = 0.0
        self._account_signature: tuple[tuple[str, int], ...] = ()
        self._anonymous_models: dict[str, dict[str, Any]] = {}
        self._models_by_account_type: dict[str, dict[str, dict[str, Any]]] = {}

    @staticmethod
    def _model_map(result: object) -> dict[str, dict[str, Any]]:
        if not isinstance(result, dict) or not isinstance(result.get("data"), list):
            raise TypeError("upstream model response has no data list")
        models: dict[str, dict[str, Any]] = {}
        for item in result["data"]:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id") or "").strip()
            if is_compatible_conversation_model(model_id) and model_id not in models:
                models[model_id] = dict(item)
        return models

    def _active_accounts_by_type(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for account in self._accounts.list_accounts():
            if not isinstance(account, dict) or account.get("status") in {"禁用", "异常"}:
                continue
            access_token = str(account.get("access_token") or "").strip()
            account_type = self._accounts._normalize_account_type(account.get("type"))
            if access_token and account_type:
                groups.setdefault(account_type, []).append(access_token)
        return groups

    @staticmethod
    def _signature(groups: dict[str, list[str]]) -> tuple[tuple[str, int], ...]:
        return tuple(
            (account_type, len(tokens))
            for account_type, tokens in sorted(groups.items())
        )

    def _fetch_models(self, access_token: str = "") -> dict[str, dict[str, Any]]:
        backend = self._backend_factory(access_token=access_token)
        try:
            return self._model_map(backend.list_models())
        finally:
            backend.close()

    def _fetch_account_type_models(
        self,
        account_type: str,
        access_tokens: list[str],
    ) -> dict[str, dict[str, Any]] | None:
        attempted_tokens: set[str] = set()
        last_error: Exception | None = None
        for access_token in access_tokens:
            try:
                resolved_token = self._accounts.refresh_access_token(
                    access_token,
                    event="list_models",
                ) or access_token
                if resolved_token in attempted_tokens:
                    continue
                attempted_tokens.add(resolved_token)
                return _with_gpt6_variants(self._fetch_models(resolved_token), account_type)
            except Exception as exc:  # noqa: BLE001 - try the next account for any upstream failure
                last_error = exc
        if last_error is not None:
            logger.warning({
                "event": "model_catalog_account_type_failed",
                "account_type": account_type,
                "error_type": type(last_error).__name__,
            })
        return None

    def _refresh(self, groups: dict[str, list[str]], signature: tuple[tuple[str, int], ...]) -> None:
        models_by_account_type: dict[str, dict[str, dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=min(4, len(groups) + 1)) as executor:
            anonymous_future = executor.submit(self._fetch_models)
            account_futures = {
                account_type: executor.submit(
                    self._fetch_account_type_models,
                    account_type,
                    access_tokens,
                )
                for account_type, access_tokens in groups.items()
            }
            try:
                anonymous_models = anonymous_future.result()
            except Exception as exc:  # noqa: BLE001 - retain cached models on upstream failure
                logger.warning({
                    "event": "model_catalog_anonymous_failed",
                    "error_type": type(exc).__name__,
                })
                anonymous_models = self._anonymous_models

            for account_type, future in account_futures.items():
                models = future.result()
                if models is not None:
                    models_by_account_type[account_type] = models
                elif account_type in self._models_by_account_type:
                    models_by_account_type[account_type] = self._models_by_account_type[account_type]

        self._anonymous_models = anonymous_models
        self._models_by_account_type = models_by_account_type
        self._account_signature = signature
        self._expires_at = self._clock() + self._cache_ttl_seconds

    def _ensure_catalog(self) -> None:
        groups = self._active_accounts_by_type()
        signature = self._signature(groups)
        with self._lock:
            if signature == self._account_signature and self._clock() < self._expires_at:
                return
            self._refresh(groups, signature)

    def list_models(self) -> dict[str, Any]:
        self._ensure_catalog()
        with self._lock:
            union: dict[str, dict[str, Any]] = {
                model_id: dict(item)
                for model_id, item in self._anonymous_models.items()
            }
            for account_type in sorted(self._models_by_account_type):
                for model_id, item in self._models_by_account_type[account_type].items():
                    union.setdefault(model_id, dict(item))
            for model_id, item in union.items():
                item["account_types"] = sorted(
                    account_type
                    for account_type, models in self._models_by_account_type.items()
                    if model_id in models
                )
                if model_id == "gpt-6":
                    item["display_name"] = "GPT-6（自动分配）"
                elif model_id in GPT6_VARIANT_MODELS:
                    item["display_name"] = f"GPT-6 {model_id.rsplit('-', 1)[1].title()}"
        return {
            "object": "list",
            "data": [union[model_id] for model_id in sorted(union)],
        }

    def route_for_model(self, model: str) -> ModelRoute:
        model = str(model or "").strip()
        self._ensure_catalog()
        with self._lock:
            account_types = frozenset(
                account_type
                for account_type, models in self._models_by_account_type.items()
                if model in models
            )
            return ModelRoute(
                account_types=account_types,
                allow_anonymous=model in self._anonymous_models,
            )

    @staticmethod
    def _model_key(value: object) -> str:
        """Normalize slugs and display names for catalog lookup.

        The web catalog has used all of ``gpt-5-6``, ``gpt-5.6`` and display
        names such as ``GPT-5.6 sol`` for the same model family.  Comparing a
        compact alphanumeric key lets the resolver accept those values while
        still returning the catalog's real slug.
        """
        return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())

    def image_upstream_model(self, model: str = "", access_token: str = "") -> str:
        """Resolve a configured web-image upstream slug to an active catalog slug.

        Image endpoints historically sent the user-configured upstream model.
        After a web rollout retires that slug, prepare can still succeed while
        the conversation endpoint fails with a bare 404.  Prefer the configured
        value when it is still advertised; otherwise walk known successor
        slugs that exist in the current catalog.
        """
        configured = str(model or "").strip()
        if not configured:
            return configured

        base, separator, suffix = configured.rpartition("-")
        effort_suffix = ""
        if separator and suffix.lower() in {"standard", "extended", "max"}:
            configured = base.strip()
            effort_suffix = f"-{suffix.lower()}"

        self._ensure_catalog()

        # Prefer the catalog belonging to the account which will actually
        # generate the image.  A global union can contain a Sol-only alias
        # belonging to another account type and would otherwise route the
        # request to an invalid slug for the selected account.
        account_models: dict[str, dict[str, Any]] = {}
        if access_token:
            account = self._accounts.get_account(access_token)
            if isinstance(account, dict) and account.get("status") not in {"禁用", "异常"}:
                account_type = self._accounts._normalize_account_type(account.get("type"))
                if account_type:
                    with self._lock:
                        account_models = dict(self._models_by_account_type.get(account_type, {}))

        with self._lock:
            catalog: dict[str, dict[str, Any]] = account_models
            if not catalog:
                catalog = {
                    model_id: dict(item)
                    for model_id, item in self._anonymous_models.items()
                }
                for models in self._models_by_account_type.values():
                    for model_id, item in models.items():
                        catalog.setdefault(model_id, dict(item))

            configured_key = self._model_key(configured)
            # Match the real id first, then upstream metadata and finally the
            # title/display name exposed by ChatGPT's model endpoint.
            for model_id, item in catalog.items():
                candidates = (model_id, item.get("id"), item.get("upstream_model"))
                if configured_key and any(self._model_key(value) == configured_key for value in candidates):
                    return str(item.get("upstream_model") or model_id) + effort_suffix
            for model_id, item in catalog.items():
                titles = (item.get("title"), item.get("display_name"), item.get("name"))
                if configured_key and any(self._model_key(value) == configured_key for value in titles):
                    return str(item.get("upstream_model") or model_id) + effort_suffix

            for candidate in IMAGE_UPSTREAM_FALLBACK_MODELS:
                item = catalog.get(candidate)
                if item is None:
                    candidate_key = self._model_key(candidate)
                    item = next((entry for model_id, entry in catalog.items()
                                 if self._model_key(model_id) == candidate_key), None)
                if item is not None:
                    resolved = str(item.get("upstream_model") or item.get("id") or candidate)
                    logger.warning({
                        "event": "image_upstream_model_fallback",
                        "configured_model": configured,
                        "upstream_model": resolved,
                    })
                    return resolved + effort_suffix

            # A retired 5.5 value must never be sent just because the model
            # catalog is temporarily unavailable.  Keep unknown custom values
            # intact so a private/forward-compatible upstream slug still works.
            configured_key = self._model_key(configured)
            if configured_key in {"gpt55", "gpt55thinking"} or configured_key in {
                self._model_key(alias) for alias in GPT6_VARIANT_MODELS
            }:
                return "gpt-6" + effort_suffix
        return configured + effort_suffix

    def upstream_model_for_account(self, model: str, access_token: str) -> str:
        """Resolve a public variant on the selected account, without downgrade.

        Re-evaluate for every retry: one account may advertise a native slug,
        while another exposes the same variant through the shared Web slug.
        """
        if model not in GPT6_VARIANT_MODELS:
            return model
        self._ensure_catalog()
        account = self._accounts.get_account(access_token) if access_token else None
        with self._lock:
            if account is not None:
                account_type = self._accounts._normalize_account_type(account.get("type"))
                item = self._models_by_account_type.get(account_type or "", {}).get(model)
                if account.get("status") in {"禁用", "异常"}:
                    item = None
            else:
                # Anonymous access is allowed only for a genuinely native slug.
                item = self._anonymous_models.get(model) if not access_token else None
            if item is None:
                raise ModelUnavailableError(
                    f"model {model!r} is not available to the selected account"
                )
            return str(item.get("upstream_model") or model)


model_catalog_service = ModelCatalogService(account_service)

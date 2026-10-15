from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services.model_service import ModelCatalogService, ModelUnavailableError
from utils.helper import UpstreamHTTPError


def model_list(*model_ids: str) -> dict:
    return {
        "object": "list",
        "data": [{"id": model_id, "object": "model"} for model_id in model_ids],
    }


class FakeAccounts:
    def list_accounts(self):
        return []

    def _normalize_account_type(self, value):
        return str(value or "")


class ImageUpstreamFallbackTests(unittest.TestCase):
    def _service(self, models: tuple[str, ...]) -> ModelCatalogService:
        return ModelCatalogService(
            FakeAccounts(),
            backend_factory=lambda access_token="": SimpleNamespace(
                list_models=lambda: model_list(*models),
                close=lambda: None,
            ),
            cache_ttl_seconds=300,
            clock=lambda: 1000.0,
        )

    def test_active_configured_model_is_kept(self) -> None:
        service = self._service(("gpt-5-5", "gpt-5-6", "gpt-6"))

        self.assertEqual(service.image_upstream_model("gpt-5-5"), "gpt-5-5")

    def test_retired_model_falls_back_to_first_active_successor(self) -> None:
        service = self._service(("gpt-5-6", "gpt-6"))

        self.assertEqual(service.image_upstream_model("gpt-5-5"), "gpt-5-6")

    def test_unknown_configured_model_falls_back_to_active_successor(self) -> None:
        service = self._service(("gpt-5-6", "gpt-6"))

        self.assertEqual(service.image_upstream_model("legacy-model"), "gpt-5-6")

    def test_thinking_suffix_is_stripped_by_backend_settings(self) -> None:
        from services.openai_backend_api import OpenAIBackendAPI

        backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
        with patch("services.openai_backend_api.config") as config:
            config.default_upstream_model_name = "gpt-5-6"
            config.default_thinking_effort = "auto"
            with patch.object(
                OpenAIBackendAPI, "_image_model_settings", wraps=backend._image_model_settings,
            ):
                pass
        # Keep this test focused on the suffix parser without importing catalog.
        with patch(
            "services.model_service.ModelCatalogService.image_upstream_model",
            return_value="gpt-5-6-max",
        ):
            model, effort = backend._image_model_settings("gpt-image-2.5-sunburst")
        self.assertEqual((model, effort), ("gpt-5-6", "max"))

    def test_image_upstream_http_error_is_wrapped_with_context(self) -> None:
        error = UpstreamHTTPError("/backend-api/f/conversation", 404, "")
        self.assertEqual(error.status_code, 404)
        self.assertIn("status=404", str(error))
        self.assertIn("/backend-api/f/conversation", str(error))


if __name__ == "__main__":
    unittest.main()

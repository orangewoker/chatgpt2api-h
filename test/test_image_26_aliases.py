"""2.6 is a Web compatibility alias, not an upstream version guarantee."""
import unittest
from unittest.mock import patch

from services.openai_backend_api import OpenAIBackendAPI
from services.protocol import openai_v1_models
from utils.helper import (
    is_codex_image_model,
    is_image_chat_request,
    is_supported_image_model,
    is_web_image_model,
    split_image_model,
)


ALIASES = ("gpt-image-2.6", "gpt-image-2.6-sunburst", "gpt-image-2.6-flare")


class Image26AliasTests(unittest.TestCase):
    def test_aliases_and_date_suffixes_use_web_image_path(self):
        names = ALIASES + tuple(name + "-2026-11-05" for name in ALIASES[1:])
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(split_image_model(name), (None, name))
                self.assertTrue(is_supported_image_model(name))
                self.assertTrue(is_web_image_model(name))
                self.assertTrue(is_image_chat_request({"model": name}))
                self.assertFalse(is_codex_image_model(name))

    def test_unregistered_versions_and_codex_names_are_rejected(self):
        for name in ("gpt-image-2.7", "gpt-image-2.6-other", "codex-gpt-image-2.7",
                     "plus-gpt-image-2.6", "gpt-image-2.6-flare-invalid"):
            with self.subTest(name=name):
                self.assertFalse(is_supported_image_model(name))

    def test_aliases_keep_configured_upstream_not_an_invented_model(self):
        with patch("services.openai_backend_api.config") as config:
            config.default_upstream_model_name = "gpt-5-5"
            config.default_thinking_effort = ""
            for name in ALIASES:
                self.assertEqual(
                    OpenAIBackendAPI._image_model_settings(object.__new__(OpenAIBackendAPI), name),
                    ("gpt-5-5", ""),
                )

    def test_catalog_preserves_source_restrictions(self):
        for source in ("web", "password", "codex"):
            with self.subTest(source=source), patch.object(
                openai_v1_models.model_catalog_service, "list_models",
                return_value={"object": "list", "data": []},
            ), patch.object(openai_v1_models.account_service, "list_accounts", return_value=[
                {"access_token": "test", "type": "Plus", "source_type": source, "status": "正常"},
            ]):
                ids = {item["id"] for item in openai_v1_models.list_models()["data"]}
                self.assertEqual(set(ALIASES).issubset(ids), source != "codex")
                if source == "codex":
                    self.assertFalse(set(ALIASES) & ids)
                    self.assertIn("codex-gpt-image-2", ids)
                    self.assertIn("codex-gpt-image-2.6-flare", ids)
                    self.assertIn("plus-codex-gpt-image-2.6-sunburst", ids)

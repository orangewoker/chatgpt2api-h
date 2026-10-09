from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from services.account_service import AccountService
from services.model_service import ModelCatalogService, ModelUnavailableError
from services.openai_backend_api import OpenAIBackendAPI
from services.protocol import anthropic_v1_messages, conversation, openai_v1_chat_complete, openai_v1_response
from services.protocol.chat_completion_cache import chat_completion_cache
from services.storage.json_storage import JSONStorageBackend
from test.test_model_catalog_service import FakeBackend, model_list


class Gpt6VariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.accounts = AccountService(JSONStorageBackend(Path(self.temp_dir.name) / "accounts.json"))
        self.accounts.add_account_items([
            {"access_token": "free", "type": "free", "status": "正常"},
            {"access_token": "plus-one", "type": "Plus", "source_type": "password", "status": "正常"},
            {"access_token": "plus-two", "type": "Plus", "status": "正常"},
        ])
        self.accounts.refresh_access_token = lambda token, **kwargs: token
        self.now = 1000.0
        self.outcomes = {
            "": model_list("gpt-6"),
            "free": model_list("gpt-6", "gpt-6-mini"),
            "plus-one": model_list("gpt-6", "gpt-6-mini"),
            "plus-two": model_list("gpt-6", "gpt-6-mini"),
        }
        self.catalog = ModelCatalogService(
            self.accounts,
            backend_factory=lambda access_token="": FakeBackend(access_token, self.outcomes, [], []),
            clock=lambda: self.now,
        )
        chat_completion_cache.clear()
        self.addCleanup(chat_completion_cache.clear)

    def test_mixed_plans_publish_distinct_variants_and_keep_shared_id(self) -> None:
        items = {item["id"]: item for item in self.catalog.list_models()["data"]}
        self.assertEqual(set(items), {"gpt-6", "gpt-6-mini", "gpt-6-sol", "gpt-6-luna"})
        self.assertEqual(items["gpt-6-sol"]["account_types"], ["Plus"])
        self.assertEqual(items["gpt-6-luna"]["account_types"], ["free"])
        self.assertEqual(items["gpt-6-sol"]["upstream_model"], "gpt-6")
        self.assertEqual(items["gpt-6-sol"]["routing_basis"], "account_plan")
        self.assertEqual(items["gpt-6"]["display_name"], "GPT-6（自动分配）")
        self.assertTrue(self.catalog.route_for_model("gpt-6").allow_anonymous)

    def test_sol_and_luna_select_only_their_matching_accounts(self) -> None:
        with mock.patch("services.model_service.model_catalog_service", self.catalog):
            sol_tokens = {self.accounts.get_text_access_token(model="gpt-6-sol") for _ in range(8)}
            luna_tokens = {self.accounts.get_text_access_token(model="gpt-6-luna") for _ in range(8)}
        self.assertEqual(sol_tokens, {"plus-one", "plus-two"})
        self.assertEqual(luna_tokens, {"free"})

    def test_exhausted_sol_pool_does_not_fall_back_to_free_or_anonymous(self) -> None:
        with mock.patch("services.model_service.model_catalog_service", self.catalog):
            with self.assertRaises(ModelUnavailableError):
                self.accounts.get_text_access_token(
                    model="gpt-6-sol", excluded_tokens={"plus-one", "plus-two"},
                )

    def test_missing_plan_does_not_publish_its_alias(self) -> None:
        self.accounts.delete_accounts(["plus-one", "plus-two"])
        items = {item["id"] for item in self.catalog.list_models()["data"]}
        self.assertIn("gpt-6-luna", items)
        self.assertNotIn("gpt-6-sol", items)
        self.assertFalse(self.catalog.route_for_model("gpt-6-sol").allow_anonymous)

    def test_adding_and_removing_plan_updates_aliases_before_ttl(self) -> None:
        self.accounts.delete_accounts(["plus-one", "plus-two"])
        self.catalog.list_models()
        self.accounts.add_account_items([{"access_token": "plus-one", "type": "Plus"}])
        self.assertIn("gpt-6-sol", {item["id"] for item in self.catalog.list_models()["data"]})
        self.accounts.delete_accounts(["plus-one"])
        self.assertNotIn("gpt-6-sol", {item["id"] for item in self.catalog.list_models()["data"]})

    def test_explicit_upstream_title_overrides_plan_assumption(self) -> None:
        self.outcomes["plus-one"]["data"][0]["title"] = "GPT-6 Luna"
        self.outcomes["free"]["data"][0]["title"] = "GPT-6 Sol"
        self.assertEqual(self.catalog.route_for_model("gpt-6-luna").account_types, frozenset({"Plus"}))
        self.assertEqual(self.catalog.route_for_model("gpt-6-sol").account_types, frozenset({"free"}))
        items = {item["id"]: item for item in self.catalog.list_models()["data"]}
        self.assertEqual(items["gpt-6-sol"]["routing_basis"], "upstream_title")

    def test_accounts_without_gpt6_do_not_publish_variants(self) -> None:
        self.outcomes["free"] = model_list("gpt-5-6")
        self.outcomes["plus-one"] = model_list("gpt-5-6")
        ids = {item["id"] for item in self.catalog.list_models()["data"]}
        self.assertNotIn("gpt-6-sol", ids)
        self.assertNotIn("gpt-6-luna", ids)

    def test_native_variant_slug_is_not_rewritten(self) -> None:
        self.outcomes["plus-one"] = model_list("gpt-6", "gpt-6-sol", "gpt-6-luna")
        self.assertEqual(self.catalog.upstream_model_for_account("gpt-6-sol", "plus-one"), "gpt-6-sol")
        self.assertEqual(self.catalog.route_for_model("gpt-6-luna").account_types, frozenset({"free", "Plus"}))

    def test_wrong_account_cannot_resolve_alias(self) -> None:
        with self.assertRaises(ModelUnavailableError):
            self.catalog.upstream_model_for_account("gpt-6-sol", "free")
        with self.assertRaises(ModelUnavailableError):
            self.catalog.upstream_model_for_account("gpt-6-luna", "plus-one")
        with self.assertRaises(ModelUnavailableError):
            self.catalog.upstream_model_for_account("gpt-6-sol", "")

    def test_unrelated_and_image_models_are_passed_through(self) -> None:
        for model in ["auto", "gpt-6", "gpt-6-mini", "gpt-image-2", "gpt-image-2.5", "codex-gpt-image-2"]:
            with self.subTest(model=model):
                self.assertEqual(self.catalog.upstream_model_for_account(model, "missing"), model)

    def test_conversation_sends_shared_slug_not_local_alias(self) -> None:
        backend = mock.Mock(access_token="plus-one")
        backend.stream_conversation.return_value = iter(["[DONE]"])
        with mock.patch("services.model_service.model_catalog_service", self.catalog):
            list(conversation.conversation_events(backend, model="gpt-6-sol", prompt="hello"))
        self.assertEqual(backend.stream_conversation.call_args.kwargs["model"], "gpt-6")

    def test_protocol_responses_preserve_requested_alias(self) -> None:
        def backend_factory(access_token):
            return SimpleNamespace(
                access_token=access_token,
                close=lambda: None,
                stream_conversation=lambda **kwargs: iter([
                    json.dumps({"message": {
                        "author": {"role": "assistant"},
                        "channel": "final",
                        "content": {"content_type": "text", "parts": ["hello"]},
                        "recipient": "all",
                    }}),
                    "[DONE]",
                ]),
            )

        with (
            mock.patch("services.model_service.model_catalog_service", self.catalog),
            mock.patch.object(conversation, "account_service", self.accounts),
            mock.patch.object(conversation, "OpenAIBackendAPI", side_effect=backend_factory),
            mock.patch.object(anthropic_v1_messages, "account_service", self.accounts),
            mock.patch.object(anthropic_v1_messages, "OpenAIBackendAPI", side_effect=backend_factory),
        ):
            response = openai_v1_chat_complete.handle({
                "model": "gpt-6-sol", "messages": [{"role": "user", "content": "hi"}],
            })
            self.assertEqual(response["model"], "gpt-6-sol")
            self.assertEqual(response["choices"][0]["message"]["content"], "hello")
            response = openai_v1_response.handle({"model": "gpt-6-luna", "input": "hi"})
            self.assertEqual(response["model"], "gpt-6-luna")
            response = anthropic_v1_messages.handle({
                "model": "gpt-6-sol", "messages": [{"role": "user", "content": "hi"}],
            })
            self.assertEqual(response["model"], "gpt-6-sol")
            self.assertEqual(response["content"][0]["text"], "hello")
            chunks = list(openai_v1_chat_complete.handle({
                "model": "gpt-6-sol", "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }))
            self.assertTrue(all(chunk["model"] == "gpt-6-sol" for chunk in chunks))
            chunks = list(openai_v1_response.handle({"model": "gpt-6-luna", "input": "hi", "stream": True}))
            completed = next(chunk for chunk in chunks if chunk.get("type") == "response.completed")
            self.assertEqual(completed["response"]["model"], "gpt-6-luna")
            chunks = list(anthropic_v1_messages.handle({
                "model": "gpt-6-sol", "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }))
            started = next(chunk for chunk in chunks if chunk.get("type") == "message_start")
            self.assertEqual(started["message"]["model"], "gpt-6-sol")

    def test_token_retry_keeps_sol_pool_and_resolves_each_attempt(self) -> None:
        calls = []

        def backend_factory(access_token):
            def stream(**kwargs):
                calls.append((access_token, kwargs["model"]))
                if access_token == "plus-one":
                    raise RuntimeError("token_invalidated")
                yield json.dumps({"p": "/message/content/parts/0", "o": "append", "v": "ok"})
                yield "[DONE]"
            return SimpleNamespace(access_token=access_token, close=lambda: None, stream_conversation=stream)

        with (
            mock.patch("services.model_service.model_catalog_service", self.catalog),
            mock.patch.object(conversation, "account_service", self.accounts),
            mock.patch.object(conversation, "OpenAIBackendAPI", side_effect=backend_factory),
            mock.patch.object(self.accounts, "remove_invalid_token"),
        ):
            text = conversation.collect_text(SimpleNamespace(access_token="plus-one"), conversation.ConversationRequest(
                model="gpt-6-sol", messages=[{"role": "user", "content": "hi"}],
            ))
        self.assertEqual(text, "ok")
        self.assertEqual(calls, [("plus-one", "gpt-6"), ("plus-two", "gpt-6")])


class UpstreamModelMetadataTests(unittest.TestCase):
    def test_backend_preserves_title_for_variant_detection(self) -> None:
        backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
        backend.access_token = "fake"
        backend.base_url = "https://upstream.test"
        backend.session = mock.Mock()
        backend.session.get.return_value.status_code = 200
        backend.session.get.return_value.json.return_value = {"models": [
            {"slug": "gpt-6", "title": "GPT-6 Sol", "description": "A test model"},
        ]}
        with mock.patch.object(backend, "_bootstrap"), mock.patch.object(backend, "_headers", return_value={}):
            result = backend.list_models()
        self.assertEqual(result["data"][0]["title"], "GPT-6 Sol")
        self.assertEqual(result["data"][0]["description"], "A test model")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from services.account_service import AccountService
from services.image_providers import AgnesImageError
from services.model_service import ModelCatalogService
from services.protocol import openai_v1_models
from services.protocol.openai_v1_response import extract_response_images
from services.protocol.conversation import (
    ConversationRequest,
    ImageGenerationError,
    collect_image_outputs,
    stream_image_outputs_with_pool,
)
from services.storage.json_storage import JSONStorageBackend
from utils.helper import is_agnes_image_model, is_supported_image_model


class FakeModelBackend:
    def __init__(self, access_token: str, calls: list[str]) -> None:
        self.access_token = access_token
        self.calls = calls

    def list_models(self) -> dict:
        self.calls.append(self.access_token)
        return {"object": "list", "data": []}

    def close(self) -> None:
        pass


class FakeAgnesProvider:
    outcomes: dict[str, object] = {}
    calls: list[tuple[str, dict]] = []

    def __init__(self, api_key: str, *, account: dict | None = None) -> None:
        self.api_key = api_key

    def generate(self, **kwargs):
        self.calls.append((self.api_key, kwargs))
        outcome = self.outcomes[self.api_key]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self) -> None:
        pass


class AgnesAccountRoutingTests(unittest.TestCase):
    def test_agnes_account_is_external_and_never_enters_text_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items([{
                "access_token": "agnes-key",
                "type": "agnes",
                "status": "正常",
            }])

            account = service.get_account("agnes-key")
            self.assertIsNotNone(account)
            self.assertEqual(account["provider"], "agnes")
            self.assertEqual(account["source_type"], "api_key")
            self.assertEqual(account["quota_mode"], "external")
            self.assertTrue(service._is_image_account_available(account))
            self.assertEqual(service.get_text_access_token(), "")

            token = service.get_available_access_token(provider="agnes")
            self.assertEqual(token, "agnes-key")
            service.mark_image_result(token, True)
            updated = service.get_account(token)
            self.assertEqual(updated["quota"], 0)
            self.assertEqual(updated["success"], 1)
            self.assertEqual(updated["image_inflight"] if "image_inflight" in updated else 0, 0)

    def test_model_catalog_does_not_use_agnes_key_as_chatgpt_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items([{
                "access_token": "agnes-key",
                "provider": "agnes",
                "status": "正常",
            }])
            calls: list[str] = []
            catalog = ModelCatalogService(
                service,
                backend_factory=lambda access_token="": FakeModelBackend(access_token, calls),
            )

            catalog.list_models()

            self.assertEqual(calls, [""])

    def test_agnes_key_never_enters_oauth_refresh_or_keepalive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items([{
                "access_token": "agnes-key",
                "refresh_token": "must-not-be-used",
                "provider": "agnes",
                "created_at": "2020-01-01T00:00:00+00:00",
            }])

            with mock.patch.object(service, "_request_access_token_refresh") as refresh:
                self.assertEqual(
                    service.refresh_access_token("agnes-key", force=True),
                    "agnes-key",
                )

            refresh.assert_not_called()
            self.assertNotIn("agnes-key", service.list_expiring_access_tokens())
            self.assertNotIn("agnes-key", service.list_refresh_token_keepalive_tokens())

    def test_unknown_provider_is_rejected_instead_of_entering_chatgpt_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))

            with self.assertRaisesRegex(ValueError, "unsupported account provider"):
                service.add_account_items([{
                    "access_token": "must-not-leak",
                    "provider": "agnse",
                }])

            self.assertIsNone(service.get_account("must-not-leak"))
            self.assertEqual(service.get_text_access_token(), "")

    def test_expired_external_rate_limit_recovers_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items([{
                "access_token": "agnes-key",
                "provider": "agnes",
                "status": "限流",
                "restore_at": "2020-01-01T00:00:00+00:00",
            }])

            token = service.get_available_access_token(provider="agnes")
            service.mark_image_result(token, True)

            account = service.get_account(token)
            self.assertEqual(account["status"], "正常")
            self.assertIsNone(account["restore_at"])

    def test_public_model_list_advertises_agnes_only_with_available_key(self) -> None:
        accounts = [{
            "access_token": "agnes-key",
            "provider": "agnes",
            "type": "Agnes",
            "status": "正常",
            "quota": 0,
            "quota_mode": "external",
        }]
        with (
            mock.patch.object(
                openai_v1_models.model_catalog_service,
                "list_models",
                return_value={"object": "list", "data": []},
            ),
            mock.patch.object(openai_v1_models.account_service, "list_accounts", return_value=accounts),
        ):
            result = openai_v1_models.list_models()

        ids = {item["id"] for item in result["data"]}
        self.assertEqual(ids, {"agnes-image-2.1-flash"})


class AgnesGenerationRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeAgnesProvider.calls = []
        FakeAgnesProvider.outcomes = {}

    def test_model_is_registered(self) -> None:
        self.assertTrue(is_supported_image_model("agnes-image-2.1-flash"))
        self.assertTrue(is_agnes_image_model("agnes-image-2.1-flash"))
        self.assertFalse(is_agnes_image_model("gpt-image-2"))

    def test_responses_preserves_multiple_reference_images(self) -> None:
        encoded = base64.b64encode(b"image-bytes").decode("ascii")
        images = extract_response_images([{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded}"},
            ],
        }])

        self.assertEqual(images, [(b"image-bytes", "image/png"), (b"image-bytes", "image/jpeg")])

    def test_generation_uses_agnes_provider_and_openai_result_shape(self) -> None:
        encoded = base64.b64encode(b"generated").decode("ascii")
        FakeAgnesProvider.outcomes = {
            "agnes-key": {
                "created": 123,
                "data": [{"b64_json": encoded, "revised_prompt": "revised"}],
            }
        }
        with (
            mock.patch("services.protocol.conversation.AgnesImageProvider", FakeAgnesProvider),
            mock.patch.object(
                openai_v1_models.account_service,
                "get_available_access_token",
                return_value="agnes-key",
            ),
            mock.patch.object(
                openai_v1_models.account_service,
                "get_account",
                return_value={"provider": "agnes", "quota_mode": "external", "label": "Agnes"},
            ),
            mock.patch.object(openai_v1_models.account_service, "mark_image_result") as mark_result,
            mock.patch("services.protocol.conversation.save_image_bytes", return_value="https://local/images/result.png"),
        ):
            result = collect_image_outputs(stream_image_outputs_with_pool(ConversationRequest(
                model="agnes-image-2.1-flash",
                prompt="draw",
                size="1024x1024",
                response_format="b64_json",
            )))

        self.assertEqual(result["created"], 123)
        self.assertEqual(result["data"], [{
            "b64_json": encoded,
            "url": "https://local/images/result.png",
            "revised_prompt": "revised",
        }])
        self.assertNotIn("_account_email", result)
        self.assertEqual(FakeAgnesProvider.calls[0][0], "agnes-key")
        mark_result.assert_called_once_with("agnes-key", True, release_slot=False)

    def test_auth_failure_rotates_to_next_agnes_key(self) -> None:
        encoded = base64.b64encode(b"generated").decode("ascii")
        FakeAgnesProvider.outcomes = {
            "bad-key": AgnesImageError(
                "invalid credential",
                status_code=401,
                error_type="authentication_error",
                code="invalid_api_key",
            ),
            "good-key": {"created": 123, "data": [{"b64_json": encoded}]},
        }

        def select_key(*, excluded_tokens=None, **_kwargs):
            return "good-key" if "bad-key" in set(excluded_tokens or set()) else "bad-key"

        with (
            mock.patch("services.protocol.conversation.AgnesImageProvider", FakeAgnesProvider),
            mock.patch.object(
                openai_v1_models.account_service,
                "get_available_access_token",
                side_effect=select_key,
            ),
            mock.patch.object(
                openai_v1_models.account_service,
                "get_account",
                side_effect=lambda key: {"provider": "agnes", "label": key},
            ),
            mock.patch.object(openai_v1_models.account_service, "mark_image_result") as mark_result,
            mock.patch.object(openai_v1_models.account_service, "update_account") as update_account,
            mock.patch("services.protocol.conversation.save_image_bytes", return_value="https://local/result.png"),
        ):
            result = collect_image_outputs(stream_image_outputs_with_pool(ConversationRequest(
                model="agnes-image-2.1-flash",
                prompt="draw",
            )))

        self.assertEqual(len(result["data"]), 1)
        self.assertEqual([key for key, _kwargs in FakeAgnesProvider.calls], ["bad-key", "good-key"])
        self.assertEqual(
            mark_result.call_args_list,
            [
                mock.call("bad-key", False, release_slot=False),
                mock.call("good-key", True, release_slot=False),
            ],
        )
        update_account.assert_called_once_with("bad-key", {"status": "异常"}, quiet=True)

    def test_parallel_invalid_request_preserves_structured_400_error(self) -> None:
        FakeAgnesProvider.outcomes = {
            "agnes-key": AgnesImageError(
                "unsupported Agnes image ratio: 5:4",
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_ratio",
            ),
        }
        with (
            mock.patch("services.protocol.conversation.AgnesImageProvider", FakeAgnesProvider),
            mock.patch.object(
                openai_v1_models.account_service,
                "get_available_access_token",
                return_value="agnes-key",
            ),
            mock.patch.object(
                openai_v1_models.account_service,
                "get_account",
                return_value={"provider": "agnes"},
            ),
            mock.patch.object(openai_v1_models.account_service, "release_image_slot"),
        ):
            with self.assertRaises(ImageGenerationError) as caught:
                collect_image_outputs(stream_image_outputs_with_pool(ConversationRequest(
                    model="agnes-image-2.1-flash",
                    prompt="draw",
                    n=2,
                    ratio="5:4",
                )))

        self.assertEqual(getattr(caught.exception, "status_code", None), 400)
        self.assertEqual(getattr(caught.exception, "code", None), "invalid_ratio")

    def test_missing_agnes_key_returns_structured_quota_error(self) -> None:
        with mock.patch.object(
            openai_v1_models.account_service,
            "get_available_access_token",
            side_effect=RuntimeError("no available agnes image quota"),
        ):
            with self.assertRaises(ImageGenerationError) as caught:
                collect_image_outputs(stream_image_outputs_with_pool(ConversationRequest(
                    model="agnes-image-2.1-flash",
                    prompt="draw",
                )))

        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(caught.exception.code, "insufficient_quota")

    def test_rate_limit_uses_retry_after_and_returns_sanitized_429(self) -> None:
        FakeAgnesProvider.outcomes = {
            "limited-key": AgnesImageError(
                "sensitive upstream detail",
                status_code=429,
                error_type="rate_limit_error",
                code="vendor_private_code",
                retry_after_seconds=120,
            ),
        }

        def select_key(*, excluded_tokens=None, **_kwargs):
            if "limited-key" in set(excluded_tokens or set()):
                raise RuntimeError("no available agnes image quota")
            return "limited-key"

        with (
            mock.patch("services.protocol.conversation.AgnesImageProvider", FakeAgnesProvider),
            mock.patch.object(
                openai_v1_models.account_service,
                "get_available_access_token",
                side_effect=select_key,
            ),
            mock.patch.object(
                openai_v1_models.account_service,
                "get_account",
                return_value={"provider": "agnes"},
            ),
            mock.patch.object(openai_v1_models.account_service, "mark_image_result"),
            mock.patch.object(openai_v1_models.account_service, "release_image_slot"),
            mock.patch.object(openai_v1_models.account_service, "update_account") as update_account,
        ):
            with self.assertRaises(ImageGenerationError) as caught:
                collect_image_outputs(stream_image_outputs_with_pool(ConversationRequest(
                    model="agnes-image-2.1-flash",
                    prompt="draw",
                )))

        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(caught.exception.code, "rate_limit_exceeded")
        self.assertNotIn("sensitive upstream detail", str(caught.exception))
        updates = update_account.call_args.args[1]
        self.assertEqual(updates["status"], "限流")
        self.assertTrue(updates["restore_at"])

    def test_account_persist_failure_does_not_fail_paid_result_or_double_release(self) -> None:
        encoded = base64.b64encode(b"generated").decode("ascii")
        FakeAgnesProvider.outcomes = {
            "agnes-key": {"created": 123, "data": [{"b64_json": encoded}]},
        }
        with (
            mock.patch("services.protocol.conversation.AgnesImageProvider", FakeAgnesProvider),
            mock.patch.object(
                openai_v1_models.account_service,
                "get_available_access_token",
                return_value="agnes-key",
            ),
            mock.patch.object(
                openai_v1_models.account_service,
                "get_account",
                return_value={"provider": "agnes"},
            ),
            mock.patch.object(
                openai_v1_models.account_service,
                "mark_image_result",
                side_effect=OSError("storage unavailable"),
            ),
            mock.patch.object(openai_v1_models.account_service, "release_image_slot") as release,
            mock.patch("services.protocol.conversation.save_image_bytes", return_value="https://local/result.png"),
        ):
            result = collect_image_outputs(stream_image_outputs_with_pool(ConversationRequest(
                model="agnes-image-2.1-flash",
                prompt="draw",
            )))

        self.assertEqual(len(result["data"]), 1)
        release.assert_called_once_with("agnes-key")


if __name__ == "__main__":
    unittest.main()

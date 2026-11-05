import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from services.image_storage_service import ImageStorageService, WebDAVClient
from services.image_upscale import upscale_image_bytes
from services.openai_backend_api import OpenAIBackendAPI
from services.protocol import conversation, openai_v1_image_generations
from services.protocol.conversation import ConversationRequest, ImageGenerationError
from test.test_image_task_service import OWNER, ImageTaskServiceTests, wait_for_task
from utils.helper import codex_tool_model, normalize_image_output, split_image_model


def animation():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(
        buf, format="GIF", save_all=True,
        append_images=[Image.new("RGB", (8, 8), "blue")], duration=200, loop=0,
    )
    return buf.getvalue()


class GifTests(unittest.TestCase):
    def test_model_mapping_including_plan_prefixes(self):
        for prefix in ("", "plus-", "team-", "pro-"):
            for suffix, upstream in (("", "flare"), ("-flare", "flare"), ("-sunburst", "sunburst")):
                model = prefix + "codex-gpt-image-2.6" + suffix
                self.assertEqual(codex_tool_model(model), "gpt-image-2.6-" + upstream)
                self.assertIsNotNone(split_image_model(model)[1])
        self.assertEqual(codex_tool_model("codex-gpt-image-2"), "gpt-image-2")
        with self.assertRaises(ValueError):
            codex_tool_model("gpt-image-2.6-flare")

    def test_bounds_and_invalid_combinations(self):
        self.assertEqual(normalize_image_output("gif"), ("gif", 9, 200))
        self.assertEqual(normalize_image_output(), ("png", None, None))
        for args in (("jpg",), ("png", 9), ("gif", 1), ("gif", 65), ("gif", True),
                     ("gif", 2.5), ("gif", 9, 25), ("gif", 9, 2001), ("gif", 9, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                normalize_image_output(*args)

    def test_actual_codex_payload_not_just_alias(self):
        backend = object.__new__(OpenAIBackendAPI)
        backend.access_token = "test-not-a-secret"
        backend.base_url = "https://chatgpt.com"
        for model in ("codex-gpt-image-2", "plus-codex-gpt-image-2.6-flare", "codex-gpt-image-2.6-sunburst"):
            with patch.object(backend, "_ensure_codex_source_account"), patch.object(
                backend, "_iter_codex_response_events", return_value=iter([])
            ), patch("services.openai_backend_api.urllib.request.urlopen") as send:
                list(backend.iter_codex_image_response_events(
                    "ball", model=model, output_format="gif", frame_count=4, frame_duration_ms=100,
                ))
                payload = json.loads(send.call_args.args[0].data)
                tool = payload["tools"][0]
                self.assertEqual(tool["model"], codex_tool_model(model))
                self.assertEqual(tool["output_format"], "gif")
                self.assertEqual(tool["frame_count"], 4)
                self.assertEqual(tool["frame_duration_ms"], 100)
        with patch.object(backend, "_ensure_codex_source_account"), patch.object(
            backend, "_iter_codex_response_events", return_value=iter([])
        ), patch("services.openai_backend_api.urllib.request.urlopen") as send:
            list(backend.iter_codex_image_response_events("ball"))
            tool = json.loads(send.call_args.args[0].data)["tools"][0]
            self.assertEqual(tool["model"], "gpt-image-2")
            self.assertEqual(tool["output_format"], "png")
            self.assertNotIn("frame_count", tool)

    def test_gif_bytes_survive_resize_save_and_result(self):
        raw = animation()
        self.assertEqual(upscale_image_bytes(raw, "4096x4096"), raw)
        self.assertTrue(ImageStorageService().make_relative_path(raw).endswith(".gif"))
        with patch.object(conversation, "save_image_bytes", return_value="https://example.test/a.gif") as save:
            result = conversation.format_image_result(
                [{"b64_json": base64.b64encode(raw).decode()}], "ball", "b64_json", output_format="gif",
            )
            self.assertEqual(save.call_args.args[0], raw)
            self.assertEqual(base64.b64decode(result["data"][0]["b64_json"]), raw)
            self.assertEqual(Image.open(io.BytesIO(raw)).n_frames, 2)

    def test_static_output_cannot_masquerade_as_animation(self):
        for fmt in ("PNG", "GIF"):
            buf = io.BytesIO()
            Image.new("RGB", (8, 8)).save(buf, format=fmt)
            with self.assertRaises(ImageGenerationError):
                conversation.format_image_result(
                    [{"b64_json": base64.b64encode(buf.getvalue()).decode()}], "ball", "url", output_format="gif",
                )

    def test_webdav_mime(self):
        client = WebDAVClient.__new__(WebDAVClient)
        with patch.object(client, "_request") as request, patch.object(client, "ensure_dirs"), patch.object(client, "remote_url", return_value="u"):
            client.put("a.gif", animation(), content_type="image/gif")
            self.assertEqual(request.call_args.kwargs["headers"]["Content-Type"], "image/gif")

    def test_gif_rejected_for_edits_and_gemini_before_account_selection(self):
        for request in (ConversationRequest(model="codex-gpt-image-2.6", output_format="gif", images=["a"]),
                        ConversationRequest(model="gemini-image", output_format="gif")):
            with self.assertRaises(ImageGenerationError):
                list(conversation.stream_image_outputs_with_pool(request))

    def test_generation_protocol_passes_fields(self):
        with patch.object(openai_v1_image_generations, "stream_image_outputs_with_pool", return_value=[]) as stream:
            openai_v1_image_generations.handle({"model": "codex-gpt-image-2.6", "prompt": "ball", "output_format": "gif", "frame_count": 4, "frame_duration_ms": 100})
            request = stream.call_args.args[0]
            self.assertEqual((request.output_format, request.frame_count, request.frame_duration_ms), ("gif", 4, 100))

    def test_tasks_preserve_output_fields(self):
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.json"
            service = ImageTaskServiceTests().make_service(path, lambda payload: seen.append(payload) or {"data": [{"url": "https://example.test/a.gif"}]})
            service.submit_generation(OWNER, client_task_id="gif", prompt="ball", model="codex-gpt-image-2.6", size=None, output_format="gif", frame_count=4, frame_duration_ms=100)
            task = wait_for_task(service, OWNER, "gif", "success")
            self.assertEqual(seen[0]["frame_count"], 4)
            self.assertEqual(task["output_format"], "gif")
            restored = ImageTaskServiceTests().make_service(path).list_tasks(OWNER, ["gif"])["items"][0]
            self.assertEqual(restored["frame_duration_ms"], 100)

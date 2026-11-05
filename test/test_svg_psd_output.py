import base64
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from utils.helper import is_svg_bytes, normalize_image_output
from services.image_storage_service import ImageStorageService
from services.image_upscale import upscale_image_bytes
from services.protocol import conversation, openai_v1_image_generations
from services.svg_output import resolve_svg_outputs
from services import image_file_generation as files

SVG = b'<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12"><path d="M0 0L12 12"/></svg>'

class FormatTests(unittest.TestCase):
    def test_normalize(self):
        for fmt in ('svg', 'psd'):
            self.assertEqual(normalize_image_output(fmt), (fmt, None, None))
            with self.assertRaises(ValueError): normalize_image_output(fmt, 9)

    def test_svg_validation(self):
        for raw in (SVG, b'<?xml version="1.0"?>'+SVG): self.assertTrue(is_svg_bytes(raw))
        for raw in (b'<html/>', b'<svg', b'PNG', b'<!DOCTYPE svg>'+SVG): self.assertFalse(is_svg_bytes(raw))

    def test_preserve_and_save_svg(self):
        self.assertEqual(upscale_image_bytes(SVG, '4096x4096'), SVG)
        self.assertTrue(ImageStorageService().make_relative_path(SVG).endswith('.svg'))
        with patch.object(conversation, 'save_image_bytes', return_value='/images/a.svg') as save:
            result = conversation.format_image_result([{'b64_json':base64.b64encode(SVG).decode()}], 'p', 'b64_json', output_format='svg')
            self.assertEqual(base64.b64decode(result['data'][0]['b64_json']), SVG)
            self.assertEqual(save.call_args.args[0], SVG)
        with self.assertRaises(conversation.ImageGenerationError):
            conversation.format_image_result([{'b64_json':base64.b64encode(b'PNG').decode()}], 'p', 'url', output_format='svg')

    def test_svg_uses_picture_handler(self):
        with patch.object(openai_v1_image_generations, 'stream_image_outputs_with_pool', return_value=iter([])) as pool, patch.object(openai_v1_image_generations, 'collect_image_outputs', return_value={'data': []}):
            openai_v1_image_generations.handle({'prompt':'p', 'output_format':'svg'})
            self.assertEqual(pool.call_args.args[0].output_format, 'svg')
        self.assertIn('SVG', conversation.build_image_prompt('p', None, output_format='svg'))

    def test_disallowed_routes(self):
        for model, images in [('codex-gpt-image-2', []), ('gemini-image', []), ('gpt-image-2', ['x'])]:
            with self.assertRaises(conversation.ImageGenerationError):
                list(conversation.stream_image_outputs_with_pool(conversation.ConversationRequest(model=model, images=images, output_format='svg')))

    def test_inline_svg_does_not_create_file_turn(self):
        backend = MagicMock()
        out = resolve_svg_outputs(backend, 'c', [], [], SVG.decode(), 0)
        self.assertEqual(base64.b64decode(out[0]['b64_json']), SVG)
        backend._run_editable_conversation.assert_not_called()

    def test_svg_asset(self):
        backend = MagicMock()
        backend.resolve_conversation_image_urls.return_value = ['https://example.test/a']
        backend.download_image_bytes.return_value = [SVG]
        self.assertTrue(resolve_svg_outputs(backend, 'c', ['file-a'], [], '', 0))
        backend._run_editable_conversation.assert_not_called()

    def test_svg_attachment(self):
        backend = MagicMock()
        backend.resolve_conversation_image_urls.return_value = []
        backend._get_conversation.return_value = {'mapping': {}}
        backend._extract_editable_artifacts.return_value = [SimpleNamespace(author_role='assistant')]
        backend._looks_like_editable_primary.return_value = True
        backend._resolve_editable_download_url.return_value = 'https://example.test/a.svg'
        backend.session.get.return_value = SimpleNamespace(status_code=200, content=SVG)
        self.assertTrue(resolve_svg_outputs(backend, 'c', [], [], '', 0))

    def test_missing_svg(self):
        backend = MagicMock()
        backend.resolve_conversation_image_urls.return_value = []
        backend._get_conversation.return_value = {'mapping': {}}
        backend._extract_editable_artifacts.return_value = []
        self.assertEqual(resolve_svg_outputs(backend, 'c', [], [], '', 0), [])

    def test_psd_validation(self):
        files.validate_artifact(b'8BPS\x00\x01'+b'\x00'*20, 'psd')
        with self.assertRaises(ValueError): files.validate_artifact(SVG, 'psd')

    def test_psd_file_route(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(files, '_editable_access_token', return_value='token'), patch.object(files, 'OpenAIBackendAPI') as factory, patch.object(files, 'account_service'), patch.object(files, '_file_url', return_value='/files/a.psd'):
            path = Path(tmp)/'a.psd'; path.write_bytes(b'8BPS\x00\x01'+b'\x00'*20)
            backend = factory.return_value
            backend._export_editable_file_zip.return_value = SimpleNamespace(primary_path=path)
            out = openai_v1_image_generations.handle({'prompt':'logo', 'output_format':'psd', 'response_format':'url'})
            self.assertEqual(out['data'][0]['url'], '/files/a.psd')
            self.assertFalse(backend._export_editable_file_zip.call_args.kwargs['include_zip'])
            backend.close.assert_called_once()

    def test_psd_rejects_stream_before_account(self):
        with patch.object(files, '_editable_access_token') as account:
            with self.assertRaises(ValueError): files.generate_image_file({'prompt':'p', 'model':'gpt-image-2', 'output_format':'psd', 'stream':True})
            account.assert_not_called()

if __name__ == '__main__': unittest.main()

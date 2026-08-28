from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from PIL import Image

from parse_bench.inference.providers.base import ProviderPermanentError
from parse_bench.inference.providers.parse.amazon_nova import AmazonNovaProvider
from parse_bench.inference.providers.parse.anthropic import AnthropicProvider
from parse_bench.inference.providers.parse.gemma4 import Gemma4Provider
from parse_bench.inference.providers.parse.google import GoogleProvider
from parse_bench.inference.providers.parse.infinity_parser2 import InfinityParser2Provider
from parse_bench.inference.providers.parse.nemotron_omni import NemotronOmniProvider
from parse_bench.inference.providers.parse.openai import OpenAIProvider
from parse_bench.inference.providers.parse.qwen3_5 import Qwen35Provider
from parse_bench.inference.providers.parse.tesseract import TesseractProvider
from parse_bench.inference.providers.parse.textract import TextractProvider

ENCODERS = [
    (AmazonNovaProvider, "_image_to_jpeg_bytes"),
    (AnthropicProvider, "_image_to_base64"),
    (GoogleProvider, "_image_to_bytes"),
    (OpenAIProvider, "_image_to_base64"),
]

SINGLE_IMAGE_READERS = [Gemma4Provider, NemotronOmniProvider, Qwen35Provider]


@pytest.mark.parametrize(("provider_class", "method_name"), ENCODERS)
@pytest.mark.parametrize("fail", [False, True], ids=["success", "exception"])
def test_vision_encoders_close_derived_images_but_not_caller_image(
    provider_class: type[Any],
    method_name: str,
    fail: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = object.__new__(provider_class)
    provider.MAX_IMAGE_DIMENSION = 4
    provider.MAX_IMAGE_SIZE_BYTES = 1024 * 1024
    original = Image.new("RGBA", (8, 8), "white")
    derived: list[Image.Image] = []
    real_resize = Image.Image.resize
    real_convert = Image.Image.convert
    real_save = Image.Image.save
    inside_pillow_operation = False

    def track(image: Image.Image) -> Image.Image:
        image.close = Mock(wraps=image.close)
        derived.append(image)
        return image

    def resize(image: Image.Image, *args: Any, **kwargs: Any) -> Image.Image:
        nonlocal inside_pillow_operation
        if inside_pillow_operation:
            return real_resize(image, *args, **kwargs)
        inside_pillow_operation = True
        try:
            resized = real_resize(image, *args, **kwargs)
        finally:
            inside_pillow_operation = False
        return track(resized)

    def convert(image: Image.Image, *args: Any, **kwargs: Any) -> Image.Image:
        nonlocal inside_pillow_operation
        if inside_pillow_operation:
            return real_convert(image, *args, **kwargs)
        inside_pillow_operation = True
        try:
            converted = real_convert(image, *args, **kwargs)
        finally:
            inside_pillow_operation = False
        return track(converted)

    def save(image: Image.Image, *args: Any, **kwargs: Any) -> None:
        if fail:
            raise RuntimeError("encoding failed")
        real_save(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", resize)
    monkeypatch.setattr(Image.Image, "convert", convert)
    monkeypatch.setattr(Image.Image, "save", save)

    if fail:
        with pytest.raises(RuntimeError, match="encoding failed"):
            getattr(provider, method_name)(original)
    else:
        assert getattr(provider, method_name)(original)

    assert len(derived) == 2
    assert all(isinstance(image.close, Mock) and image.close.call_count == 1 for image in derived)
    assert original.getpixel((0, 0)) == (255, 255, 255, 255)
    original.close()


@pytest.mark.parametrize("fail", [False, True], ids=["success", "exception"])
def test_textract_closes_all_resizes_but_not_caller_image(fail: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = object.__new__(TextractProvider)
    provider._MAX_DIMENSION = 4
    provider._TARGET_BYTES = 0
    original = Image.new("RGB", (8, 8), "white")
    derived: list[Image.Image] = []
    real_resize = Image.Image.resize
    real_save = Image.Image.save

    def resize(image: Image.Image, *args: Any, **kwargs: Any) -> Image.Image:
        if len(derived) >= 2:
            previous = derived[-1]
            assert isinstance(previous.close, Mock) and previous.close.call_count == 1
        resized = real_resize(image, *args, **kwargs)
        resized.close = Mock(wraps=resized.close)
        derived.append(resized)
        return resized

    save_calls = 0

    def save(image: Image.Image, *args: Any, **kwargs: Any) -> None:
        nonlocal save_calls
        save_calls += 1
        if fail and save_calls == 2:
            raise RuntimeError("encoding failed")
        real_save(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", resize)
    monkeypatch.setattr(Image.Image, "save", save)

    if fail:
        with pytest.raises(RuntimeError, match="encoding failed"):
            provider._resize_image_for_textract(original)
    else:
        assert provider._resize_image_for_textract(original)

    assert derived
    assert all(isinstance(image.close, Mock) and image.close.call_count == 1 for image in derived)
    assert original.getpixel((0, 0)) == (255, 255, 255)
    original.close()


@pytest.mark.parametrize("fail", [False, True], ids=["success", "exception"])
def test_tesseract_single_image_is_closed_on_success_and_failure(
    tmp_path: Path, fail: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (8, 6), "white").save(source)
    provider = object.__new__(TesseractProvider)
    provider._output_type = "text"
    provider._lang = "eng"
    provider._config = ""
    real_open = Image.open
    opened: list[Image.Image] = []

    class TrackedImageContext:
        def __init__(self, path: str | Path) -> None:
            self.image = real_open(path)
            self.image.close = Mock(wraps=self.image.close)
            opened.append(self.image)

        def __enter__(self) -> Image.Image:
            return self.image

        def __exit__(self, *args: object) -> None:
            self.image.close()

    def image_to_string(*args: Any, **kwargs: Any) -> str:
        if fail:
            raise RuntimeError("ocr failed")
        return "page text"

    import pytesseract

    monkeypatch.setattr(Image, "open", TrackedImageContext)
    monkeypatch.setattr(pytesseract, "image_to_string", image_to_string)
    monkeypatch.setattr(pytesseract, "Output", SimpleNamespace(DICT="dict"))

    if fail:
        with pytest.raises(ProviderPermanentError, match="Error during OCR: ocr failed"):
            provider._ocr_image(str(source))
    else:
        result = provider._ocr_image(str(source))
        assert result["pages"][0]["text"] == "page text"
        assert result["pages"][0]["width"] == 8

    assert len(opened) == 1
    assert isinstance(opened[0].close, Mock)
    assert opened[0].close.call_count == 1


@pytest.mark.parametrize("provider_class", SINGLE_IMAGE_READERS)
@pytest.mark.parametrize("fail", [False, True], ids=["success", "exception"])
def test_vllm_single_image_reader_closes_opened_image(
    provider_class: type[Any], tmp_path: Path, fail: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (8, 6), "white").save(source)
    provider = object.__new__(provider_class)
    real_open = Image.open
    opened: list[Image.Image] = []

    class TrackedImageContext:
        def __init__(self, path: str | Path) -> None:
            self.image = real_open(path)
            self.image.close = Mock(wraps=self.image.close)
            opened.append(self.image)

        def __enter__(self) -> Image.Image:
            return self.image

        def __exit__(self, *args: object) -> None:
            self.image.close()

    monkeypatch.setattr(Image, "open", TrackedImageContext)
    if fail:
        monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(RuntimeError("read failed")))

    if fail:
        with pytest.raises(ProviderPermanentError, match="Error reading image file: read failed"):
            provider._read_image_with_size(source)
    else:
        image_bytes, width, height = provider._read_image_with_size(source)
        assert image_bytes
        assert (width, height) == (8, 6)

    assert len(opened) == 1
    assert isinstance(opened[0].close, Mock)
    assert opened[0].close.call_count == 1


def _infinity_provider(parser: Mock, *, deep_parsing: bool = False) -> InfinityParser2Provider:
    provider = object.__new__(InfinityParser2Provider)
    provider._parser = SimpleNamespace(parse=parser)
    provider._task_type = "doc2json"
    provider._batch_size = 1
    provider._output_format = "json"
    provider._max_new_tokens = None
    provider._temperature = 0.0
    provider._deep_parsing_mode = deep_parsing
    provider._model_name = "test-model"
    provider._api_url = "http://provider.invalid"
    provider._base_config = {}
    return provider


@pytest.mark.parametrize("source_kind", ["image", "pdf"])
@pytest.mark.parametrize("fail", [False, True], ids=["success", "exception"])
def test_infinity_parser_closes_opened_and_converted_images(
    source_kind: str, fail: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from parse_bench.inference.providers.parse import infinity_parser2 as infinity_module

    source = tmp_path / f"page.{source_kind if source_kind == 'pdf' else 'png'}"
    source.touch()
    opened_image = Image.new("RGBA", (8, 6), "white")
    opened_image.close = Mock(wraps=opened_image.close)
    converted: list[Image.Image] = []
    real_convert = Image.Image.convert

    def convert(image: Image.Image, *args: Any, **kwargs: Any) -> Image.Image:
        derived = real_convert(image, *args, **kwargs)
        derived.close = Mock(wraps=derived.close)
        converted.append(derived)
        return derived

    monkeypatch.setattr(Image.Image, "convert", convert)
    if source_kind == "pdf":
        monkeypatch.setattr(infinity_module, "convert_from_path", lambda *args, **kwargs: [opened_image])
    else:

        class OpenedImageContext:
            def __enter__(self) -> Image.Image:
                return opened_image

            def __exit__(self, *args: object) -> None:
                opened_image.close()

        monkeypatch.setattr(infinity_module.PILImage, "open", lambda path: OpenedImageContext())

    parser = Mock(side_effect=RuntimeError("parse failed")) if fail else Mock(return_value="[]")
    provider = _infinity_provider(parser)

    if fail:
        with pytest.raises(ProviderPermanentError, match="Error parsing document: parse failed"):
            provider._parse_document(str(source))
    else:
        assert provider._parse_document(str(source))["result"] == "[]"

    assert isinstance(opened_image.close, Mock)
    assert opened_image.close.call_count == 1
    assert len(converted) == 1
    assert isinstance(converted[0].close, Mock)
    assert converted[0].close.call_count == 1


@pytest.mark.parametrize("fail", [False, True], ids=["success", "exception"])
def test_infinity_deep_parsing_closes_every_crop(fail: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from parse_bench.inference.providers.parse import infinity_parser2 as infinity_module

    source = tmp_path / "page.png"
    source.touch()
    opened_image = Image.new("RGB", (8, 8), "white")
    converted_image = Image.new("RGB", (8, 8), "white")
    converted_image.close = Mock(wraps=converted_image.close)
    opened_image.close = Mock(wraps=opened_image.close)

    class OpenedImageContext:
        def __enter__(self) -> Image.Image:
            return opened_image

        def __exit__(self, *args: object) -> None:
            opened_image.close()

    monkeypatch.setattr(infinity_module.PILImage, "open", lambda path: OpenedImageContext())
    monkeypatch.setattr(Image.Image, "convert", lambda image, *args, **kwargs: converted_image)

    crops: list[Image.Image] = []
    real_crop = Image.Image.crop

    def crop(image: Image.Image, *args: Any, **kwargs: Any) -> Image.Image:
        derived = real_crop(image, *args, **kwargs)
        derived.close = Mock(wraps=derived.close)
        crops.append(derived)
        return derived

    monkeypatch.setattr(Image.Image, "crop", crop)
    shallow = json.dumps(
        [
            {"category": "figure", "bbox": [0, 0, 4, 4], "text": "first"},
            {"category": "figure", "bbox": [4, 4, 8, 8], "text": "second"},
        ]
    )
    call_count = 0

    def parse(image: Image.Image, **kwargs: Any) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return shallow
        if fail and call_count == 3:
            raise RuntimeError("deep parse failed")
        return f"table {call_count - 1}"

    provider = _infinity_provider(Mock(side_effect=parse), deep_parsing=True)
    if fail:
        with pytest.raises(ProviderPermanentError, match="Error during deep parsing: deep parse failed") as caught:
            provider._parse_document(str(source))
        assert isinstance(caught.value.__cause__, RuntimeError)
    else:
        result = provider._parse_document(str(source))["result"]
        assert [element["text"] for element in json.loads(result)] == ["table 1", "table 2"]
    assert len(crops) == 2
    assert all(isinstance(image.close, Mock) and image.close.call_count == 1 for image in crops)
    assert isinstance(opened_image.close, Mock) and opened_image.close.call_count == 1
    assert isinstance(converted_image.close, Mock) and converted_image.close.call_count == 1


def test_infinity_deep_parsing_preserves_classified_error_and_closes_crop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from parse_bench.inference.providers.base import ProviderTransientError
    from parse_bench.inference.providers.parse import infinity_parser2 as infinity_module

    source = tmp_path / "page.png"
    source.touch()
    opened_image = Image.new("RGB", (8, 8), "white")
    converted_image = Image.new("RGB", (8, 8), "white")
    opened_image.close = Mock(wraps=opened_image.close)
    converted_image.close = Mock(wraps=converted_image.close)

    class OpenedImageContext:
        def __enter__(self) -> Image.Image:
            return opened_image

        def __exit__(self, *args: object) -> None:
            opened_image.close()

    monkeypatch.setattr(infinity_module.PILImage, "open", lambda path: OpenedImageContext())
    monkeypatch.setattr(Image.Image, "convert", lambda image, *args, **kwargs: converted_image)
    crops: list[Image.Image] = []
    real_crop = Image.Image.crop

    def crop(image: Image.Image, *args: Any, **kwargs: Any) -> Image.Image:
        derived = real_crop(image, *args, **kwargs)
        derived.close = Mock(wraps=derived.close)
        crops.append(derived)
        return derived

    monkeypatch.setattr(Image.Image, "crop", crop)
    shallow = json.dumps([{"category": "figure", "bbox": [0, 0, 4, 4], "text": "figure"}])
    classified = ProviderTransientError("deep provider timed out")
    parser = Mock(side_effect=[shallow, classified])
    provider = _infinity_provider(parser, deep_parsing=True)

    with pytest.raises(ProviderTransientError, match="deep provider timed out") as caught:
        provider._parse_document(str(source))

    assert caught.value is classified
    assert len(crops) == 1
    assert isinstance(crops[0].close, Mock) and crops[0].close.call_count == 1
    assert isinstance(opened_image.close, Mock) and opened_image.close.call_count == 1
    assert isinstance(converted_image.close, Mock) and converted_image.close.call_count == 1


@pytest.mark.parametrize(
    "response",
    [json.dumps({"error": "model diagnostic"}), ""],
    ids=["diagnostic-dict", "empty"],
)
def test_infinity_deep_parsing_rejects_invalid_layout_response(response: str) -> None:
    provider = _infinity_provider(Mock(), deep_parsing=True)
    with Image.new("RGB", (8, 8), "white") as image:
        with pytest.raises(ProviderPermanentError, match="deep-parsing input"):
            provider._apply_deep_parsing(response, image)


@pytest.mark.parametrize("deep_response", ["", "   ", None, {"error": "diagnostic"}])
def test_infinity_deep_parsing_rejects_empty_or_non_text_figure_result(
    deep_response: object,
) -> None:
    shallow = json.dumps([{"category": "figure", "bbox": [0, 0, 4, 4], "text": "shallow figure"}])
    provider = _infinity_provider(Mock(return_value=deep_response), deep_parsing=True)

    with Image.new("RGB", (8, 8), "white") as image:
        with pytest.raises(ProviderPermanentError, match="deep response for figure 1"):
            provider._apply_deep_parsing(shallow, image)


@pytest.mark.parametrize(
    "result",
    [json.dumps({"error": "diagnostic"}), "not json", json.dumps(["invalid element"])],
    ids=["diagnostic-dict", "malformed-json", "non-object-element"],
)
def test_infinity_normalize_rejects_invalid_results(result: str) -> None:
    provider = _infinity_provider(Mock())
    raw_result = SimpleNamespace(
        raw_output={
            "result": result,
            "_config": {"page_width": 8, "page_height": 8},
        },
        pipeline_name="infinity-test",
        request=SimpleNamespace(example_id="document"),
    )

    with pytest.raises(ProviderPermanentError):
        provider._normalize(raw_result)

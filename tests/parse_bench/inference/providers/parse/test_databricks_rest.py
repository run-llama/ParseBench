import base64
import json

import pytest
import requests

from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.base import ProviderPermanentError, ProviderTransientError
from parse_bench.inference.providers.parse.databricks_ai_parse import DatabricksAiParseProvider
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType


@pytest.mark.parametrize("status", [200, 400, 429])
def test_rest_request(tmp_path, monkeypatch, status):
    monkeypatch.delenv("DATABRICKS_SQL_WAREHOUSE_ID", raising=False)
    monkeypatch.delenv("DATABRICKS_AI_PARSE_VOLUME", raising=False)
    pipeline = get_pipeline("databricks_ai_parse")
    provider = DatabricksAiParseProvider(
        "databricks_ai_parse", {**pipeline.config, "host": "https://example.invalid", "token": "test"}
    )
    source = tmp_path / "document.pdf"
    source.write_bytes(b"document bytes")
    parsed = {
        "document": {"pages": [], "elements": [{"type": "text", "content": "Parsed text"}]},
        "metadata": {"id": "request-id"},
    }
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(parsed).encode()

    def post(url, **kwargs):
        assert url == "https://example.invalid/api/2.0/ai-functions/ai_parse_document"
        assert kwargs["json"] == {
            "content": [{"type": "file", "data": base64.b64encode(source.read_bytes()).decode("ascii")}],
            "options": {"version": "2.0"},
        }
        return response

    monkeypatch.setattr(requests, "post", post)
    request = InferenceRequest(example_id="chart", source_file_path=str(source), product_type=ProductType.PARSE)
    if status == 200:
        result = provider.run_inference(pipeline, request)
        assert result.raw_output["ai_parse_document"] == parsed
        assert result.raw_output["_config"]["transport"] == "rest"
        assert provider.normalize(result).output.markdown == "Parsed text"
    else:
        with pytest.raises(ProviderTransientError if status == 429 else ProviderPermanentError):
            provider.run_inference(pipeline, request)

"""Provider for Amazon Nova vision-based PARSE via the Bedrock Converse API.

Amazon Nova 2 models are served through Amazon Bedrock's ``converse`` API
(``bedrock-runtime``). Pages are rendered to images and sent as Converse
``image`` content blocks, mirroring the OpenAI/Anthropic vision parse
providers so the resulting metrics are comparable.

Nova 2 Lite is not available for in-region inference in most Regions, so
pipelines address it through a cross-Region inference profile ID
(``us.``/``eu.``/``jp.`` for a geography, ``global.`` for worldwide routing).
See the model card:
https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-2-lite.html
"""

import io
import os
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, NoReturn

from PIL import Image

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse._layout_utils import (
    SYSTEM_PROMPT_LAYOUT,
    USER_PROMPT_LAYOUT,
    build_layout_pages,
    close_open_ended_bands,
    extract_layout_blocks_lenient,
    items_to_markdown,
    parse_layout_blocks,
    validated_sorted_page_records,
)
from parse_bench.inference.providers.parse._multipage_image import (
    annotate_attempt_costs,
    append_attempt_usages,
    attempt_usages_complete,
    close_derived_images,
    open_document_page_images,
    run_page_with_retries,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

# Cross-Region inference profile prefixes. Bedrock bills a geo profile
# (us./eu./jp.) at the source Region's standard on-demand rate and the global
# profile at its own (cheaper) cross-region-global rate, so pricing is keyed by
# the base model ID plus whether the request goes through `global.`.
_GEO_PROFILE_PREFIXES = ("us.", "eu.", "jp.", "apac.", "global.")

# Amazon Bedrock on-demand token pricing, USD per million tokens (input, output),
# us-east-1 "Standard" service tier. Values come from the AWS Price List API
# (offer code AmazonBedrock, us-east-1), usage types
# `USE1-Nova2.0Lite-{input,output}-tokens` and their `-cross-region-global`
# variants. Reasoning tokens are billed at the output rate.
# Source: https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/us-east-1/index.json
_NOVA_PRICING_PER_M: dict[str, tuple[float, float]] = {
    # base-model-id: (input_per_M, output_per_M)
    "amazon.nova-2-lite-v1:0": (0.33, 2.75),
}

# Same models routed through the `global.` inference profile.
_NOVA_GLOBAL_PRICING_PER_M: dict[str, tuple[float, float]] = {
    "amazon.nova-2-lite-v1:0": (0.30, 2.50),
}

# Reasoning ("extended thinking") is disabled by default on Nova 2; when enabled
# an effort level is required.
_VALID_REASONING_EFFORTS = ("low", "medium", "high")

# Bedrock client error codes that the runner should retry.
_TRANSIENT_ERROR_CODES = frozenset(
    {
        "InternalServerException",
        "ModelNotReadyException",
        "ModelTimeoutException",
        "ServiceUnavailableException",
    }
)
_RATE_LIMIT_ERROR_CODES = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceQuotaExceededException",
    }
)


@register_provider("amazon_nova")
class AmazonNovaProvider(Provider):
    """
    Provider for Amazon Nova vision-based document parsing on Bedrock.

    Renders PDF pages to images and sends each page through the Bedrock
    Converse API with the shared layout prompt, so one pipeline produces both
    the markdown and the per-element bounding boxes.
    """

    PDF_RENDER_DPI = 150

    # Nova image-understanding limits: 8,000 x 8,000 px, 25 MB total request
    # payload. Images are base64-encoded on the wire, so the raw byte budget is
    # 3/4 of the payload cap.
    # Source: https://docs.aws.amazon.com/nova/latest/nova2-userguide/using-multimodal-models.html
    MAX_IMAGE_DIMENSION = 8000
    MAX_IMAGE_SIZE_BYTES = int(24 * 1024 * 1024 * 3 / 4)  # ~18 MB raw -> ~24 MB base64

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - `model`: Bedrock model or inference profile ID
              (default: "us.amazon.nova-2-lite-v1:0")
            - `region`: AWS Region for the bedrock-runtime endpoint
              (default: `$AWS_REGION`, else "us-east-1")
            - `dpi`: DPI for PDF to image conversion (default: 150)
            - `max_tokens`: Max tokens per response (default: 8192, model max 64K)
            - `timeout`: Per-request read timeout in seconds (default: 300)
            - `reasoning_effort`: "low", "medium" or "high" to enable extended
              thinking. Unset (default) leaves reasoning disabled, which is the
              Bedrock default for Nova 2.
            - `temperature`: sampling temperature (default: 0 when reasoning is
              disabled, omitted when it is enabled)
            - `top_p`: nucleus sampling threshold (default: omitted)
        """
        super().__init__(provider_name, base_config)

        self._model: str = self.base_config.get("model", "us.amazon.nova-2-lite-v1:0")
        self._region: str = self.base_config.get("region") or os.environ.get("AWS_REGION") or "us-east-1"
        self._dpi = self.base_config.get("dpi", self.PDF_RENDER_DPI)
        self._max_tokens = self.base_config.get("max_tokens", 8192)
        self._timeout = self.base_config.get("timeout", 300)
        self._reasoning_effort = self.base_config.get("reasoning_effort", None)
        self._top_p = self.base_config.get("top_p", None)
        if self._reasoning_effort is not None and self._reasoning_effort not in _VALID_REASONING_EFFORTS:
            raise ProviderConfigError(
                f"Invalid reasoning_effort '{self._reasoning_effort}'. Must be one of {_VALID_REASONING_EFFORTS}."
            )

        # Nova 2 rejects sampling parameters when reasoning runs at high effort,
        # so temperature defaults to 0 (deterministic parsing) only while
        # reasoning is disabled.
        # https://docs.aws.amazon.com/nova/latest/nova2-userguide/using-converse-api.html
        default_temperature = 0 if self._reasoning_effort is None else None
        self._temperature = self.base_config.get("temperature", default_temperature)
        if self._reasoning_effort is not None and (self._temperature is not None or self._top_p is not None):
            raise ProviderConfigError(
                "Nova 2 rejects temperature/top_p when reasoning is enabled; "
                "drop them from the pipeline config or unset reasoning_effort."
            )

        if not os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
            raise ProviderConfigError(
                "No AWS credentials found. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
                "or AWS_BEARER_TOKEN_BEDROCK for Bedrock access."
            )

        try:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._region,
                config=Config(
                    read_timeout=self._timeout,
                    connect_timeout=30,
                    # run_page_with_retries owns the complete retry budget for
                    # each billable page; botocore performs one HTTP attempt.
                    retries={"total_max_attempts": 1, "mode": "standard"},
                ),
            )
        except ImportError as e:
            raise ProviderConfigError("boto3 package not installed. Run: pip install boto3") from e

    def _base_model_id(self) -> str:
        """Strip any cross-Region inference profile prefix from the model ID."""
        for prefix in _GEO_PROFILE_PREFIXES:
            if self._model.startswith(prefix):
                return self._model[len(prefix) :]
        return self._model

    def _get_pricing(self) -> tuple[float, float] | None:
        """Return known (input_rate, output_rate) in USD per million tokens."""
        table = _NOVA_GLOBAL_PRICING_PER_M if self._model.startswith("global.") else _NOVA_PRICING_PER_M
        return table.get(self._base_model_id())

    def _raise_bedrock_error(self, e: Exception) -> NoReturn:
        """Classify a Bedrock/botocore exception as transient, rate-limited or permanent."""
        from botocore.exceptions import BotoCoreError, ClientError

        if isinstance(e, ClientError):
            code = e.response.get("Error", {}).get("Code", "")
            if code in _RATE_LIMIT_ERROR_CODES:
                raise ProviderRateLimitError(f"Bedrock rate limited ({code}): {e}") from e
            if code in _TRANSIENT_ERROR_CODES:
                raise ProviderTransientError(f"Transient Bedrock error ({code}): {e}") from e
            raise ProviderPermanentError(f"Error calling Bedrock Converse API ({code}): {e}") from e

        if isinstance(e, BotoCoreError):
            # Connection/read timeouts and endpoint resolution blips.
            raise ProviderTransientError(f"Transient error calling Bedrock Converse API: {e}") from e

        raise ProviderPermanentError(f"Error calling Bedrock Converse API: {e}") from e

    @staticmethod
    def _extract_usage(response: dict[str, Any]) -> dict[str, int]:
        """Extract token counts from a Converse response.

        Nova bills reasoning tokens as output tokens and does not report them
        separately, so ``thinking_tokens`` is always 0 — the spend is already
        inside ``output_tokens``.
        """
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return {}
        result = {"thinking_tokens": 0}
        for key, source_key in (("input_tokens", "inputTokens"), ("output_tokens", "outputTokens")):
            value = usage.get(source_key)
            if value is not None:
                result[key] = int(value)
        total_value = usage.get("totalTokens")
        if total_value is not None:
            result["total_tokens"] = int(total_value)
        elif "input_tokens" in result and "output_tokens" in result:
            result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
        return result

    @staticmethod
    def _extract_text(response: dict[str, Any]) -> str:
        """Concatenate the text blocks of a Converse response.

        Reasoning blocks (``reasoningContent``) are skipped — Nova 2 returns
        them as ``[REDACTED]`` and they are not part of the parsed document.
        """
        content = response.get("output", {}).get("message", {}).get("content", []) or []
        return "".join(block["text"] for block in content if isinstance(block, dict) and "text" in block)

    def _prepare_image_for_api(self, image: Image.Image) -> Image.Image:
        """Resize an image that exceeds Nova's pixel limits."""
        width, height = image.size
        max_dim = max(width, height)

        if max_dim <= self.MAX_IMAGE_DIMENSION:
            return image

        scale = self.MAX_IMAGE_DIMENSION / max_dim
        return image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)

    def _image_to_jpeg_bytes(self, image: Image.Image) -> bytes:
        """Encode a PIL image as JPEG bytes within Nova's payload budget."""
        with close_derived_images(image) as track:
            image = track(self._prepare_image_for_api(image))

            if image.mode in ("RGBA", "P"):
                image = track(image.convert("RGB"))

            quality = 85
            min_quality = 20

            while quality >= min_quality:
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=quality)
                data = buffer.getvalue()
                if len(data) <= self.MAX_IMAGE_SIZE_BYTES:
                    return data
                quality -= 10

            while True:
                width, height = image.size
                new_width, new_height = int(width * 0.8), int(height * 0.8)
                if new_width < 100 or new_height < 100:
                    break

                image = track(image.resize((new_width, new_height), Image.Resampling.LANCZOS))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=min_quality)
                data = buffer.getvalue()
                if len(data) <= self.MAX_IMAGE_SIZE_BYTES:
                    return data

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=min_quality)
            return buffer.getvalue()

    def _converse(self, image: Image.Image, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, int], str]:
        """Send one page image to Bedrock Converse and return (text, usage, stop_reason)."""
        image_bytes = self._image_to_jpeg_bytes(image)

        inference_config: dict[str, Any] = {"maxTokens": self._max_tokens}
        if self._temperature is not None:
            inference_config["temperature"] = self._temperature
        if self._top_p is not None:
            inference_config["topP"] = self._top_p

        kwargs: dict[str, Any] = {
            "modelId": self._model,
            "system": [{"text": system_prompt}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}},
                        {"text": user_prompt},
                    ],
                }
            ],
            "inferenceConfig": inference_config,
        }
        if self._reasoning_effort is not None:
            kwargs["additionalModelRequestFields"] = {
                "reasoningConfig": {"type": "enabled", "maxReasoningEffort": self._reasoning_effort}
            }

        try:
            response = self._client.converse(**kwargs)
        except Exception as e:
            self._raise_bedrock_error(e)

        stop_reason = str(response.get("stopReason", ""))
        text = self._extract_text(response)
        usage = self._extract_usage(response)

        # Bedrock's built-in content filter returns HTTP 200 with a canned
        # notice in the text block ("The generated text has been blocked by our
        # content filters.") and zero output tokens. That notice is not page
        # content, so it must never reach ParseOutput. Treated as transient so
        # the runner retries — matching how the Gemini provider handles blocked
        # responses — and a page that stays blocked surfaces as a failed doc
        # rather than as an empty parse.
        if stop_reason == "content_filtered":
            raise ProviderTransientError(
                f"Bedrock content filter blocked the response (stopReason={stop_reason})",
                attempt_stats=usage,
            )
        if not text.strip():
            raise ProviderTransientError(
                f"Bedrock Converse returned no text (stopReason={stop_reason or 'unknown'})",
                attempt_stats=usage,
            )

        return text, usage, stop_reason

    def _parse_image_with_layout(self, image: Image.Image) -> tuple[list[dict[str, Any]], str, dict[str, int], str]:
        """Parse a page image to layout-annotated markdown blocks."""
        text, usage, stop_reason = self._converse(image, SYSTEM_PROMPT_LAYOUT, USER_PROMPT_LAYOUT)
        # Prefer the lenient reader (Nova uses <TABLE>/<p> wrappers and leaves
        # them unclosed); fall back to the strict shared parser if it finds
        # nothing, so this can never score worse than the default path.
        if text.strip() == "[]":
            items: list[dict[str, Any]] = []
        else:
            items = extract_layout_blocks_lenient(text) or parse_layout_blocks(text)
            if not items:
                raise ProviderTransientError(
                    "Bedrock Converse returned malformed non-empty layout output",
                    attempt_stats=usage,
                )
        return close_open_ended_bands(items), text, usage, stop_reason

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        """
        Run inference and return raw results.

        :param pipeline: Pipeline specification
        :param request: Inference request
        :return: Raw inference result
        """
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"AmazonNovaProvider only supports PARSE product type, got {request.product_type}"
            )

        source_path = Path(request.source_file_path)
        if not source_path.exists():
            raise ProviderPermanentError(f"Source file not found: {source_path}")

        supported_extensions = {".pdf", ".png", ".jpg", ".jpeg"}
        if source_path.suffix.lower() not in supported_extensions:
            raise ProviderPermanentError(
                f"AmazonNovaProvider supports {supported_extensions}, got {source_path.suffix}"
            )

        started_at = datetime.now()

        try:
            page_usages: list[dict[str, int]] = []
            api_attempts: list[dict[str, object]] = []

            pages: list[dict[str, Any]] = []
            with open_document_page_images(source_path, dpi=self._dpi) as images:
                for page_index, image in enumerate(images):
                    attempts: list[dict[str, object]] = []
                    items, raw_content, usage, stop_reason = run_page_with_retries(
                        partial(self._parse_image_with_layout, image),
                        provider_name=pipeline.provider_name,
                        page_number=page_index + 1,
                        attempt_ledger=attempts,
                        prior_attempt_ledger=api_attempts,
                    )
                    api_attempts.extend(attempts)
                    append_attempt_usages(page_usages, attempts)
                    pages.append(
                        {
                            "page_index": page_index,
                            "items": items,
                            "raw_content": raw_content,
                            "stop_reason": stop_reason,
                            "width": image.width,
                            "height": image.height,
                        }
                    )
                num_pages = len(images)

            completed_at = datetime.now()
            latency_ms = int((completed_at - started_at).total_seconds() * 1000)

            usage_summary: dict[str, int | float] = {}
            if attempt_usages_complete(page_usages):
                total_input = sum(u["input_tokens"] for u in page_usages)
                total_output = sum(u["output_tokens"] for u in page_usages)
                total_thinking = sum(u["thinking_tokens"] for u in page_usages)
                total_all = sum(u["total_tokens"] for u in page_usages)
                usage_summary.update(
                    {
                        "input_tokens": total_input,
                        "output_tokens": total_output,
                        "thinking_tokens": total_thinking,
                        "total_tokens": total_all,
                        "input_tokens_per_page": total_input / num_pages if num_pages > 0 else 0.0,
                        "output_tokens_per_page": total_output / num_pages if num_pages > 0 else 0.0,
                    }
                )
            pricing = self._get_pricing()
            if pricing is not None and attempt_usages_complete(page_usages):
                input_rate, output_rate = pricing
                annotate_attempt_costs(
                    api_attempts,
                    input_rate_per_million=input_rate,
                    output_rate_per_million=output_rate,
                )
                cost = (total_input * input_rate + (total_output + total_thinking) * output_rate) / 1_000_000
                usage_summary.update(
                    {
                        "cost_usd": cost,
                        "cost_per_page_usd": cost / num_pages if num_pages > 0 else 0.0,
                    }
                )

            config_info: dict[str, Any] = {
                "dpi": self._dpi,
                "max_tokens": self._max_tokens,
                "region": self._region,
            }
            if self._reasoning_effort is not None:
                config_info["reasoning_effort"] = self._reasoning_effort
            if self._temperature is not None:
                config_info["temperature"] = self._temperature
            if self._top_p is not None:
                config_info["top_p"] = self._top_p

            raw_output = {
                "pages": pages,
                "num_pages": num_pages,
                "model": self._model,
                "config": config_info,
                **usage_summary,
                "num_api_calls": len(api_attempts),
                "api_attempts": api_attempts,
            }

            return RawInferenceResult(
                request=request,
                pipeline=pipeline,
                pipeline_name=pipeline.pipeline_name,
                product_type=request.product_type,
                raw_output=raw_output,
                started_at=started_at,
                completed_at=completed_at,
                latency_in_ms=latency_ms,
            )

        except (ProviderPermanentError, ProviderRateLimitError, ProviderTransientError, ProviderConfigError):
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        """
        Normalize raw inference result to produce ParseOutput.

        :param raw_result: Raw inference result from run_inference()
        :return: Inference result with both raw and normalized outputs
        """
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"AmazonNovaProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        pages: list[PageIR] = []
        page_markdowns: list[str] = []
        layout_pages: list[ParseLayoutPageIR] = []

        for page_data in validated_sorted_page_records(raw_result.raw_output.get("pages", [])):
            page_index = page_data.get("page_index", 0)

            items = page_data.get("items", [])
            image_width = page_data.get("width", 0)
            image_height = page_data.get("height", 0)
            markdown = items_to_markdown(items)
            layout_pages.extend(
                build_layout_pages(
                    items,
                    image_width,
                    image_height,
                    markdown,
                    page_number=page_index + 1,
                )
            )

            pages.append(PageIR(page_index=page_index, markdown=markdown))
            page_markdowns.append(markdown)

        full_markdown = "\n\n".join(page_markdowns)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            markdown=full_markdown,
            layout_pages=layout_pages,
        )

        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_result.raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )

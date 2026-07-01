import torch
from transformers import Gemma3Config
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.gemma3 import Gemma3ForConditionalGeneration

from ltx_core.loader import KeyValueOperationResult
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
from ltx_core.text_encoders.gemma.embeddings_connector import (
    AudioEmbeddings1DConnectorConfigurator,
    Embeddings1DConnectorConfigurator,
)
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor
from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
from ltx_core.text_encoders.gemma.feature_extractor import (
    FeatureExtractorV1,
    FeatureExtractorV2,
)


class GemmaTextEncoderConfigurator(ModelConfigurator[GemmaTextEncoder]):
    @classmethod
    def from_config(cls, config: dict) -> GemmaTextEncoder:  # noqa: ARG003
        gemma_config = Gemma3Config.from_dict(GEMMA3_CONFIG_FOR_LTX.to_dict())
        with torch.device("meta"):
            model = Gemma3ForConditionalGeneration(gemma_config)

        return GemmaTextEncoder(model=model)


class EmbeddingsProcessorConfigurator(ModelConfigurator[EmbeddingsProcessor]):
    @classmethod
    def from_config(cls, config: dict) -> EmbeddingsProcessor:
        transformer_config = config.get("transformer", {})

        # Create video embeddings connector (always needed)
        video_connector = Embeddings1DConnectorConfigurator.from_config(config)

        # Create audio embeddings connector
        audio_connector = AudioEmbeddings1DConnectorConfigurator.from_config(config)

        # Create feature extractor
        feature_extractor = _create_feature_extractor(transformer_config)

        return EmbeddingsProcessor(
            video_connector=video_connector,
            audio_connector=audio_connector,
            feature_extractor=feature_extractor,
        )


_V2_EXPECTED_CONFIG = {
    "caption_proj_before_connector": True,
    "caption_projection_first_linear": False,
    "caption_proj_input_norm": False,
    "caption_projection_second_linear": False,
}


def _create_feature_extractor(transformer_config: dict) -> torch.nn.Module:
    """Select and create the appropriate feature extractor based on config.
    Detection logic:
    - V1: V2 config keys absent → projection lives in transformer
    - V2: V2 config keys present with exact expected values → per-token RMS norm with dual aggregate embeds
    - Anything else: NotImplementedError (config drift)
    """
    gemma_text_config = GEMMA3_CONFIG_FOR_LTX.text_config
    embedding_dim = gemma_text_config.hidden_size
    num_layers = gemma_text_config.num_hidden_layers + 1  # +1 for the embedding layer
    flat_dim = embedding_dim * num_layers

    overlapping_keys = transformer_config.keys() & _V2_EXPECTED_CONFIG.keys()
    if not overlapping_keys:
        aggregate_embed = torch.nn.Linear(flat_dim, embedding_dim, bias=False)
        return FeatureExtractorV1(aggregate_embed=aggregate_embed, is_av=True)

    missing_keys = _V2_EXPECTED_CONFIG.keys() - overlapping_keys
    if missing_keys:
        raise NotImplementedError("Partial V2 config — missing keys: " + ", ".join(sorted(missing_keys)))

    unexpected_value_keys = {k for k in overlapping_keys if transformer_config[k] != _V2_EXPECTED_CONFIG[k]}
    if unexpected_value_keys:
        raise NotImplementedError(
            "Unknown config: "
            + ", ".join(
                f"{k}={transformer_config[k]!r} (expected {_V2_EXPECTED_CONFIG[k]!r})" for k in unexpected_value_keys
            )
        )

    video_inner_dim = transformer_config["num_attention_heads"] * transformer_config["attention_head_dim"]
    audio_inner_dim = transformer_config["audio_num_attention_heads"] * transformer_config["audio_attention_head_dim"]
    return FeatureExtractorV2(
        video_aggregate_embed=torch.nn.Linear(flat_dim, video_inner_dim, bias=True),
        embedding_dim=embedding_dim,
        audio_aggregate_embed=torch.nn.Linear(flat_dim, audio_inner_dim, bias=True),
    )


# --- Split SDOps: Gemma LLM keys vs Embeddings Processor keys ---

GEMMA_LLM_KEY_OPS = (
    SDOps("GEMMA_LLM_KEY_OPS")
    # 1. Map language model layers (note the double .model prefix)
    .with_matching(prefix="language_model.model.")
    .with_replacement("language_model.model.", "model.model.language_model.")
    # 2. Map the Vision Tower
    .with_matching(prefix="vision_tower.")
    .with_replacement("vision_tower.vision_model.", "model.model.vision_tower.")
    .with_replacement("vision_tower.", "model.model.vision_tower.")
    # 3. Map the Multi-Modal Projector
    .with_matching(prefix="multi_modal_projector.")
    .with_replacement("multi_modal_projector.", "model.model.multi_modal_projector.")
    # 4. Duplicate embed_tokens to lm_head (needed for prompt enhancement via generate())
    .with_kv_operation(
        operation=lambda key, value: [
            KeyValueOperationResult(key, value),
            KeyValueOperationResult("model.lm_head.weight", value),
        ],
        key_prefix="model.model.language_model.embed_tokens.weight",
    )
)

EMBEDDINGS_PROCESSOR_KEY_OPS = (
    SDOps("EMBEDDINGS_PROCESSOR_KEY_OPS")
    # 1. Map the feature extractor (V1: aggregate_embed inside feature_extractor)
    .with_matching(prefix="text_embedding_projection.aggregate_embed.")
    .with_replacement("text_embedding_projection.aggregate_embed.", "feature_extractor.aggregate_embed.")
    # V2 dual aggregate embeds
    .with_matching(prefix="text_embedding_projection.video_aggregate_embed.")
    .with_replacement("text_embedding_projection.video_aggregate_embed.", "feature_extractor.video_aggregate_embed.")
    .with_matching(prefix="text_embedding_projection.audio_aggregate_embed.")
    .with_replacement("text_embedding_projection.audio_aggregate_embed.", "feature_extractor.audio_aggregate_embed.")
    # 2. Map the connectors
    .with_matching(prefix="model.diffusion_model.video_embeddings_connector.")
    .with_replacement("model.diffusion_model.video_embeddings_connector.", "video_connector.")
    .with_matching(prefix="model.diffusion_model.audio_embeddings_connector.")
    .with_replacement("model.diffusion_model.audio_embeddings_connector.", "audio_connector.")
)

VIDEO_ONLY_EMBEDDINGS_PROCESSOR_KEY_OPS = (
    SDOps("VIDEO_ONLY_EMBEDDINGS_PROCESSOR_KEY_OPS")
    # 1. Map the feature extractor (V1: aggregate_embed inside feature_extractor)
    .with_matching(prefix="text_embedding_projection.aggregate_embed.")
    .with_replacement("text_embedding_projection.aggregate_embed.", "feature_extractor.aggregate_embed.")
    # V2 video aggregate embed
    .with_matching(prefix="text_embedding_projection.video_aggregate_embed.")
    .with_replacement("text_embedding_projection.video_aggregate_embed.", "feature_extractor.video_aggregate_embed.")
    # 2. Map the connectors
    .with_matching(prefix="model.diffusion_model.embeddings_connector.")
    .with_replacement("model.diffusion_model.embeddings_connector.", "embeddings_processor.video_connector.")
)


def _resolve_rope_type(config: Gemma3Config) -> str:
    """Resolve RoPE type from config with legacy-key compatibility.

    Some transformer versions store rope_scaling as {"type": ...} while newer
    versions use {"rope_type": ...}. We normalize to the newer shape so
    downstream code sees a consistent config.
    """
    def _fallback_rope_type() -> str:
        # Different transformers versions expose different keys for unscaled RoPE.
        for candidate in ("default", "none", "linear", "dynamic", "yarn", "longrope", "llama3"):
            if candidate in ROPE_INIT_FUNCTIONS:
                return candidate
        supported = ", ".join(sorted(ROPE_INIT_FUNCTIONS.keys()))
        raise ValueError(f"No usable rope initializer found. Supported rope types: {supported}")

    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        return _fallback_rope_type()

    if not isinstance(rope_scaling, dict):
        raise TypeError(f"Expected rope_scaling to be dict or None, got {type(rope_scaling).__name__}")

    rope_type = rope_scaling.get("rope_type")
    legacy_type = rope_scaling.get("type")

    if rope_type is None and legacy_type is not None:
        normalized_scaling = dict(rope_scaling)
        normalized_scaling["rope_type"] = legacy_type
        config.rope_scaling = normalized_scaling
        rope_type = legacy_type

    if rope_type is None:
        # Infer the intended mode from known rope_scaling payload shapes.
        if "short_factor" in rope_scaling and "long_factor" in rope_scaling:
            rope_type = "longrope"
        elif "low_freq_factor" in rope_scaling and "high_freq_factor" in rope_scaling:
            rope_type = "llama3"
        elif "beta_fast" in rope_scaling and "beta_slow" in rope_scaling:
            rope_type = "yarn"
        elif "factor" in rope_scaling:
            rope_type = "linear"
        else:
            rope_type = _fallback_rope_type()

    if rope_type not in ROPE_INIT_FUNCTIONS:
        # Alias across transformers versions: some builds used "none" instead of "default".
        if rope_type == "default" and "none" in ROPE_INIT_FUNCTIONS:
            rope_type = "none"
        elif rope_type == "none" and "default" in ROPE_INIT_FUNCTIONS:
            rope_type = "default"
        else:
            supported = ", ".join(sorted(ROPE_INIT_FUNCTIONS.keys()))
            raise ValueError(f"Unsupported rope_type={rope_type!r}. Supported rope types: {supported}")

    return rope_type


def _normalize_rope_scaling_for_type(config: Gemma3Config, rope_type: str) -> None:
    """Normalize rope_scaling payload so selected rope_type has required fields.

    Some checkpoints/configs expose legacy keys or omit optional fields that
    newer transformers initializers expect.
    """
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        rope_scaling = {}
    if not isinstance(rope_scaling, dict):
        raise TypeError(f"Expected rope_scaling to be dict or None, got {type(rope_scaling).__name__}")

    normalized = dict(rope_scaling)
    normalized["rope_type"] = rope_type

    if rope_type == "linear":
        factor = normalized.get("factor")
        if factor is None:
            factor = normalized.get("scaling_factor")
        if factor is None:
            factor = normalized.get("scale")
        if factor is None:
            # Neutral linear scale when config omits factor.
            factor = 1.0
        normalized["factor"] = float(factor)

    config.rope_scaling = normalized


def _normalize_rope_base_params(config: Gemma3Config) -> None:
    """Ensure base RoPE parameters are populated for transformers initializers.

    Some environments deserialize Gemma text config with None for rope_theta.
    Transformers RoPE init expects a numeric base and will fail with None.
    """
    default_text_cfg = GEMMA3_CONFIG_FOR_LTX.text_config
    if getattr(config, "rope_theta", None) is None:
        config.rope_theta = default_text_cfg.rope_theta
    if getattr(config, "rope_local_base_freq", None) is None:
        config.rope_local_base_freq = default_text_cfg.rope_local_base_freq


def _register_inv_freq_if_present(module: torch.nn.Module, attr_name: str, inv_freq: torch.Tensor) -> None:
    rotary_emb = getattr(module, attr_name, None)
    if rotary_emb is not None:
        rotary_emb.register_buffer("inv_freq", inv_freq)


def _register_gemma3_rotary_buffers(l_model: torch.nn.Module, inv_freqs: torch.Tensor, local_rope_freqs: torch.Tensor) -> None:
    rotary_emb = getattr(l_model, "rotary_emb", None)
    if rotary_emb is None:
        return

    buffer_names = set(dict(rotary_emb.named_buffers()).keys())
    if "full_attention_inv_freq" in buffer_names:
        rotary_emb.register_buffer("full_attention_inv_freq", inv_freqs)
    if "full_attention_original_inv_freq" in buffer_names:
        rotary_emb.register_buffer("full_attention_original_inv_freq", inv_freqs)
    if "sliding_attention_inv_freq" in buffer_names:
        rotary_emb.register_buffer("sliding_attention_inv_freq", local_rope_freqs)
    if "sliding_attention_original_inv_freq" in buffer_names:
        rotary_emb.register_buffer("sliding_attention_original_inv_freq", local_rope_freqs)


def create_and_populate(module: GemmaTextEncoder) -> GemmaTextEncoder:
    model = module.model
    # Transformers variants may expose either:
    # 1) model.model.vision_tower.vision_model (wrapper style), or
    # 2) model.model.vision_tower (SiglipVisionModel directly).
    vision_tower = model.model.vision_tower
    v_model = getattr(vision_tower, "vision_model", vision_tower)
    l_model = model.model.language_model

    config = model.config.text_config
    _normalize_rope_base_params(config)
    dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    # Some transformers/Gemma3TextConfig variants do not expose `rope_local_base_freq`.
    # Use the canonical local RoPE default when missing.
    base = getattr(config, "rope_local_base_freq", 10000)
    local_rope_freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim))
    rope_type = _resolve_rope_type(config)
    _normalize_rope_scaling_for_type(config, rope_type)
    inv_freqs, _ = ROPE_INIT_FUNCTIONS[rope_type](config)

    positions_length = len(v_model.embeddings.position_ids[0])
    position_ids = torch.arange(positions_length, dtype=torch.long, device="cpu").unsqueeze(0)
    v_model.embeddings.register_buffer("position_ids", position_ids)
    embed_scale = torch.tensor(model.config.text_config.hidden_size**0.5, device="cpu")
    l_model.embed_tokens.register_buffer("embed_scale", embed_scale)
    _register_inv_freq_if_present(l_model, "rotary_emb_local", local_rope_freqs)
    _register_inv_freq_if_present(l_model, "rotary_emb", inv_freqs)
    _register_gemma3_rotary_buffers(l_model, inv_freqs, local_rope_freqs)

    return module


GEMMA_MODEL_OPS = ModuleOps(
    name="GemmaModel",
    matcher=lambda module: hasattr(module, "model") and isinstance(module.model, Gemma3ForConditionalGeneration),
    mutator=create_and_populate,
)

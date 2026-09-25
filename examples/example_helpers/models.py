"""Shared model presets for the Strands examples."""

from botocore.config import Config
from strands.models import BedrockModel, CacheConfig

# Enable automatic prompt caching across an agent's turns.
_CACHE = CacheConfig(strategy="auto")

large = BedrockModel(
    model_id="global.anthropic.claude-opus-5",
    # Long agent runs can produce long messages, and a first response to a large prompt can be slow.
    max_tokens=65536,
    boto_client_config=Config(read_timeout=900, connect_timeout=30),
    cache_config=_CACHE,
)
medium = BedrockModel(cache_config=_CACHE)
small = BedrockModel(model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0", cache_config=_CACHE)

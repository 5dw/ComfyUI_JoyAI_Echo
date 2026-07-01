"""ComfyUI nodes for JoyAI-Echo: minute-level multi-shot audio-video generation."""

import sys
from pathlib import Path

_NODE_ROOT = Path(__file__).resolve().parent
_LIBS = str(_NODE_ROOT / "libs")

if _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)

from .nodes import JoyEcho_ModelLoader, JoyEcho_TextEncode, JoyEcho_Generate, JoyEcho_SingleShotGenerate, JoyEcho_PromptFormat, JoyEcho_LLMEnhance, JoyEcho_StoryToVideo, JoyEcho_StoryShotToVideo, JoyEcho_PromptAtIndex

NODE_CLASS_MAPPINGS = {
    "JoyEcho_ModelLoader": JoyEcho_ModelLoader,
    "JoyEcho_TextEncode": JoyEcho_TextEncode,
    "JoyEcho_Generate": JoyEcho_Generate,
    "JoyEcho_SingleShotGenerate": JoyEcho_SingleShotGenerate,
    "JoyEcho_PromptFormat": JoyEcho_PromptFormat,
    "JoyEcho_LLMEnhance": JoyEcho_LLMEnhance,
    "JoyEcho_StoryToVideo": JoyEcho_StoryToVideo,
    "JoyEcho_StoryShotToVideo": JoyEcho_StoryShotToVideo,
    "JoyEcho_PromptAtIndex": JoyEcho_PromptAtIndex,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoyEcho_ModelLoader": "JoyEcho Model Loader",
    "JoyEcho_TextEncode": "JoyEcho Text Encode",
    "JoyEcho_Generate": "JoyEcho Generate (Multi-Shot)",
    "JoyEcho_SingleShotGenerate": "JoyEcho Single Shot Generate",
    "JoyEcho_PromptFormat": "JoyEcho Prompt Format (Helper)",
    "JoyEcho_LLMEnhance": "JoyEcho LLM Enhance",
    "JoyEcho_StoryToVideo": "JoyEcho Story To Video",
    "JoyEcho_StoryShotToVideo": "JoyEcho Story Shot To Video",
    "JoyEcho_PromptAtIndex": "JoyEcho Prompt At Index",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

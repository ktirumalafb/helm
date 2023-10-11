from typing import Dict, Optional, Union
from dataclasses import dataclass
import re
import os
from helm.common.hierarchical_logger import hlog
from helm.proxy.models import (
    Model,
    ALL_MODELS,
    MODEL_NAME_TO_MODEL,
    TEXT_MODEL_TAG,
    FULL_FUNCTIONALITY_TEXT_MODEL_TAG,
    LOCAL_HUGGINGFACE_MODEL_TAG,
)

# The path where local HuggingFace models should be downloaded or symlinked, e.g. ./huggingface_models/llama-7b
LOCAL_HUGGINGFACE_MODEL_DIR = "huggingface_models"


@dataclass(frozen=True)
class HuggingFaceHubModelConfig:
    namespace: Optional[str]
    """Name of the group or user that owns the model. e.g. 'stanford-crfm'

    May be None if the model (e.g. gpt2) does not have a namespace."""

    model_name: str
    """Name of the model. e.g. 'BioMedLM'

    Does not include the namespace."""

    revision: Optional[str]
    """Revision of the model to use e.g. 'main'.

    If None, use the default revision."""
    load_adapters: bool
    num_bits: int
    load_layer_norm: bool
    layer_norm_weights_path: str

    @property
    def model_id(self) -> str:
        """Return the model ID.

        Examples:
        - 'gpt2'
        - 'stanford-crfm/BioMedLM'"""
        if self.namespace:
            return f"{self.namespace}/{self.model_name}"
        return self.model_name

    def __str__(self) -> str:
        """Return the full model name used by HELM in the format "[namespace/]model_name[@revision]".

        Examples:
        - 'gpt2'
        - 'stanford-crfm/BioMedLM'
        - 'stanford-crfm/BioMedLM@main'"""
        result = self.model_name
        if self.namespace:
            result = f"{self.namespace}/{result}"
        if self.revision:
            result = f"{result}@{self.revision}"
        return result

    @staticmethod
    def from_string(raw: str, num_bits: int = 16, load_adapters: bool = False, load_layer_norm: bool = False, layer_norm_weights_path: str = None) -> "HuggingFaceHubModelConfig":
        """Parses a string in the format "[namespace/]model_name[@revision]" to a HuggingFaceHubModelConfig.

        Examples:
        - 'gpt2'
        - 'stanford-crfm/BioMedLM'
        - 'stanford-crfm/BioMedLM@main'"""
        pattern = r"((?P<namespace>[^/@]+)/)?(?P<model_name>[^/@]+)(@(?P<revision>[^/@]+))?"
        match = re.fullmatch(pattern, raw)
        if not match:
            raise ValueError(f"Could not parse model name: '{raw}'; Expected format: [namespace/]model_name[@revision]")
        model_name = match.group("model_name")
        assert model_name
        return HuggingFaceHubModelConfig(
            namespace=match.group("namespace"), model_name=model_name, revision=match.group("revision"), num_bits=num_bits, load_adapters=load_adapters, load_layer_norm=load_layer_norm, layer_norm_weights_path=layer_norm_weights_path
        )

@dataclass(frozen=True)
class HuggingFaceLocalModelConfig:
    model_name: str
    """Name of the model. e.g. 'llama-7b'"""

    path: str
    """Local path to the Hugging Face model weights.
    For pre-registered local models that are already in _huggingface_model_registry below,
    this will get set to LOCAL_HUGGINGFACE_MODEL_DIR by default.
    Otherwise, this is specified using the flag --enable-local-huggingface-models <path>."""

    load_adapters: bool
    num_bits: int
    layer_dropping: bool
    layers_to_drop: str
    load_layer_norm: bool
    layer_norm_weights_path: str

    @property
    def model_id(self) -> str:
        """Return the model ID.

        Examples:
        - 'huggingface/llama-7b'"""
        return f"huggingface/{self.model_name}"

    def __str__(self) -> str:
        """Return the full model name used by HELM in the format "[namespace/]model_name[@revision]".
        Local models don't have a revision and the namespace is set to huggingface.

        Examples:
        - 'huggingface/llama-7b'"""
        return f"huggingface/{self.model_name}"

    @staticmethod
    def from_path(path: str, num_bits: int = 16, load_adapters: bool = False, layer_dropping:bool = False, layers_to_drop:str = None,  load_layer_norm: bool = False, layer_norm_weights_path: str = None) -> "HuggingFaceLocalModelConfig":
        """Generates a HuggingFaceHubModelConfig from a (relative or absolute) path to a local HuggingFace model."""
        model_name = os.path.split(path)[-1]
        return HuggingFaceLocalModelConfig(model_name=model_name, path=path, load_adapters=load_adapters, num_bits=num_bits, layer_dropping=layer_dropping, layers_to_drop=layers_to_drop, load_layer_norm=load_layer_norm, layer_norm_weights_path=layer_norm_weights_path)


HuggingFaceModelConfig = Union[HuggingFaceHubModelConfig, HuggingFaceLocalModelConfig]


# Initialize registry with local models from models.py
_huggingface_model_registry: Dict[str, HuggingFaceModelConfig] = {
    model.name: HuggingFaceLocalModelConfig.from_path(os.path.join(LOCAL_HUGGINGFACE_MODEL_DIR, model.engine))
    for model in ALL_MODELS
    if LOCAL_HUGGINGFACE_MODEL_TAG in model.tags
}


def register_huggingface_hub_model_config(model_name: str, num_bits: int, load_adapters: bool, load_layer_norm: bool, layer_norm_weights_path: str) -> HuggingFaceHubModelConfig:
    """Register a AutoModelForCausalLM model from Hugging Face Model Hub for later use.

    model_name format: namespace/model_name[@revision]"""
    config = HuggingFaceHubModelConfig.from_string(model_name, num_bits, load_adapters, load_layer_norm, layer_norm_weights_path)
    if config.model_id in _huggingface_model_registry:
        raise ValueError(f"A Hugging Face model is already registered for model_id {model_name}")
    _huggingface_model_registry[config.model_id] = config

    # HELM model names require a namespace
    if not config.namespace:
        raise Exception("Registration of Hugging Face models without a namespace is not supported")
    if config.model_id in MODEL_NAME_TO_MODEL:
        raise ValueError(f"A HELM model is already registered for model name: {config.model_id}")
    description = f"HuggingFace model {config.model_id}"
    if config.revision:
        description += f" at revision {config.revision}"
    model = Model(
        group=config.namespace,
        name=model_name,
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG],
    )
    MODEL_NAME_TO_MODEL[config.model_id] = model
    ALL_MODELS.append(model)
    hlog(f"Registered Hugging Face model: {model} config: {config}")
    return config


def register_huggingface_local_model_config(path: str, num_bits: int, load_adapters: bool, layer_dropping: bool, layers_to_drop: str) -> HuggingFaceLocalModelConfig:
    """Register a AutoModelForCausalLM model from a local directory for later use.

    path: a path to your HF model"""
    config = HuggingFaceLocalModelConfig.from_path(path, num_bits, load_adapters, layer_dropping, layers_to_drop)
    if config.model_id in _huggingface_model_registry:
        raise ValueError(f"A Hugging Face model is already registered for model_id {config.model_id}")
    _huggingface_model_registry[config.model_id] = config

    if config.model_name in MODEL_NAME_TO_MODEL:
        raise ValueError(f"A HELM model is already registered for model name: {config.model_name}")
    model = Model(
        group="huggingface",
        name=config.model_id,
        tags=[TEXT_MODEL_TAG, FULL_FUNCTIONALITY_TEXT_MODEL_TAG, LOCAL_HUGGINGFACE_MODEL_TAG],
    )
    MODEL_NAME_TO_MODEL[config.model_id] = model
    ALL_MODELS.append(model)
    hlog(f"Registered Hugging Face model: {model} config: {config}")
    return config


def get_huggingface_model_config(model_name: str) -> Optional[HuggingFaceModelConfig]:
    """Returns a HuggingFaceModelConfig for the model_id."""
    return _huggingface_model_registry.get(model_name)

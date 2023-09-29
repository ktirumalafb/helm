from copy import deepcopy
import torch
from dataclasses import asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Any, Dict, List

from helm.common.cache import Cache, CacheConfig
from helm.common.hierarchical_logger import htrack_block, hlog
from helm.common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT, Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)
from .client import Client, wrap_request_time, truncate_sequence, cleanup_tokens
from .huggingface_tokenizer import HuggingFaceTokenizers
from helm.proxy.clients.huggingface_model_registry import (
    get_huggingface_model_config,
    HuggingFaceModelConfig,
    HuggingFaceHubModelConfig,
    HuggingFaceLocalModelConfig,
)
from threading import Lock
import os
import json
from peft import PeftModel
from torch import nn
import torch


# Map of HELM model name to Hugging Face Hub model name where they differ.
_KNOWN_MODEL_ALIASES: Dict[str, str] = {
    "huggingface/gpt2": "gpt2",
    "huggingface/starcoder": "bigcode/starcoder",
}


class HuggingFaceServer:
    def __init__(self, model_config: HuggingFaceModelConfig):
        if torch.cuda.is_available():
            hlog("CUDA is available, initializing with a GPU...")
            self.device: str = "cuda:0"
        else:
            self.device = "cpu"
        model_kwargs = {}
        # If the HuggingFace model is stored locally, it will have a path defined and we should load it from there.
        # Otherwise, download it from the HuggingFace hub by passing in its identifier.
        if isinstance(model_config, HuggingFaceLocalModelConfig):
            model_name = model_config.path
        elif isinstance(model_config, HuggingFaceHubModelConfig):
            model_name = model_config.model_id
            if model_config.revision:
                model_kwargs["revision"] = model_config.revision
        else:
            raise Exception(f"Unknown type of model_config: {model_config}")

        if not model_config.load_adapters:
            with htrack_block(f"Loading Hugging Face model + tokenizer for config {model_config}"):
                # WARNING this may fail if your GPU does not have enough memory
                self.model, self.tokenizer = self.load_non_peft_model(model_name, model_config, model_kwargs)

                if model_config.load_layer_norm:
                    # Load normally at first
                    hlog(f"Loading with different layer norm parameters given by path -> {model_config.layer_norm_weights_path}")
                    self.model = self.merge_LN(self.model, model_config.layer_norm_weights_path)
        else:
            with htrack_block(f"Loading Hugging Face model + tokenizer in PEFT mode for config {model_config}"):
                self.model, self.tokenizer = self.load_peft_model_with_adapters(model_name, layer_dropping=model_config.layer_dropping, layers_to_drop=model_config.layers_to_drop)

    def merge_LN(self, model, path_to_params):
        #turn off all the gradidents
        for p in model.parameters():
            p.requires_grad=False 

        #read the LN weights
        ln_params = {}
        with open(path_to_params, 'r') as file:
            ln_params = json.load(file)

        #original model and tuned model have different name for the modules. Differ by '_orig_mod.'
        #double loop is not optimal, but works
        for (n, p) in model.named_parameters():
            for key in ln_params.keys():
                new_key = key.replace('_orig_mod.', '', 1) # strip the '_orig_mod.'
                if new_key == n:
                    p.copy_(torch.tensor(ln_params[key]).to("cuda")) #copy the tensor
        return model

    def load_non_peft_model(self, model_name: str, model_config, model_kwargs):
        device_map = "auto"
        if (model_config.num_bits == 4) or (model_config.num_bits == 8):
            if model_config.num_bits == 4:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    quantization_config=bnb_config,
                    device_map=device_map,
                    trust_remote_code=True
                )

            elif model_config.num_bits == 8:

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True,
                    torch_dtype=torch.bfloat16,
                    device_map=device_map,
                    trust_remote_code=True,
                    **model_kwargs
                )


        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True, **model_kwargs
            ).to(self.device)
            
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **model_kwargs)
    
        # model = torch.compile(model)
        
        assert model is not None
        assert tokenizer is not None
        return model, tokenizer

    def load_peft_model_with_adapters(self, MODEL_PATH: str, layer_dropping: bool = False, layers_to_drop: str = None):
        # Set device
        device_map = "auto"

        config_path = os.path.join(MODEL_PATH, "config.json")
        if not os.path.exists(config_path):
            print(f"Config path {config_path} does not exist... exiting...")

        config = None
        with open(config_path) as f:
            config = json.load(f)

        if 'quantization_config' in config:
            q_config = config["quantization_config"]
            # Read the bits and bytes config from the `config.json` to make sure we use the same one
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=q_config["load_in_8bit"],
                load_in_4bit=q_config["load_in_4bit"],
                llm_int8_threshold=q_config["llm_int8_threshold"],
                llm_int8_skip_modules=q_config["llm_int8_skip_modules"],
                llm_int8_enable_fp32_cpu_offload=q_config["llm_int8_enable_fp32_cpu_offload"],
                llm_int8_has_fp16_weight=q_config["llm_int8_has_fp16_weight"],
                bnb_4bit_compute_dtype=q_config["bnb_4bit_compute_dtype"],
                bnb_4bit_quant_type=q_config["bnb_4bit_quant_type"],
                bnb_4bit_use_double_quant=q_config["bnb_4bit_use_double_quant"],
            )


        # Set path to the base model with which we will merge our trained LoRA weights. Can be local.
        base_model_path = config["_name_or_path"]

        # Load base model
        model_base = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            quantization_config=bnb_config,
            device_map=device_map
        )

        TOKENIZER_FILE = os.path.join(base_model_path, "tokenizer.json")

        # Get tokenizer from the same place.
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, local_files_only=False, use_fast=True#, **tokenizer_kwargs
        )

        # Tokenizer config consistent with Platypus.
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id
        print("pre-trained model's BOS EOS and PAD token id:",bos,eos,pad," => It should be 1 2 None")

        tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
        tokenizer.padding_side = "right"

        # If we decide to drop layers, do it before loading LORA adapters
        if layer_dropping:
            # Cutting a few layers out.
            def get_layers_to_drop(ix, start, end):
                    start = start
                    end = end
                    step = (end - start) // (ix)
                    layer_ixs = torch.arange(start, end, step)
                    return layer_ixs

            ix,start_idx,end_idx = layers_to_drop.strip().split(",")
            ix,start_idx,end_idx = int(ix),int(start_idx),int(end_idx)
            blocks_to_remove = get_layers_to_drop(ix, start_idx, end_idx)
            print(f"We are removing: {blocks_to_remove} layers")

            model_base.model.layers = nn.ModuleList([block for idx, block in enumerate(model_base.model.layers) if idx not in blocks_to_remove])

        # Load & merge
        tuned_model = PeftModel.from_pretrained(
            model_base,
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
        )

        # tuned_model = torch.compile(tuned_model)

        return tuned_model, tokenizer

    def serve_request(self, raw_request: Dict[str, Any]):
        encoded_input = self.tokenizer(raw_request["prompt"], return_tensors="pt", return_token_type_ids=False).to(
            self.device
        )
        raw_request = deepcopy(raw_request)
        raw_request["do_sample"] = True
        raw_request["return_dict_in_generate"] = True
        raw_request["output_scores"] = True
        top_k_per_token: int = raw_request["top_k_per_token"]
        del raw_request["top_k_per_token"]
        if len(raw_request["stop_sequences"]) > 0:
            stop_sequence_ids = self.tokenizer(
                raw_request["stop_sequences"], return_token_type_ids=False, add_special_tokens=False
            )
            assert len(stop_sequence_ids.input_ids) == 1, "Total number of stop words should be 1."
            assert len(stop_sequence_ids.input_ids[0]) == 1, "Total number of tokens in each stop word should be 1."
            del raw_request["stop_sequences"]
            raw_request["eos_token_id"] = stop_sequence_ids.input_ids[0][0]

        # Strip out irrelevant parameters
        relevant_raw_request = {
            key: raw_request[key]
            for key in raw_request
            if key not in ["engine", "prompt", "echo_prompt", "stop_sequences"]
        }

        # Use HuggingFace's `generate` method.
        output = self.model.generate(**encoded_input, **relevant_raw_request)
        sequences = output.sequences
        scores = output.scores

        # Compute logprobs for each completed sequence.
        all_logprobs_of_chosen_tokens = []
        all_top_logprobs_dicts = []
        for completion_id in range(raw_request["num_return_sequences"]):
            logprobs_of_chosen_tokens = []
            top_logprobs_dicts = []
            for i in range(len(sequences[completion_id]) - len(encoded_input.input_ids[0])):
                logprobs = torch.nn.functional.log_softmax(scores[i][completion_id], dim=0)

                # Get top tokens in terms of log probability.
                topk_logprobs = torch.topk(logprobs, k=top_k_per_token)
                top_logprobs_dicts.append(
                    {
                        self.tokenizer.convert_ids_to_tokens(k.item()): v.item()
                        for (k, v) in zip(topk_logprobs.indices, topk_logprobs.values)
                    }
                )

                # Get log probability of chosen token.
                j = i + len(encoded_input.input_ids[0])
                logprobs_of_chosen_tokens.append(logprobs[sequences[completion_id][j]].item())
            all_logprobs_of_chosen_tokens.append(logprobs_of_chosen_tokens)
            all_top_logprobs_dicts.append(top_logprobs_dicts)

        # Remove prompt from the start of each sequence if echo_prompt is False.
        if not raw_request["echo_prompt"]:
            sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]

        all_tokens = [[self.tokenizer.decode(token) for token in sequence_tokens] for sequence_tokens in sequences]
        all_decoded_text = self.tokenizer.batch_decode(sequences)

        completions = []
        for decoded_text, tokens, logprobs_of_chosen_tokens, top_logprobs_dicts in zip(
            all_decoded_text, all_tokens, all_logprobs_of_chosen_tokens, all_top_logprobs_dicts
        ):
            completions.append(
                {
                    "text": decoded_text,
                    "tokens": tokens,
                    "logprobs": logprobs_of_chosen_tokens,
                    "top_logprobs_dicts": top_logprobs_dicts,
                }
            )

        return {"completions": completions, "input_length": len(encoded_input.input_ids[0])}


_servers_lock: Lock = Lock()
_servers: Dict[str, HuggingFaceServer] = {}


def _get_singleton_server(model_config: HuggingFaceModelConfig) -> HuggingFaceServer:
    """Lookup or create a new HuggingFaceServer that will be shared among all threads.

    When --num-threads > 1, multiple threads will attempt to instantiate
    `HuggingFaceServer`s simultaneously. Since we have limited GPU memory, we want to
    just share a single copy of each model we are using. So, this function uses a lock
    to make sure that for each model, only one thread creates a HuggingFaceServer.
    The other threads can share that same server in the global _servers dictionary."""
    global _servers_lock
    global _servers
    with _servers_lock:
        if model_config.model_id not in _servers:
            _servers[model_config.model_id] = HuggingFaceServer(model_config)
    return _servers[model_config.model_id]


class HuggingFaceClient(Client):
    def __init__(self, cache_config: CacheConfig):
        self.cache = Cache(cache_config)
        self.model_server_instances: Dict[str, HuggingFaceServer] = {}

    def get_model_server_instance(self, model: str) -> HuggingFaceServer:
        model_config = get_huggingface_model_config(model)
        # Special-case some models in so that users don't have to enable them with --enable-huggingface-models
        if not model_config:
            if model in _KNOWN_MODEL_ALIASES:
                model_config = HuggingFaceHubModelConfig.from_string(_KNOWN_MODEL_ALIASES[model])
            else:
                model_config = HuggingFaceHubModelConfig.from_string(model)
        return _get_singleton_server(model_config)

    def make_request(self, request: Request) -> RequestResult:
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        # Only a single stop sequence is supported as we can only pass in a single value for `eos_token_id`
        if len(request.stop_sequences) > 1:
            raise ValueError("More than one stop sequence is not supported.")

        raw_request = {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "temperature": 1e-7 if request.temperature == 0 else request.temperature,
            "num_return_sequences": request.num_completions,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
            "echo_prompt": request.echo_prompt,
            "top_k_per_token": request.top_k_per_token,
            "stop_sequences": request.stop_sequences,
        }

        # Get cached model server instance if possible (to save on model and tokenizer
        # loading times).
        model_server_instance: HuggingFaceServer = self.get_model_server_instance(request.model)

        try:

            def do_it():
                return model_server_instance.serve_request(raw_request)

            cache_key = Client.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:  # Do something if error is encountered.
            error: str = f"HuggingFace error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions = []
        for raw_completion in response["completions"]:
            sequence_logprob: float = 0
            tokens: List[Token] = []

            if request.echo_prompt:
                # Add prompt to list of generated tokens.
                generated_tokens = raw_completion["tokens"][response["input_length"] :]
                for token_text in raw_completion["tokens"][: response["input_length"]]:
                    tokens.append(Token(text=token_text, logprob=0.0, top_logprobs={}))
            else:
                generated_tokens = raw_completion["tokens"]

            # Compute logprob for the entire sequence.
            for token_text, logprob, top_logprobs_dict in zip(
                generated_tokens, raw_completion["logprobs"], raw_completion["top_logprobs_dicts"]
            ):
                tokens.append(Token(text=token_text, logprob=logprob, top_logprobs=top_logprobs_dict))
                sequence_logprob += logprob

            completion = Sequence(text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens)
            completion = truncate_sequence(completion, request)
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        tokenizer = HuggingFaceTokenizers.get_tokenizer(request.tokenizer)
        cache_key = asdict(request)

        try:

            def do_it():
                if request.encode:
                    if request.truncation:
                        tokens = tokenizer.encode(
                            request.text,
                            truncation=request.truncation,
                            max_length=request.max_length,
                            add_special_tokens=False,
                        )
                    else:
                        tokens = tokenizer.encode(request.text, add_special_tokens=False)
                else:
                    if "gpt" in request.tokenizer or request.tokenizer in [
                        "bigscience/bloom",
                        "Writer/palmyra-base",
                        "facebook/opt-66b",
                    ]:
                        # These models already handle the "▁" character correctly with the
                        # convert_tokens_to_string method. We prefer to use this method instead
                        # of the hacky cleanup_tokens method below as it might handle cases
                        # we haven't thought of in cleanup_tokens.
                        tokens = [
                            tokenizer.convert_tokens_to_string([token]) for token in tokenizer.tokenize(request.text)
                        ]
                    else:
                        # Tokenizes the text and returns the tokens as a list of strings,
                        # not a list of token objects (otherwise "Hello world" would be"
                        # ["Hello", "▁world"] and not ["Hello", " world"])
                        # We could do this with a simple replace like this:
                        # tokens = [tokenizer.convert_tokens_to_string([i]) for i in tokenizer.tokenize(request.text)]
                        # But this replaces all the "▁" characters by "", which is not what we want.
                        # This would be problematic as tokenize(" Hello", encode=False) would return ["Hello"]
                        # Just like tokenize("Hello", encode=False) would return ["Hello"].
                        tokens = tokenizer.tokenize(request.text)
                        tokens = cleanup_tokens(tokens, request.tokenizer)
                return {"tokens": tokens}

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"HuggingFace error: {e}"
            return TokenizationRequestResult(success=False, cached=False, error=error, text="", tokens=[])

        return TokenizationRequestResult(
            success=True,
            cached=cached,
            text=request.text,
            tokens=[TokenizationToken(value) for value in result["tokens"]],
            request_time=result["request_time"],
        )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        tokenizer = HuggingFaceTokenizers.get_tokenizer(request.tokenizer)
        cache_key = asdict(request)

        try:

            def do_it():
                return {
                    "text": tokenizer.decode(
                        request.tokens, clean_up_tokenization_spaces=request.clean_up_tokenization_spaces
                    )
                }

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"HuggingFace error: {e}"
            return DecodeRequestResult(success=False, cached=False, error=error, text="")

        return DecodeRequestResult(
            success=True, cached=cached, text=result["text"], request_time=result["request_time"]
        )

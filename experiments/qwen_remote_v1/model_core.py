"""NF4 LoRA classifier over five Qwen next-token logits."""
from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_tokenizer(config):
    m = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"],
                                               local_files_only=True, padding_side="right")
    answer = []
    for text in config["input"]["answer_strings"]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids, skip_special_tokens=True) != text:
            raise RuntimeError(f"Rating {text} is not one round-trip token")
        answer.append(ids[0])
    if len(set(answer)) != 5:
        raise RuntimeError("Five rating IDs are not distinct")
    return tokenizer, answer


def load_model(config, trainable, seed, adapter_path=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = config["model"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required by the locked run")
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("Locked BF16 compute requires a BF16-capable GPU")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        m["id"], revision=m["revision"], local_files_only=True,
        quantization_config=quant, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa",
        low_cpu_mem_usage=True)
    model.config.use_cache = False
    # Apply the same frozen-base dtype preparation in train and inference
    # loads. Otherwise FP32 RMSNorms in a LoRA run are compared against BF16
    # RMSNorms after adapter reload, creating a false prediction mismatch.
    prep_kwargs = {"use_gradient_checkpointing": trainable}
    if trainable:
        prep_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    model = prepare_model_for_kbit_training(model, **prep_kwargs)
    model.get_input_embeddings().to(dtype=torch.bfloat16)
    output = model.get_output_embeddings()
    if output is not None:
        output.to(dtype=torch.bfloat16)
    if trainable:
        l = config["training"]
        if adapter_path is None:
            model = get_peft_model(model, LoraConfig(
                r=l["lora_r"], lora_alpha=l["lora_alpha"],
                lora_dropout=l["lora_dropout"], target_modules=l["lora_target_modules"],
                bias=l["bias"], task_type="CAUSAL_LM"))
        else:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.train()
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        if not trainable_names or any("lora_" not in n for n in trainable_names):
            raise RuntimeError("Unexpected trainable parameters; expected LoRA only")
    else:
        model.eval().requires_grad_(False)
        trainable_names = []
    # Transformers 4.51 may produce FP32 hidden states after the final RMSNorm
    # even when the frozen LM head is BF16. Cast at the frozen projection
    # boundary; this operation is differentiable, so LoRA gradients are intact.
    output = model.get_output_embeddings()
    if output is not None:
        def _match_lm_head_dtype(module, args):
            return (args[0].to(dtype=module.weight.dtype), *args[1:])
        output.register_forward_pre_hook(_match_lm_head_dtype)
    return model, trainable_names


def five_logits(model, input_ids, answer_token_ids, train):
    attention_mask = torch.ones_like(input_ids)
    if train:
        output = model(input_ids=input_ids, attention_mask=attention_mask,
                       use_cache=False, logits_to_keep=1)
        logits = output.logits[:, -1, :]
        return logits[:, answer_token_ids].float()
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask,
                       use_cache=False, logits_to_keep=1)
        return output.logits[:, -1, answer_token_ids].float()


def predict_probability(model, ids, answer_ids):
    logits = five_logits(model, torch.tensor(ids, device="cuda", dtype=torch.long).reshape(1, -1),
                         answer_ids, train=False)
    return torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()


def save_adapter(model, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)

from dataclasses import dataclass, field
import pathlib
from typing import Optional, List


import transformers
from point_navigator_3dllm.train.point_navigator_3dllm_trainer import point_navigator_3dllmTrainer

from point_navigator_3dllm import conversation as conversation_lib
from point_navigator_3dllm.model import *
from point_navigator_3dllm.data import make_object_point_data_module


from point_navigator_3dllm.utils import build_logger
from peft import get_peft_model, LoraConfig, TaskType
import torch

IGNORE_INDEX = -100

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"

dtype_mapping = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    version: Optional[str] = field(default="v1")


@dataclass
class DataArguments:
    data_path: str = field(
        default="ScanNet", metadata={"help": "Path to the training data."}
    )
    img_data_path: str = field(
        default=None, metadata={"help": "Path to the image data."}
    )
    anno_path: str = field(
        default=None,
        metadata={
            "help": "Path to the utterance data. If None, will use referit3d by defautl."
        },
    )
    val_anno_path: str = field(
        default=None,
        metadata={
            "help": "Path to the utterance data. If None, will use referit3d by defautl."
        },
    )
    use_color: bool = field(default=False, metadata={"help": "Whether to use color."})
    data_debug_num: int = field(
        default=0,
        metadata={
            "help": "Number of data to use in debug mode. If larger than 0, use debug mode, else use the whole data"
        },
    )
    split_train_val: bool = field(
        default=False, metadata={"help": "Whether to split train and val."}
    )
    split_ratio: float = field(
        default=0.9, metadata={"help": "Ratio of train and val."}
    )
    pointnum: int = field(default=8192, metadata={"help": "Number of points."})
    conversation_types: List[str] = field(
        default_factory=lambda: ["simple_description"],
        metadata={"help": "Conversation types to use."},
    )
    is_multimodal: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):

    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    model_debug: bool = field(
        default=False, metadata={"help": "Whether to use small model."}
    )
    fix_llm: bool = field(default=True, metadata={"help": "Whether to fix the LLM."})
    fix_pointnet: bool = field(
        default=True, metadata={"help": "Whether to fix the PointNet."}
    )
    ddp_find_unused_parameters: bool = field(
        default=False, metadata={"help": "ddp_find_unused_parameters"}
    )
    use_lora: bool = field(default=False, metadata={"help": "Use lora in LLM."})

    torch_dtype: str = field(
        default="bfloat16", metadata={"help": "choices=float32, float16, bfloat16"}
    )

    remove_unused_columns: bool = field(default=False)
    force_fsdp: bool = field(default=False)

    tune_mm_mlp_adapter: bool = field(default=True)
    stage_2: bool = field(default=False)
    pretrained_mm_mlp_adapter: Optional[str] = field(default=None)
    detatch_point_token: bool = field(default=False)

    point_backbone_ckpt: str = field(default=None)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):

    torch.cuda.synchronize()
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                cpu_state_dict[key] = value.cpu()
            else:
                cpu_state_dict[key] = value
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.log_level = "info"

    logger = build_logger(__name__, training_args.output_dir + "/train.log")

    if training_args.model_debug:

        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=dtype_mapping[training_args.torch_dtype],
        )
        model = point_navigator_3dllmLlamaForCausalLM._from_config(config)
    else:
        model = point_navigator_3dllmLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=dtype_mapping[training_args.torch_dtype],
        )

    model.config.use_cache = False

    if training_args.fix_llm:

        logger.info("LLM is fixed. Fix_llm flag is set to True")

        model.requires_grad_(False)
        model.get_model().fix_llm = True
        model.get_model().point_proj.requires_grad_(True)
        model.get_model().point_backbone.requires_grad_(True)
    else:
        model.get_model().fix_llm = False
        logger.warning("LLM is trainable. Fix_llm flag is set to False")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0" or "v0" in model_args.model_name_or_path:
        raise ValueError("v0 is deprecated.")
    else:
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            "vicuna_v1_1"
        ]

    if not training_args.fix_pointnet:

        logger.info(
            "Point backbone is trainable. Fix_pointnet flag is set to False, pointnet grad will be recorded."
        )
        model.get_model().fix_pointnet = False
    else:
        logger.info(
            "Point backbone is fixed. Fix_pointnet flag is set to True, pointnet grad will not be recorded."
        )
        model.get_model().fix_pointnet = True
        if not training_args.stage_2:
            logger.info("Set requires_grad of point backbone to False")
            model.get_model().point_backbone.requires_grad_(False)

    if training_args.tune_mm_mlp_adapter:

        logger.info("Point projection layer is trainable.")
    else:
        model.get_model().point_proj.requires_grad_(False)
        logger.info("Point prejcetion layer is fixed.")

    if not training_args.stage_2 and training_args.point_backbone_ckpt is not None:

        print(f"Default point_backbone_ckpt is {training_args.point_backbone_ckpt}.")
        model.get_model().load_point_backbone_checkpoint(
            training_args.point_backbone_ckpt
        )
        model.initialize_tokenizer_point_backbone_config(
            tokenizer=tokenizer,
            device=training_args.device,
            fix_llm=training_args.fix_llm,
        )
    else:

        model.initialize_tokenizer_point_backbone_config_wo_embedding(
            tokenizer=tokenizer
        )

    point_backbone_config = model.get_model().point_backbone_config

    data_args.point_token_len = point_backbone_config["point_token_len"]
    data_args.mm_use_point_start_end = point_backbone_config["mm_use_point_start_end"]
    data_args.point_backbone_config = point_backbone_config

    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print(
                    "[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}".format(
                        len(params_no_grad), params_no_grad
                    )
                )
            else:
                print(
                    "[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)".format(
                        len(params_no_grad), ", ".join(params_no_grad[:10])
                    )
                )
            print(
                "[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental."
            )
            print(
                "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining"
            )

            from torch.distributed.fsdp.fully_sharded_data_parallel import (
                FullyShardedDataParallel as FSDP,
            )

            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop("use_orig_params", True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)

                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    data_module = make_object_point_data_module(
        tokenizer=tokenizer, data_args=data_args
    )

    if training_args.use_lora and not training_args.fix_llm:
        model.enable_input_require_grads()
        print(model.named_parameters())
        target_modules = []
        for name, param in model.named_parameters():

            import re

            if "self_attn" in name:

                match = re.match(
                    r"model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
                    name,
                )
                if match:

                    target_modules.append(name.replace(".weight", ""))
        print(target_modules)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            target_modules=target_modules,
            lora_alpha=16,
            lora_dropout=0.05,
        )
        import os

        model = get_peft_model(model, lora_config)
        for name, param in model.named_parameters():
            if training_args.fix_pointnet:
                if "lora" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                if "lora" in name or "point" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        if os.environ.get("DEBUG_flag") == "debug_lora":
            import ipdb

            ipdb.set_trace()

    model = model.to(dtype_mapping[training_args.torch_dtype])
    trainer = point_navigator_3dllmTrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    trainer.train()
    if training_args.use_lora:
        model = model.merge_and_unload()
        trainer = point_navigator_3dllmTrainer(
            model=model, tokenizer=tokenizer, args=training_args, **data_module
        )

    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

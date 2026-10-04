import os
import torch
import torch.nn as nn

from transformers import Trainer
from typing import Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import nltk

import wandb
import ipdb

from point_navigator_3dllm.train.llama_flash_attn_monkey_patch import original_forward
from torch.cuda.amp import autocast
from point_navigator_3dllm.model.utils import KeywordsStoppingCriteria
import copy
from point_navigator_3dllm.conversation import conv_templates, SeparatorStyle


def unwrap_model(model: nn.Module) -> nn.Module:

    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class point_navigator_3dllmTrainer(Trainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):

            _state_dict = state_dict
            if _state_dict is None:

                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()

            weight_to_save = {}
            keys_to_match = ["point_proj", "embed_tokens", "embed_in"]
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v

            current_folder = output_dir.split("/")[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "point_proj")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save,
                    os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                )
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f"point_proj.bin"))

        super(point_navigator_3dllmTrainer, self)._save(output_dir, state_dict)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = self.model
        model.eval()
        model = model.to("cpu")
        eval_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        model_test = copy.deepcopy(model)
        model_test = model_test.merge_and_unload()
        model_test = model_test.to("cuda")
        conv_mode = "vicuna_v1_1"
        conv = conv_templates[conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]

        with torch.no_grad():
            for inputs in eval_dataloader:

                inputs = {key: value.to(device) for key, value in inputs.items()}

                input_ids = inputs["input_ids"]
                point_clouds = inputs["point_clouds"]
                labels = inputs["labels"]
                mask_for_input_ids = labels == -100
                filter_input_ids = input_ids[mask_for_input_ids]
                filter_labels = labels[~mask_for_input_ids]
                filter_input_ids = filter_input_ids.unsqueeze(0)

                stopping_criteria = KeywordsStoppingCriteria(
                    keywords, self.tokenizer, filter_input_ids
                )

                with autocast():
                    output_ids = model_test.generate(
                        filter_input_ids,
                        point_clouds=point_clouds,
                        do_sample=True,
                        temperature=0.2,
                        top_k=50,
                        max_new_tokens=32,
                        top_p=0.95,
                        stopping_criteria=[stopping_criteria],
                    )

                output_ids = output_ids[0, filter_input_ids.shape[1] :]
                num_batches += 1

                all_predictions.append(output_ids.cpu().numpy())
                all_labels.append(filter_labels.cpu().numpy())

        smooth = SmoothingFunction().method4

        bleu_scores = []
        meteor_scores = []
        exact_match_scores = []

        for pred, label in zip(all_predictions, all_labels):

            pred_text = self.tokenizer.decode(pred, skip_special_tokens=True)
            label_text = self.tokenizer.decode(label, skip_special_tokens=True)

            bleu_score = sentence_bleu(
                [label_text.split()], pred_text.split(), smoothing_function=smooth
            )
            bleu_scores.append(bleu_score * 100)

            meteor_score_value = meteor_score([label_text.split()], pred_text.split())
            meteor_scores.append(meteor_score_value)

            exact_match_score = 1.0 if pred_text == label_text else 0.0
            exact_match_scores.append(exact_match_score)

        avg_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        avg_meteor_score = (
            sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
        )
        avg_exact_match = (
            sum(exact_match_scores) / len(exact_match_scores)
            if exact_match_scores
            else 0
        )

        metrics = {
            "avg_bleu_score": avg_bleu_score,
            "avg_meteor_score": avg_meteor_score,
            "avg_exact_match": avg_exact_match,
        }

        self.log(metrics)
        del model_test
        torch.cuda.empty_cache()
        model = model.to("cuda")

    def compute_bleu(self, ground_truth, generated_text):

        bleu_1 = sentence_bleu(
            [ground_truth.split()],
            generated_text.split(),
            weights=(1, 0, 0, 0),
            smoothing_function=self.smoothing_function,
        )
        bleu_2 = sentence_bleu(
            [ground_truth.split()],
            generated_text.split(),
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=self.smoothing_function,
        )
        bleu_3 = sentence_bleu(
            [ground_truth.split()],
            generated_text.split(),
            weights=(0.33, 0.33, 0.33, 0),
            smoothing_function=self.smoothing_function,
        )
        bleu_4 = sentence_bleu(
            [ground_truth.split()],
            generated_text.split(),
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=self.smoothing_function,
        )
        return bleu_1, bleu_2, bleu_3, bleu_4

    def compute_rouge(self, ground_truth, generated_text):

        scores = self.rouge.get_scores(generated_text, ground_truth)[0]
        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"],
        }

    def compute_meteor(self, ground_truth, generated_text):

        return meteor_score([ground_truth.split()], generated_text.split())

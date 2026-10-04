import os
import json
import torch
import random
from tqdm import tqdm
from pathlib import Path
from typing import Dict
import torch.nn as nn
from tools.common_utils import all_gather
from tools.parser import read_args, random_seed
from tasks.loaders import create_dataloaders
from tasks.feature_db import create_feature_db, create_object_feature_db

from tools.optims import dist_models, save_checkpoint
from tools.trie import Trie
import wandb
from datetime import datetime
from peft import get_peft_model, LoraConfig, TaskType
import copy
from torch.nn.parallel import DistributedDataParallel as DDP


class Metrics(object):
    def __init__(self):
        self.num = 0
        self.total = 0

    def accumulate(self, x):
        self.num += 1
        self.total += x

    @property
    def average(self):
        if self.num == 0:
            return 0
        return self.total / self.num


def train_one_epoch(
    args,
    global_cfg,
    model,
    optimizer,
    lr_scheduler,
    criterion,
    dataloaders,
    agents,
    epoch,
    logger,
    stage="multi",
):

    model.train()
    entropy_metric = Metrics()
    loss_metric = Metrics()
    instr_pred_metric = Metrics()

    num_batches_per_epoch = dataloaders.num_batches
    total_training_steps = num_batches_per_epoch * args.num_epochs

    pbar = tqdm(
        range(dataloaders.num_batches),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )

    dataset_cfg = global_cfg.Pretrain if stage == "pretrain" else global_cfg.Multi
    loss_stats = {k: Metrics() for k in dataset_cfg.SOURCE}

    for step, (name, batch) in enumerate(dataloaders):
        loss_coef = dataset_cfg.LOSS_COEF.get(name, 1.0)

        dataset = dataloaders.loader.get_dataset(name)
        agent = agents.get(name)
        loss = agent.train(
            name,
            batch,
            args,
            global_cfg,
            model=model,
            criterion=criterion,
            dataset=dataset,
            step=step,
            entropy_metric=entropy_metric,
            instr_pred_metric=instr_pred_metric,
        )
        loss_metric.accumulate(loss.item())
        loss_stats[name].accumulate(loss.item())

        if (step + 1) % args.gradient_accumulation_step == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 40.0)
            optimizer.step()
            optimizer.zero_grad()

        lr_scheduler.step()

        if args.rank == 0:
            verbose_dict = dict(
                step=step,
                name=name,
                loss=loss_metric.average,
                entropy=entropy_metric.average,
                instr_pred_metric=instr_pred_metric.average,
                lr=lr_scheduler.get_last_lr()[0],
            )

            for k in dataset_cfg.SOURCE:
                verbose_dict[k] = loss_stats[k].average
            pbar.set_postfix(verbose_dict)
            pbar.update()

            performance_metrics_per_steps = []
            performance_metrics_per_steps.append(
                {
                    "loss_per_steps": loss_metric.average,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
            )
            for iter in performance_metrics_per_steps:

                wandb.log(iter)

            if step < num_batches_per_epoch - 1 and step > num_batches_per_epoch - 8:
                performance_metrics_per_epoch = []
                performance_metrics_per_epoch.append(
                    {
                        "loss_average_per_epoch": loss_metric.average,
                    }
                )
                for task in dataset_cfg.SOURCE:
                    performance_metrics_per_epoch.append(
                        {task + "_loss": loss_stats[task].average}
                    )
                for iter in performance_metrics_per_epoch:

                    wandb.log(iter)

        if step == num_batches_per_epoch - 1:
            logger.info("***** train [{}] epoch *****".format(epoch))
            train_stat_str = "Loss: %.2f\n" % loss_metric.average
            train_stat_str += "Instr_pred: %.2f\n" % instr_pred_metric.average
            for task in dataset_cfg.SOURCE:
                train_stat_str += "%s: %.2f\n" % (task, loss_stats[task].average)
            logger.info(train_stat_str)
            break


@torch.no_grad()
def val_one_epoch(
    args,
    global_cfg,
    model,
    optimizer,
    criterion,
    dataloaders,
    agents,
    epoch,
    logger,
) -> Dict[str, Dict[str, float]]:

    model_test = model
    model_test.eval()
    entropy_metric = Metrics()

    loss_str = "\n[Eval] {} epoch {}\n".format(args.validation_split, epoch)
    task_results = {}
    for name, loader in dataloaders.items():
        logger.info(
            "***** validate {} split on {} task *****".format(
                args.validation_split, name
            )
        )
        dataset = dataloaders[name].get_dataset()
        agent = agents[name]
        preds = agent.validate(
            name, args, global_cfg, model_test, loader, entropy_metric=entropy_metric
        )

        all_preds = all_gather(preds)
        all_preds = merge_dist_results(all_preds)

        if args.rank == 0 and not args.validation_split.startswith("test"):

            performance_metrics_val = []
            try:
                score_summary, item_metrics = dataset.eval_metrics(
                    all_preds, logger=logger, name=name
                )

                task_results[name] = score_summary
                loss_str += "\n [Eval] dataset=[{}] \n".format(name)

                for metric, val in score_summary.items():
                    if metric == "sr":
                        loss_str += "\n[Eval] ||| %s: %.2f" % (metric, val)
                    else:
                        loss_str += ", %s: %.2f" % (metric, val)
                    performance_metrics_val.append({name + str(metric): val})
                for iter in performance_metrics_val:

                    wandb.log(iter)

            except Exception as e:
                dataset.save_json(
                    all_preds,
                    os.path.join(
                        args.output_dir, f"{name}_{args.validation_split}.json"
                    ),
                    item_metrics=item_metrics if args.save_detail_results else None,
                )

        if args.rank == 0 and args.save_pred_results:
            dataset.save_json(
                all_preds,
                os.path.join(args.output_dir, f"{name}_{args.validation_split}.json"),
                item_metrics=item_metrics if args.save_detail_results else None,
            )

    logger.info(loss_str)

    return task_results


def merge_dist_results(results):
    outs = []
    for res in results:
        outs.extend(res)
    return outs


def calc_overall_score(results, cfg):
    score = 0.0
    for task in results:
        if task not in cfg.Multi.SOURCE:
            continue
        if task == "R2R":
            score += results[task]["spl"] / 60
        elif task == "REVERIE":
            score += results[task]["spl"] / 36.63
        elif task == "CVDN":
            pass
        elif task == "SOON":
            score += results[task]["spl"] / 26.58
        elif task == "SQA3D":
            pass
        elif task == "ScanQA":
            pass
        else:
            raise NotImplementedError(
                f"The method for calculating the score of {task} is not Implemented."
            )

    return score


def main():

    args, global_cfg, logger, device_id = read_args()
    if args.rank == 0:
        config_dict_wandb = {**vars(args), "global_cfg": global_cfg}
        wandb.init(
            entity="web-videos-for-robotics",
            project="3d_nav_VLN",
            id=args.script_path + datetime.now().strftime("%Y-%m-%d"),
            config=config_dict_wandb,
        )

    random_seed(args.seed + args.rank)

    feat_db = create_feature_db(
        global_cfg.Feature.feature_database, global_cfg.Feature.image_feat_size, args
    )
    obj_feat_db = create_object_feature_db(
        global_cfg.Feature.object_database, global_cfg.Feature.obj_feat_size, args
    )

    if "qwen" in args.pretrained_model_name_or_path:

        from models.qwen_nav_model import NavModel
    else:
        from models.nav_model import NavModel

    if args.mode == "train":
        train_dataloaders, train_agents = create_dataloaders(
            args,
            global_cfg,
            logger,
            training=True,
            device=device_id,
            feat_db=feat_db,
            obj_feat_db=obj_feat_db,
            stage=args.stage,
        )

    val_dataloaders, val_agents = create_dataloaders(
        args,
        global_cfg,
        logger,
        training=False,
        device=device_id,
        feat_db=feat_db,
        obj_feat_db=obj_feat_db,
        stage="multi",
    )

    model = NavModel(args, logger, global_cfg.Model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, reduction="sum")
    model, optimizer, resume_from_epoch, lr_scheduler = dist_models(args, model, logger)

    if args.mode == "test":
        logger.info("**************************** Test ****************************")
        results = val_one_epoch(
            args,
            global_cfg,
            model,
            optimizer,
            criterion,
            val_dataloaders,
            val_agents,
            resume_from_epoch,
            logger,
        )

        score = calc_overall_score(results, global_cfg)
        logger.info(f"Current Score: {score}")
        print(
            "**************************** Current Score: {score} ****************************"
        )

    elif args.mode == "train":
        logger.info("**************************** Train ****************************")

        best_results, best_score = None, None
        history_scores = []
        for epoch in range(resume_from_epoch, args.num_epochs):

            train_one_epoch(
                args,
                global_cfg,
                model,
                optimizer,
                lr_scheduler,
                criterion,
                train_dataloaders,
                train_agents,
                epoch,
                logger,
                stage=args.stage,
            )

            if args.lora_finetune:
                model_test = copy.deepcopy(model)
                if isinstance(model_test, DDP):
                    print("Model is wrapped with DDP.")
                    model_test.module.lang_model = (
                        model_test.module.lang_model.merge_and_unload()
                    )
                else:
                    model_test.lang_model = model_test.lang_model.merge_and_unload()

                results = val_one_epoch(
                    args,
                    global_cfg,
                    model_test,
                    optimizer,
                    criterion,
                    val_dataloaders,
                    val_agents,
                    epoch,
                    logger,
                )
                del model_test

                if (
                    args.merge_lora_regular != 0
                    and epoch % args.merge_lora_regular == 0
                ):
                    if isinstance(model, DDP):

                        lang_model = model.module.lang_model
                        lang_model = lang_model.merge_and_unload()
                        lang_model = get_peft_model(lang_model, args.lora_config)

                        model.module.lang_model = lang_model
                        model = DDP(
                            model.module,
                            device_ids=[device_id],
                            find_unused_parameters=True,
                        )
                    else:
                        lang_model = model.lang_model
                        lang_model = lang_model.merge_and_unload()
                        lang_model = get_peft_model(lang_model, args.lora_config)
                        model.lang_model = lang_model

                    print("Model merge_lora_regular completed.")

                if args.rank == 0:
                    score = calc_overall_score(results, global_cfg)
                    history_scores.append(score)
                    should_save_checkpoint = False

                    if best_results is None or score > best_score:
                        best_results = results
                        best_score = score
                        should_save_checkpoint = args.max_saved_checkpoints > 0

                    wandb.log({"overall_score": score})
                    logger.info(f"Current Score: {score}")
                    logger.info(f"Best Score: {best_score}")

                    if (
                        args.stage == "multi"
                        and (epoch + 1) % args.save_ckpt_per_epochs == 0
                    ):

                        model_path = Path(args.output_dir) / f"epoch_{epoch}.pt"
                        model_save = copy.deepcopy(model)

                        if isinstance(model_save, DDP):
                            print("Model is wrapped with DDP.")
                            model_save = model_save.module
                            model_save.lang_model = (
                                model_save.lang_model.merge_and_unload()
                            )
                        else:
                            print("Model is not wrapped with DDP.")

                        save_checkpoint(
                            model_save,
                            model_path,
                            optimizer,
                            epoch,
                            lr_scheduler,
                            save_states=False,
                        )
                        del model_save

            else:

                results = val_one_epoch(
                    args,
                    global_cfg,
                    model,
                    optimizer,
                    criterion,
                    val_dataloaders,
                    val_agents,
                    epoch,
                    logger,
                )
                if args.rank == 0:
                    score = calc_overall_score(results, global_cfg)
                    history_scores.append(score)
                    should_save_checkpoint = True

                    if best_results is None or score > best_score:
                        best_results = results
                        best_score = score

                    wandb.log({"overall_score": score})
                    logger.info(f"Current Score: {score}")
                    logger.info(f"Best Score: {best_score}")

                    if args.stage == "multi":

                        if should_save_checkpoint:
                            if len(history_scores) > args.max_saved_checkpoints:
                                sorted_scores = sorted(
                                    enumerate(history_scores),
                                    key=lambda x: x[1],
                                    reverse=True,
                                )

                                remove_epoch = sorted_scores[
                                    args.max_saved_checkpoints
                                ][0]
                                remove_model_path = (
                                    Path(args.output_dir) / f"epoch_{remove_epoch}.pt"
                                )
                                if os.path.exists(remove_model_path):
                                    os.remove(remove_model_path)
                                    logger.info(
                                        f"Remove Checkpoint at Epoch {remove_epoch}..."
                                    )

                            model_path = Path(args.output_dir) / f"epoch_{epoch}.pt"
                            save_checkpoint(
                                model,
                                model_path,
                                optimizer,
                                epoch,
                                lr_scheduler,
                                save_states=False,
                            )

                    elif (
                        args.stage == "pretrain"
                        and (epoch + 1) % args.save_ckpt_per_epochs == 0
                    ):
                        model_path = Path(args.output_dir) / f"pretrain_{epoch}.pt"
                        save_checkpoint(
                            model,
                            model_path,
                            optimizer,
                            epoch,
                            lr_scheduler,
                            save_states=False,
                        )

                if args.save_latest_states:

                    model_path = Path(args.output_dir) / f"latest.pt"
                    save_checkpoint(
                        model,
                        model_path,
                        optimizer,
                        epoch,
                        lr_scheduler,
                        save_states=False,
                    )

            if args.rank == 0:
                logger.info(f"Best Results:")
                logger.info(best_results)

        if args.rank == 0:
            logger.info(f"Best Results:")
            logger.info(best_results)
            wandb.finish()


if __name__ == "__main__":
    main()

"""GRPO 训练循环：采样 K 个候选，计算奖励，做策略梯度更新，含验证与可选 TensorBoard 日志。"""

import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from echoreason.data.collate import BatchCollator
from echoreason.reward import compose_reward, pixel_reward, text_reward
from echoreason.training.optimizer import build_optimizer
from echoreason.training.scheduler import build_scheduler


class GRPOTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        cfg: Dict[str, Any] = None,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.cfg = cfg or {}
        self.device = device

        self.collate_fn = BatchCollator(stage="grpo", tokenizer=None, max_length=self.cfg.get("max_length", 1024))
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.get("train_batch_size", 2),
            shuffle=True,
            num_workers=self.cfg.get("num_workers", 4),
            collate_fn=self.collate_fn,
        )
        self.eval_loader = None
        if eval_dataset is not None:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.cfg.get("eval_batch_size", self.cfg.get("train_batch_size", 2)),
                shuffle=False,
                num_workers=self.cfg.get("num_workers", 4),
                collate_fn=self.collate_fn,
            )
        self._apply_freeze()
        self._apply_tuning_modules(self.cfg.get("tuning_modules", []))
        self.optimizer = build_optimizer(self.model, self.cfg)
        total_steps = len(self.train_loader) * max(1, self.cfg.get("epochs", 1))
        self.schedulers = build_scheduler(self.optimizer, self.cfg, total_steps)

        # 训练状态
        self.global_step = 0
        self.start_epoch = 0
        self.current_epoch = 0

        self.k = int(self.cfg.get("sampling", {}).get("k", 4))
        self.temperature = float(self.cfg.get("sampling", {}).get("temperature", 1.0))
        self.top_p = float(self.cfg.get("sampling", {}).get("top_p", 0.9))
        self.reward_weight = self.cfg.get("reward_weight", {"pixel": 1.0, "text": 1.0})

        # 日志
        log_dir = self.cfg.get("log_dir", None)
        self.log_every = int(self.cfg.get("log_every_steps", 0))
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None

    def _apply_freeze(self):
        freeze_cfg = self.cfg.get("freeze", {})
        freeze_vision = freeze_cfg.get("qwen_vision", False)
        freeze_language = freeze_cfg.get("qwen_language", False)

        if freeze_vision or freeze_language:
            for name, p in self.model.qwen.model.named_parameters():
                # LoRA 权重保持可训练，冻结基座参数
                if "lora_" in name:
                    continue
                if freeze_vision and name.startswith("visual"):
                    p.requires_grad = False
                elif freeze_language and not name.startswith("visual"):
                    p.requires_grad = False

    def _apply_tuning_modules(self, patterns) -> None:
        """
        可选局部解冻：patterns 为字符串或列表，若参数名包含任一 pattern，则解冻。
        """
        if patterns is None:
            return
        if isinstance(patterns, str):
            patterns = [p.strip() for p in patterns.split(",") if p.strip()]
        if not patterns:
            return
        for n, p in self.model.named_parameters():
            if any(pat in n for pat in patterns):
                p.requires_grad = True

    def _tokenize_prompts(self, prompts: List[str], max_length: int):
        enc = self.tokenizer(
            prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        return enc.input_ids, enc.attention_mask

    def train(self):
        self.model.train()
        epochs = self.cfg.get("epochs", 1)
        max_length = self.cfg.get("max_length", 1024)

        save_dir = self.cfg.get("output_dir", "./outputs/grpo")
        os.makedirs(save_dir, exist_ok=True)
        save_every = int(self.cfg.get("save_every_steps", 0))
        eval_every = int(self.cfg.get("eval_every_steps", 0))
        metric_for_best = self.cfg.get("metric_for_best", "val_reward")
        best_metric: Optional[float] = None

        for epoch in range(epochs):
            self.current_epoch = epoch
            pbar = tqdm(self.train_loader, desc=f"GRPO Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                pixel_values = batch["pixel_values"].to(self.device)
                masks = batch.get("masks")
                if masks is not None:
                    masks = (masks > 0).long().to(self.device)
                prompts = batch["prompts"]
                targets = batch["targets"]

                input_ids, attention_mask = self._tokenize_prompts(prompts, max_length)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                rewards_all = []
                logprobs_all = []

                for _ in range(self.k):
                    outputs = self.model(
                        {
                            "pixel_values": pixel_values,
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                        },
                        mode="infer",
                        generate_kwargs={
                            "temperature": self.temperature,
                            "top_p": self.top_p,
                            "do_sample": True,
                        },
                    )

                    seg_out = outputs["segmentation"]
                    slot_out = outputs["slot"]
                    pred_masks = seg_out["logits"]
                    pred_scores = seg_out.get("scores", None)
                    pred_texts = slot_out.get("texts", [""] * pixel_values.size(0))
                    logprobs = slot_out.get("logprobs", None)

                    # 奖励
                    if masks is not None:
                        pixel_r = pixel_reward(pred_masks, masks, pred_scores=pred_scores)
                    else:
                        pixel_r = torch.zeros(pred_masks.size(0), device=pred_masks.device)
                    text_r = text_reward(
                        pred_texts,
                        targets,
                        tokenizer=self.tokenizer,
                        embedding_layer=self.model.qwen.model.get_input_embeddings(),
                    )
                    reward = compose_reward(pixel_r, text_r, self.reward_weight)  # [B]

                    rewards_all.append(reward)
                    logprobs_all.append(logprobs if logprobs is not None else torch.zeros_like(reward))

                # [K,B] -> [B,K]
                rewards_stack = torch.stack(rewards_all, dim=0).transpose(0, 1)  # [B,K]
                logprobs_stack = torch.stack(logprobs_all, dim=0).transpose(0, 1)  # [B,K]

                # 组内均值基线
                baseline = rewards_stack.mean(dim=1, keepdim=True)  # [B,1]
                advantage = rewards_stack - baseline  # [B,K]

                # policy loss
                loss = -(advantage * logprobs_stack).mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                if self.schedulers["warmup"]:
                    self.schedulers["warmup"].step()
                if self.schedulers["main"]:
                    self.schedulers["main"].step()

                self.global_step += 1

                # TensorBoard 记录
                if self.writer and self.log_every > 0 and self.global_step % self.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("train/loss", float(loss.detach().item()), self.global_step)
                    self.writer.add_scalar("train/reward_mean", float(rewards_stack.mean().item()), self.global_step)
                    self.writer.add_scalar("train/reward_pixel", float(pixel_r.mean().item()), self.global_step)
                    self.writer.add_scalar("train/reward_text", float(text_r.mean().item()), self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)

                if save_every > 0 and self.global_step % save_every == 0:
                    ckpt_path = os.path.join(save_dir, f"step{self.global_step}_epoch{epoch+1}.pt")
                    self.save_checkpoint(ckpt_path)

                # 按步验证（可选）
                if eval_every > 0 and self.global_step % eval_every == 0 and self.eval_loader is not None:
                    metrics = self.evaluate(max_length=max_length)
                    if metrics:
                        cur = metrics.get(metric_for_best, None)
                        if cur is not None and (best_metric is None or cur > best_metric):
                            best_metric = cur
                            best_path = os.path.join(save_dir, "best.pt")
                            self.save_checkpoint(best_path)
                        if self.writer:
                            for k, v in metrics.items():
                                self.writer.add_scalar(f"val/{k}", v, self.global_step)

                pbar.set_postfix({"loss": loss.item(), "reward_mean": rewards_stack.mean().item()})

            # 每个 epoch 结束保存一次 last
            last_path = os.path.join(save_dir, f"last_epoch{epoch+1}.pt")
            self.save_checkpoint(last_path)

            # 按 epoch 验证（当 eval_every==0 且提供 eval_loader）
            if eval_every == 0 and self.eval_loader is not None:
                metrics = self.evaluate(max_length=max_length)
                if metrics:
                    cur = metrics.get(metric_for_best, None)
                    if cur is not None and (best_metric is None or cur > best_metric):
                        best_metric = cur
                        best_path = os.path.join(save_dir, "best.pt")
                        self.save_checkpoint(best_path)
                    if self.writer:
                        for k, v in metrics.items():
                            self.writer.add_scalar(f"val/{k}", v, self.global_step)

        if self.writer:
            self.writer.close()

    def save_checkpoint(self, path: str, extra: Dict[str, Any] = None) -> None:
        """
        保存 GRPO 训练状态：模型（含 LoRA）+ optimizer/scheduler + 训练指针。
        """
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "schedulers": {
                k: (v.state_dict() if v is not None else None) for k, v in self.schedulers.items()
            },
            "cfg": self.cfg,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "tokenizer_name": getattr(self.tokenizer, "name_or_path", None),
            "special_tokens": getattr(self.tokenizer, "special_tokens_map_extended", None),
        }
        if extra is not None:
            state.update(extra)
        torch.save(state, path)

    def load_checkpoint(
        self,
        path: str,
        map_location: str = "cpu",
        load_optimizer: bool = True,
        load_scheduler: bool = True,
    ) -> None:
        """
        从 checkpoint 恢复：模型 + optimizer/scheduler + 训练指针。
        """
        state = torch.load(path, map_location=map_location)
        model_sd = state.get("model", state.get("state_dict", state))
        self.model.load_state_dict(model_sd, strict=False)

        if load_optimizer and "optimizer" in state and state["optimizer"] is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if load_scheduler and "schedulers" in state and state["schedulers"] is not None:
            for k, sch_state in state["schedulers"].items():
                if k in self.schedulers and sch_state is not None and self.schedulers[k] is not None:
                    self.schedulers[k].load_state_dict(sch_state)

        self.start_epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))

    @torch.no_grad()
    def evaluate(self, max_length: int = 1024) -> Dict[str, float]:
        """
        在验证集上评估当前策略，使用确定性采样（k=1，do_sample=False）。
        """
        if self.eval_loader is None:
            return {}

        self.model.eval()
        total_reward = 0.0
        total_pixel = 0.0
        total_text = 0.0
        n = 0

        for batch in self.eval_loader:
            pixel_values = batch["pixel_values"].to(self.device)
            masks = batch.get("masks")
            if masks is not None:
                masks = (masks > 0).long().to(self.device)
            prompts = batch["prompts"]
            targets = batch["targets"]

            input_ids, attention_mask = self._tokenize_prompts(prompts, max_length)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            outputs = self.model(
                {
                    "pixel_values": pixel_values,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                },
                mode="infer",
                generate_kwargs={
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "do_sample": False,
                },
            )

            seg_out = outputs["segmentation"]
            slot_out = outputs["slot"]
            pred_masks = seg_out["logits"]
            pred_scores = seg_out.get("scores", None)
            pred_texts = slot_out.get("texts", [""] * pixel_values.size(0))

            if masks is not None:
                pixel_r = pixel_reward(pred_masks, masks, pred_scores=pred_scores)
            else:
                pixel_r = torch.zeros(pred_masks.size(0), device=pred_masks.device)
            text_r = text_reward(
                pred_texts,
                targets,
                tokenizer=self.tokenizer,
                embedding_layer=self.model.qwen.model.get_input_embeddings(),
            )
            reward = compose_reward(pixel_r, text_r, self.reward_weight)  # [B]

            total_reward += float(reward.mean().item())
            total_pixel += float(pixel_r.mean().item())
            total_text += float(text_r.mean().item())
            n += 1

        self.model.train()
        if n == 0:
            return {}
        return {
            "val_reward": total_reward / n,
            "val_pixel": total_pixel / n,
            "val_text": total_text / n,
        }

"""监督微调训练循环，含验证与可选 TensorBoard 日志。"""

import os
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from echoreason.data.collate import BatchCollator
from echoreason.loss import compute_total_loss
from echoreason.training.optimizer import build_optimizer
from echoreason.training.scheduler import build_scheduler


class SFTTrainer:
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

        self.collate_fn = BatchCollator(stage="sft", tokenizer=None, max_length=self.cfg.get("max_length", 1024))
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.get("train_batch_size", 4),
            shuffle=True,
            num_workers=self.cfg.get("num_workers", 4),
            collate_fn=self.collate_fn,
        )
        self.eval_loader = None
        if eval_dataset is not None:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.cfg.get("eval_batch_size", 4),
                shuffle=False,
                num_workers=self.cfg.get("num_workers", 4),
                collate_fn=self.collate_fn,
            )

        self._apply_freeze()
        self._apply_tuning_modules(self.cfg.get("tuning_modules", []))
        self.optimizer = build_optimizer(self.model, self.cfg)
        total_steps = len(self.train_loader) * max(1, self.cfg.get("epochs", 1))
        self.schedulers = build_scheduler(self.optimizer, self.cfg, total_steps)

        # EMA（指数移动平均）模型，提升泛化稳定性
        self.use_ema = bool(self.cfg.get("use_ema", True))
        self.ema_model = None
        if self.use_ema:
            decay = float(self.cfg.get("ema_decay", 0.9999))
            self.ema_model = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(decay)).to(self.device)

        # 训练状态
        self.global_step = 0
        self.start_epoch = 0
        self.current_epoch = 0

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

    def _build_text_batch(self, prompts: List[str], targets: List[str], max_length: int):
        """
        编码 prompt+target，并将 prompt 部分 label 置为 -100，仅对 target 求损失。
        """
        enc_prompt = self.tokenizer(
            prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        full_texts = [(p + "\n" + t).strip() if t else p for p, t in zip(prompts, targets)]
        enc_full = self.tokenizer(
            full_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        labels = enc_full.input_ids.clone()
        for i in range(len(prompts)):
            prompt_len = (enc_prompt.attention_mask[i] == 1).sum()
            labels[i, :prompt_len] = -100
        return enc_full.input_ids, enc_full.attention_mask, labels

    def train(self):
        self.model.train()
        epochs = self.cfg.get("epochs", 1)
        max_length = self.cfg.get("max_length", 1024)
        accumulation_steps = int(self.cfg.get("gradient_accumulation_steps", 1))
        accumulation_steps = max(1, accumulation_steps)

        eval_every = int(self.cfg.get("eval_every_steps", 0))
        metric_for_best = self.cfg.get("metric_for_best", "val_dice")
        save_dir = self.cfg.get("output_dir", "./outputs/sft")
        save_every = int(self.cfg.get("save_every_steps", 0))
        os.makedirs(save_dir, exist_ok=True)

        best_metric = None
        best_path = None

        global_step = 0
        for epoch in range(epochs):
            self.current_epoch = epoch
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            accum_ctr = 0

            def _do_step():
                nonlocal global_step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.use_ema and self.ema_model is not None:
                    self.ema_model.update_parameters(self.model)

                if self.schedulers["warmup"]:
                    self.schedulers["warmup"].step()
                if self.schedulers["main"]:
                    self.schedulers["main"].step()
                global_step += 1

            last_losses: Dict[str, float] = {}

            for batch_idx, batch in enumerate(pbar):
                pixel_values = batch["pixel_values"].to(self.device)
                masks = batch.get("masks")
                if masks is not None:
                    masks = (masks > 0).long().to(self.device)

                prompts = batch["prompts"]
                targets = batch["targets"]
                input_ids, attention_mask, labels = self._build_text_batch(
                    prompts, targets, max_length=max_length
                )
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(
                    {
                        "pixel_values": pixel_values,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    },
                    mode="sft",
                )
                losses = compute_total_loss(outputs, {"masks": masks}, weights=self.cfg.get("loss_weight"))
                loss = losses["total"] / accumulation_steps

                loss.backward()
                accum_ctr += 1
                last_losses = {k: float(v.detach().item()) for k, v in losses.items() if torch.is_tensor(v)}

                # 仅在累积到指定步数时更新参数
                if accum_ctr % accumulation_steps == 0:
                    _do_step()
                    accum_ctr = 0
                    self.global_step += 1

                    # TensorBoard 记录
                    if self.writer and self.log_every > 0 and self.global_step % self.log_every == 0:
                        lr = self.optimizer.param_groups[0]["lr"]
                        self.writer.add_scalar("train/loss_total", last_losses.get("total", 0.0), self.global_step)
                        self.writer.add_scalar("train/loss_seg", last_losses.get("seg", 0.0), self.global_step)
                        self.writer.add_scalar("train/loss_slot", last_losses.get("slot", 0.0), self.global_step)
                        self.writer.add_scalar("train/lr", lr, self.global_step)

                    # 按 step 保存（可选）
                    if save_every > 0 and self.global_step % save_every == 0:
                        ckpt_path = os.path.join(save_dir, f"step{self.global_step}_epoch{epoch+1}.pt")
                        self.save_checkpoint(ckpt_path)

                    # 按 step 验证（可选）
                    if eval_every > 0 and self.global_step % eval_every == 0 and self.eval_loader is not None:
                        metrics = self.evaluate()
                        if metrics:
                            cur = metrics.get(metric_for_best, None)
                            if cur is not None and (best_metric is None or cur > best_metric):
                                best_metric = cur
                                best_path = os.path.join(save_dir, "best.pt")
                                self.save_checkpoint(best_path)
                            if self.writer:
                                for k, v in metrics.items():
                                    self.writer.add_scalar(f"val/{k}", v, self.global_step)

                pbar.set_postfix(
                    {
                        "loss": loss.item() * accumulation_steps,  # 还原单步 loss 显示
                        "seg": losses.get("seg", 0.0),
                        "slot": losses.get("slot", 0.0),
                    }
                )

            # 处理尾部不足累积步的残余梯度
            if accum_ctr > 0:
                _do_step()

            # 每个 epoch 结束后做一次验证（若提供 eval_loader）
            if eval_every == 0 and self.eval_loader is not None:
                metrics = self.evaluate()
                if metrics:
                    cur = metrics.get(metric_for_best, None)
                    if cur is not None and (best_metric is None or cur > best_metric):
                        best_metric = cur
                        best_path = os.path.join(save_dir, "best.pt")
                        self.save_checkpoint(best_path)
                    if self.writer:
                        for k, v in metrics.items():
                            self.writer.add_scalar(f"val/{k}", v, self.global_step)

            # 保存 last
            last_path = os.path.join(save_dir, f"last_epoch{epoch+1}.pt")
            self.save_checkpoint(last_path)

        if self.writer:
            self.writer.close()

    def save_checkpoint(self, path: str) -> None:
        """
        保存模型权重：若启用 EMA，则优先保存 EMA 权重，确保推理使用平滑参数。
        """
        state_dict = None
        if self.use_ema and self.ema_model is not None:
            model_to_save = self.ema_model.module if hasattr(self.ema_model, "module") else self.ema_model
            state_dict = model_to_save.state_dict()
        else:
            state_dict = self.model.state_dict()

        torch.save(
            {
                "state_dict": state_dict,
                "cfg": self.cfg,
                "optimizer": self.optimizer.state_dict(),
                "schedulers": {k: (v.state_dict() if v is not None else None) for k, v in self.schedulers.items()},
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "tokenizer_name": getattr(self.tokenizer, "name_or_path", None),
                "special_tokens": getattr(self.tokenizer, "special_tokens_map_extended", None),
            },
            path,
        )

    def load_checkpoint(self, path: str, map_location: str = "cpu") -> None:
        """
        加载 checkpoint：若启用 EMA，优先加载到 EMA 并同步到模型；否则直接加载到模型。
        """
        state = torch.load(path, map_location=map_location)
        sd = state.get("state_dict", state)
        if self.use_ema and self.ema_model is not None:
            model_to_load = self.ema_model.module if hasattr(self.ema_model, "module") else self.ema_model
            model_to_load.load_state_dict(sd, strict=False)
            self.model.load_state_dict(sd, strict=False)
        else:
            self.model.load_state_dict(sd, strict=False)

        # 恢复优化器/调度器
        opt_state = state.get("optimizer", None)
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)
        sch_state = state.get("schedulers", None)
        if sch_state:
            for k, v in sch_state.items():
                if k in self.schedulers and v is not None and self.schedulers[k] is not None:
                    self.schedulers[k].load_state_dict(v)

        self.start_epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        在 eval_loader 上验证，返回 val_loss / val_dice（若有）。
        """
        if self.eval_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        n_batches = 0

        for batch in self.eval_loader:
            pixel_values = batch["pixel_values"].to(self.device)
            masks = batch.get("masks")
            if masks is not None:
                masks = (masks > 0).long().to(self.device)

            prompts = batch["prompts"]
            targets = batch["targets"]
            input_ids, attention_mask, labels = self._build_text_batch(
                prompts, targets, max_length=self.cfg.get("max_length", 1024)
            )
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(
                {
                    "pixel_values": pixel_values,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                },
                mode="sft",
            )
            losses = compute_total_loss(outputs, {"masks": masks}, weights=self.cfg.get("loss_weight"))
            total_loss += float(losses["total"].item())
            if "seg" in losses and torch.is_tensor(losses["seg"]):
                total_dice += float(losses.get("seg_main_dice", losses["seg"]).item())
            n_batches += 1

        self.model.train()
        if n_batches == 0:
            return {}

        return {
            "val_loss": total_loss / n_batches,
            "val_dice": total_dice / n_batches if total_dice > 0 else 0.0,
        }

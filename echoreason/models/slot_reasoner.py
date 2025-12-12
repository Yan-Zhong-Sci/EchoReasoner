# -*- coding: utf-8 -*-
"""
SlotReasoner：基于 Qwen 生成三槽位 + 最终答案。

支持两种模式：
  - 训练（teacher forcing）：传入 labels，返回 logits/hidden_states/loss；
  - 推理：generate，自回归生成完整输出。
"""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class SlotReasoner(nn.Module):
    def __init__(
        self,
        llm: nn.Module,
        tokenizer: Any,
        max_answer_len: int = 128,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_answer_len = int(max_answer_len)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        generate: bool = False,
        **gen_kwargs: Any,
    ) -> Dict[str, Any]:
        if generate:
            gen_args = {
                "attention_mask": attention_mask,
                "max_new_tokens": gen_kwargs.pop("max_new_tokens", self.max_answer_len),
                "do_sample": gen_kwargs.pop("do_sample", True),
                "temperature": gen_kwargs.pop("temperature", 1.0),
                "top_p": gen_kwargs.pop("top_p", 0.9),
                "output_scores": True,
                "return_dict_in_generate": True,
                **gen_kwargs,
            }
            if inputs_embeds is not None:
                outputs = self.llm.generate(inputs_embeds=inputs_embeds, **gen_args)
            else:
                outputs = self.llm.generate(input_ids=input_ids, **gen_args)

            sequences = outputs.sequences
            texts = [self.tokenizer.decode(seq, skip_special_tokens=False) for seq in sequences]
            # 计算每条序列的对数概率总和（基于输出的 scores）
            logprobs = None
            if outputs.scores is not None:
                # outputs.scores 是长度 T 的列表，每个是 [B,V]
                seq_len = sequences.size(1)
                # 只对新生成的部分求 logprob
                gen_len = len(outputs.scores)
                logprobs_list = []
                for b in range(sequences.size(0)):
                    lp = 0.0
                    for t, score in enumerate(outputs.scores):
                        token_id = sequences[b, seq_len - gen_len + t]
                        lp += torch.log_softmax(score[b], dim=-1)[token_id]
                    logprobs_list.append(lp)
                logprobs = torch.stack(logprobs_list, dim=0)
            return {"sequences": sequences, "texts": texts, "logprobs": logprobs}

        outputs = self.llm(
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        return {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states,
            "loss": getattr(outputs, "loss", None),
        }

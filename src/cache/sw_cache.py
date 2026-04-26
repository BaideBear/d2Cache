import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

from src.frame import Frame, FrameDelta
from src.cache.base import dCache, AttentionContext
from src.utils import is_adapted_from_ar


class SlidingWindowCache(dCache):

    def __init__(self, model_config, window_size: int = 32):
        super().__init__(model_config)
        self.window_size = window_size
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self.active_q_mask: torch.Tensor | None = None
        self._new_block_start = False

    @contextmanager
    def model_forward(self, x: torch.Tensor):
        with super().model_forward(x=x) as ctx:
            B, T, C = x.shape
            if self.active_q_mask is not None:
                if B != self.active_q_mask.size(0):
                    self.active_q_mask = self.active_q_mask[0].expand(B, -1)
                ctx.x = x[self.active_q_mask].view(B, -1, C)

            yield ctx

            if self.active_q_mask is not None:
                assert ctx.logits is not None
                ctx.logits = torch.zeros(
                    (B, T, ctx.logits.size(-1)),
                    dtype=ctx.logits.dtype,
                    device=ctx.logits.device,
                ).masked_scatter_(self.active_q_mask.unsqueeze(-1), ctx.logits)

    @contextmanager
    def attention(
        self,
        layer_idx: int,
        x: torch.Tensor,
        attn_norm: nn.Module,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        with super().attention(
            layer_idx,
            x,
            attn_norm,
            q_proj,
            k_proj,
            v_proj,
            attention_mask,
            position_ids,
        ) as ctx:
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(ctx.k)
                self.value_cache.append(ctx.v)
            elif self._new_block_start:
                self.key_cache[layer_idx][self.active_seq_mask] = ctx.k
                self.value_cache[layer_idx][self.active_seq_mask] = ctx.v
            else:
                assert self.active_q_mask is not None
                if layer_idx == 0:
                    active_seq_idx = torch.where(self.active_seq_mask)[0]
                    m_nonzero = self.active_q_mask.nonzero(as_tuple=False)
                    self._active_q_indices = (
                        active_seq_idx[m_nonzero[:, 0]],
                        m_nonzero[:, 1],
                    )

                self.key_cache[layer_idx][self._active_q_indices] = ctx.k.flatten(0, 1)
                self.value_cache[layer_idx][self._active_q_indices] = ctx.v.flatten(
                    0, 1
                )
                ctx.k = self.key_cache[layer_idx][self.active_seq_mask]
                ctx.v = self.value_cache[layer_idx][self.active_seq_mask]

            if layer_idx == 0:
                self._q_position_ids, self._kv_position_ids = (
                    AttentionContext.select_position_ids(
                        position_ids, self.active_q_mask
                    )
                )
                self._attention_mask = AttentionContext.convert_attention_mask(
                    attention_mask,
                    dtype=ctx.k.dtype,
                    query_length=ctx.q.shape[1],
                    key_value_length=self.value_cache[layer_idx].shape[1],
                )

            ctx.q_position_ids = self._q_position_ids
            ctx.kv_position_ids = self._kv_position_ids
            ctx.attention_mask = self._attention_mask

            yield ctx

    def on_step_end(self, block_mask: torch.Tensor, frame: Frame, delta: FrameDelta):
        new_frame = frame.apply_delta(delta)
        P = frame.prompts.size(-1)
        G = new_frame.generated_tokens.size(-1)
        device = new_frame.generated_tokens.device

        remaining_mask = (
            new_frame.generated_tokens[self.active_seq_mask] == self.mask_token_id
        )
        B_active = remaining_mask.size(0)

        response_q = torch.zeros(
            (B_active, G), dtype=torch.bool, device=device
        )
        for i in range(B_active):
            positions = torch.where(remaining_mask[i])[0]
            if len(positions) == 0:
                continue
            window_positions = positions[:self.window_size]
            response_q[i, window_positions] = True

        q_mask = F.pad(response_q, (P, 0), value=False)

        if is_adapted_from_ar(self.model_config):
            q_mask = F.pad(q_mask[:, 1:], (0, 1), value=False)
            q_mask[:, P - 1] = q_mask[:, P:].any(dim=-1)

        self.active_q_mask = q_mask
        self._new_block_start = False

    def on_block_start(self, block_mask: torch.Tensor, frame: Frame):
        self._new_block_start = True
        self.active_q_mask = None

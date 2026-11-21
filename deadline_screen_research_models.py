"""Versioned adapters for the frozen eleven-arm comparison; no model changes."""
import numpy as np
import torch
from torch import nn
import deadline_screen_prototype as original
import deadline_screen_fast_runtime as fast
from controller_v3_fused_runtime import native


class Critic(original.Critic):
    def __init__(self, definition, seed):
        key = definition['model_key']
        if key != 'additive64_nonpositive':
            super().__init__(key, seed, definition['screened'])
        else:
            nn.Module.__init__(self)
            self.arm = self.base_arm = key
            self.screened = definition['screened']
            self.scaled = self.anchored = False
            self.model = original.existing.make_model(key, seed)
        assert self.scaled == definition['horizon_scaling']
        assert self.anchored == definition['terminal_anchor']


@native
def validate_unmasked(x):
    if x.shape != (533,):
        raise ValueError('533 observations required')
    for value in x:
        if not np.isfinite(value):
            raise ValueError('finite state required')
    t = np.float32(x[-1] * np.float32(40.))
    slot = round(t)
    if abs(t-slot) > 1e-4 or slot < 0 or slot >= 40:
        raise ValueError('nonterminal slot-grid state required')
    for i in range(30):
        if x[8*i+6] < 0 or x[8*i+6] > 1 or x[8*i+7] < 0 or x[8*i+7] > 1:
            raise ValueError('normalized task features required')
    return np.zeros(30, np.bool_), 40-slot


class Runtime(fast.Runtime):
    def __init__(self, model):
        self.arm, self.scaled, self.screened = model.base_arm, model.scaled, model.screened
        self.model = (model.model.model if model.arm in original.existing.CORRECTED else model.model).eval()

    @torch.inference_mode()
    def values(self, state):
        x = np.asarray(state)
        if x.dtype != np.float32 or x.ndim != 1:
            raise ValueError('one FP32 state required')
        excluded, h = fast.validate_and_screen(x) if self.screened else validate_unmasked(x)
        tensor = torch.from_numpy(x)[None]
        if self.arm in original.NEW_ARMS:
            raw = self.model.net(tensor).numpy()[0].copy()
            for i in np.flatnonzero(excluded):
                raw[5*i] = np.float32(np.inf)
            if self.arm == 'plain24_nonpositive':
                a, q = fast.decoder.flat_decode(raw, 2)
            else:
                a, q = fast.decoder.v3_decode(raw, x, self.arm == 'amortized_workload_price24')
        elif self.arm == 'additive64_nonpositive':
            raw = self.model(tensor).numpy()[0].reshape(-1).copy()
            for i in np.flatnonzero(excluded):
                raw[5*i] = np.float32(np.inf)
            a, q = fast.decoder.flat_decode(raw, 1)
        else:
            raw = self.model(tensor).numpy()[0].reshape(-1)
            a, q = fast.branch_values(raw, excluded, 1 if self.arm == 'bdq64_uniform_adapted' else 0)
        return a[None], np.array([np.float32(q)*np.float32(h if self.scaled else 1.)], np.float32)

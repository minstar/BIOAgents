"""Domain-balanced sampler for veRL.

Ensures each batch has proportional representation from all domains.
Implements veRL's AbstractSampler interface.

Usage in config:
    data.sampler.class_path=scripts/verl/domain_balanced_sampler.py
    data.sampler.class_name=DomainBalancedSampler
"""

import json
import random
from collections import defaultdict
from typing import Iterator

import torch
from torch.utils.data import Sampler


def _get_domain(extra_info) -> str:
    """Extract domain from extra_info field."""
    if isinstance(extra_info, str):
        return json.loads(extra_info).get("domain", "unknown")
    elif isinstance(extra_info, dict):
        return extra_info.get("domain", "unknown")
    return "unknown"


class DomainBalancedSampler(Sampler):
    """Sampler that yields indices ensuring balanced domain coverage.

    Each epoch cycles through all domains equally. Within each domain,
    samples are shuffled. Smaller domains are repeated to match the
    target per-domain count.
    """

    def __init__(self, dataset, seed: int = 42, **kwargs):
        super().__init__(dataset)
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

        # Build domain -> indices mapping
        self.domain_indices = defaultdict(list)
        for idx in range(len(dataset)):
            row = dataset[idx]
            extra_info = row.get("extra_info", {})
            domain = _get_domain(extra_info)
            self.domain_indices[domain].append(idx)

        self.domains = sorted(self.domain_indices.keys())
        self.max_domain_size = max(len(v) for v in self.domain_indices.values())

        # Total length: max_domain_size * num_domains (all domains see equal samples)
        self._len = self.max_domain_size * len(self.domains)

        print(f"[DomainBalancedSampler] {len(self.domains)} domains, "
              f"max_size={self.max_domain_size}, total={self._len}")
        for d in self.domains:
            print(f"  {d}: {len(self.domain_indices[d])} samples "
                  f"(repeat {self.max_domain_size / len(self.domain_indices[d]):.1f}x)")

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)

        # For each domain, create a shuffled list of indices, repeated to match max_domain_size
        domain_pools = {}
        for domain in self.domains:
            indices = self.domain_indices[domain].copy()
            rng.shuffle(indices)
            # Repeat to fill
            repeats = self.max_domain_size // len(indices)
            remainder = self.max_domain_size % len(indices)
            pool = indices * repeats + indices[:remainder]
            rng.shuffle(pool)
            domain_pools[domain] = pool

        # Interleave: round-robin across domains, then shuffle within blocks
        all_indices = []
        for i in range(self.max_domain_size):
            block = [domain_pools[d][i] for d in self.domains]
            rng.shuffle(block)
            all_indices.extend(block)

        self.epoch += 1
        return iter(all_indices)

    def __len__(self) -> int:
        return self._len

    def update(self, **kwargs):
        """Curriculum learning interface (no-op for now)."""
        pass

    def set_epoch(self, epoch: int):
        self.epoch = epoch

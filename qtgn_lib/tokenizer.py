import re
import torch

class BlockTokenizer:
    def __init__(self, dataset_pairs=None):
        self.block_to_idx = {"<PAD>": 0, "<SOS>": 1}
        self.idx_to_block = {0: "<PAD>", 1: "<SOS>"}
        self.vocab_size = 2
        
        if dataset_pairs:
            self.build_vocab(dataset_pairs)

    def _split_text(self, text):
        return [b for b in re.split(r'(-->)|([.!?])', text) if b and b.strip()]

    def build_vocab(self, dataset_pairs):
        all_blocks = set()
        for p, t in dataset_pairs:
            all_blocks.update(self._split_text(p))
            all_blocks.update(self._split_text(t))
        
        for block in sorted(list(all_blocks)):
            if block not in self.block_to_idx:
                idx = len(self.block_to_idx)
                self.block_to_idx[block] = idx
                self.idx_to_block[idx] = block
        
        self.vocab_size = len(self.block_to_idx)

    def encode(self, text):
        blocks = self._split_text(text)
        indices = [self.block_to_idx.get(b, 0) for b in blocks]
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        return "".join([self.idx_to_block.get(i, "?") for i in indices])

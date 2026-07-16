"""[dgxarley] minimax_m2.py: add set_embed_and_head for NEXTN speculative decoding.

Patch MiniMaxM2ForCausalLM: add set_embed_and_head for NEXTN speculative decoding.
The model has get_embed_and_head but is missing the setter, which eagle_worker.py
calls to share the target model's embed/head weights with the draft model.
Every other NEXTN-capable model (DeepSeek, GLM, Llama) has this method.
"""

from _patchlib import Patch

patch = Patch(name="MiniMaxM2ForCausalLM set_embed_and_head for NEXTN", target="sglang/srt/models/minimax_m2.py")

OLD = "    def get_embed_and_head(self):"

NEW = """    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_embed_and_head(self):"""


@patch.run
def apply(p: Patch) -> None:
    p.replace(OLD, NEW, marker="def set_embed_and_head", what="set_embed_and_head NEXTN setter")

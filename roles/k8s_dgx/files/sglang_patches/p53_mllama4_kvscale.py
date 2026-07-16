"""[dgxarley] Llama-4 NVFP4 KV-scale patches (QUALITY: load the baked FP8 KV scales).

The modelopt-NVFP4 checkpoint bakes FP8 KV scales (...k_proj.k_scale /
...v_proj.v_scale). Without these two SGLang never loads them -> falls back to
scale 1.0 ("less accurate results"). Verified on 0.5.15-sm121: 0 not-loaded
warnings, and the triton attn backend uses the loaded scale (cache_k.div_(k_scale)).
NOT a load-blocker (serves without them, just at scale 1.0). Both are needed
(A alone is a no-op). Same no-gate rationale as the loader patches above:
llama4.py / mllama4.py are imported only for the Llama4 arch.
  A) llama4.py: Llama4Attention built RadixAttention() WITHOUT quant_config
     (unlike llama.py / qwen2.py) -> create_weights never ran -> k_scale/v_scale
     stayed plain None attrs, never in named_parameters(). Fix: pass quant_config.
  B) mllama4.py: _handle_scale_remapping returned a bool but never COPIED the
     remapped scale into the param (llama.py's loader does; mllama4's reimpl
     dropped the copy). Fix: thread loaded_weight in + _handle_default_weight().
"""

from _patchlib import Patch

patch_llama4 = Patch(name="Llama4Attention RadixAttention quant_config (A)", target="sglang/srt/models/llama4.py")

OLD_A = """        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
            use_irope=self.use_rope,
        )"""

NEW_A = """        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            use_irope=self.use_rope,
        )"""


@patch_llama4.run
def apply_a(p: Patch) -> None:
    p.replace(OLD_A, NEW_A, what="A-quant_config")


patch_mllama4 = Patch(
    name="mllama4 _handle_scale_remapping copies the remapped KV scale (B)", target="sglang/srt/models/mllama4.py"
)

OLD_B_DEF = '''    def _handle_scale_remapping(self, name: str, params_dict: dict) -> bool:
        """Handle scale parameter remapping. Returns True if handled."""
        if "scale" in name and "expert" not in name:
            remapped_name = maybe_remap_kv_scale_name(name, params_dict)
            return remapped_name != name
        return False'''

NEW_B_DEF = '''    def _handle_scale_remapping(
        self, name: str, loaded_weight: torch.Tensor, params_dict: dict
    ) -> bool:
        """Handle scale parameter remapping. Returns True if handled."""
        if "scale" in name and "expert" not in name:
            remapped_name = maybe_remap_kv_scale_name(name, params_dict)
            if remapped_name is None:
                return True
            if remapped_name != name:
                self._handle_default_weight(remapped_name, loaded_weight, params_dict)
                return True
            return False
        return False'''

OLD_B_CALL = """            if self._handle_scale_remapping(name, params_dict):"""

NEW_B_CALL = """            if self._handle_scale_remapping(name, loaded_weight, params_dict):"""


@patch_mllama4.run
def apply_b(p: Patch) -> None:
    p.replace(OLD_B_DEF, NEW_B_DEF, what="B-def")
    p.replace(OLD_B_CALL, NEW_B_CALL, what="B-call")

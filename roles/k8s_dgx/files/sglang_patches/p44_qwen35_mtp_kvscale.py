"""[dgxarley] qwen3_5_mtp.py: load the baked FP8 KV scales onto the MTP attention.

Companion to p51_qwen35_kvscale.py. p51 makes qwen3_5.py's load_weights apply
`QWEN3_5_KV_SCALE_MAPPER` (a WeightsMapper that renames the checkpoint's
...self_attn.k_proj.k_scale / ...v_proj.v_scale onto RadixAttention's ...attn.
k_scale / ...attn.v_scale) so the MAIN model's baked FP8 KV scales load. But the
MTP/NEXTN draft head has its OWN load_weights in qwen3_5_mtp.py, which p51 does
not touch: it strips ".self_attn" from the name but never turns k_proj.k_scale
into attn.k_scale, so the mapped name never matches the registered param and the
scale is silently dropped. The draft attention then keeps the -1.0 sentinel and
falls back to KV scale 1.0.

Only relevant once the MTP head is quantized with baked KV scales (p43 +
kikube's requant_mtp_nvfp4.py, which borrows the last full-attn layer's k/v
scales onto the MTP attention). Verified against
qwen3.6-35b-a3b-nvfp4-mtp-modelopt on GB10/sm121: with this patch the MTP draft
attn (layer_id 0) loads k_scale=0.03982... -- byte-identical to the main model's
last full-attn layer (layer 39), the requant's borrow source -- instead of the
1.0 default. INERT for a BF16 MTP (no k_scale keys in the stream -> the mapper is
a no-op). No model-name gate: qwen3_5_mtp.py is imported only for this arch.

Fix: apply the same QWEN3_5_KV_SCALE_MAPPER (defined in qwen3_5.py by p51) to the
weight stream at the top of the MTP load_weights, before the ".self_attn" strip.
Imported lazily so a drifted/absent p51 degrades to "no MTP KV scales" (a warning
+ scale 1.0) rather than crashlooping the draft worker. Deletable together with
p51/p43 once SGLang upstream loads a quantized MTP head's baked KV scales.
"""

from _patchlib import Patch

patch = Patch(
    name="load baked FP8 KV scales onto the MTP attention",
    target="sglang/srt/models/qwen3_5_mtp.py",
)

OLD = """    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ):
        stacked_params_mapping = ["""

NEW = """    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ):
        # [patch dgxarley] remap baked FP8 KV scales onto the MTP attention. The
        # main model gets this via p51 (qwen3_5.py); the MTP has its own
        # load_weights that never turns k_proj.k_scale into attn.k_scale.
        try:
            from sglang.srt.models.qwen3_5 import QWEN3_5_KV_SCALE_MAPPER as _kv_map
            weights = _kv_map.apply(weights)
        except Exception as _kv_e:
            import logging as _kv_log
            _kv_log.getLogger("sglang").warning(
                "[patch] qwen35 MTP kv-scale mapper unavailable (p51 drift?): %s", _kv_e
            )
        stacked_params_mapping = ["""


@patch.run
def apply(p: Patch) -> None:
    p.replace(OLD, NEW, marker="QWEN3_5_KV_SCALE_MAPPER as _kv_map", what="mtp kv-scale mapper")

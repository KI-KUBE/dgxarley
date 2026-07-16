# Plan: SGLang-Runtime-Patches aus `sglang_launch.sh` in `files/sglang_patches/` auslagern

Status (2026-07-16): **Phase 0 bis 3 sind umgesetzt und verifiziert.** `launch.sh` ist von 3899 auf
670 Zeilen (193 KB auf 34 KB) geschrumpft, **36 Patches** liegen als Dateien vor, die ConfigMap misst
198 KB (Limit 1 MiB). In `launch.sh` ist **kein einziger Source-Patch mehr** — nur noch Launcher-Logik
(apt/pip-Bootstrap, der `SGLANG_HUNYUAN_TOKEN_SUFFIX`-Export, die beiden `.pth`-Installationen, der
Flag-Bau und `exec`). **Noch nicht deployt** (kein Rollout ohne Freigabe). Offen: Phase 4.

## Testabdeckung: Gate-Profile (Phase 3 hat das erzwungen)

Mit Phase 3 sind die bash-`if`-Gates zu `when=`-Gates geworden. Ein Lauf mit `neutral/none`
exerziert davon **keinen einzigen**. Die Harness fährt deshalb vier Profile, und deren
Referenzmengen unterscheiden sich messbar:

| Profil | Env | berührte Dateien | deckt ab |
|---|---|---|---|
| `neutral` | `SGLANG_MODEL=neutral/none` | 26 | alle ungegateten (DSA, Phase-2-Set) |
| `hy3` | Hy3-Modell + `SPECULATIVE_ENABLED=true` + hunyuan-Parser | 29 | `p64`, `p40`, `p41`, `p42` |
| `glm5` | GLM-5-Modellname | 27 | `p13` |
| `spec` | `SPECULATIVE_ENABLED=true` | 27 | `p42` ohne Hunyuan |

**Nicht abgedeckt: `p62` / `p63`** (Hunyuan tool/reasoning parser). Sie sind auf `0.5.15-sm121`
**dauerhafte No-Ops**: das Image enthält PR #29920 bereits, der Guard greift, beide Seiten tun nichts.
Der Tree-Diff kann über ihre Konversion also nichts aussagen. Ihr eigener RE-SYNC-Hinweis sagt
"DELETE this block" für genau diesen Fall — Kandidaten zum Löschen, bewusst nicht eigenmächtig getan.

## Abweichungen vom ursprünglichen Plan (erzwungen, nicht kosmetisch)

* **Dateinamen `p<NN>_...` statt `<NN>_...`** — mypy läuft strict über das ganze Repo und lehnt einen
  Modulnamen mit führender Ziffer ab ("invalid module name"). Der Runner-Glob ist entsprechend
  `p[0-9][0-9]_*.py`. Achtung: ein Name wie `p23b_...` matcht NICHT (nach zwei Ziffern muss `_`
  folgen); zum Einsortieren eine freie Nummer nehmen.
* **Kein `PYTHONPATH` nötig** — beim Aufruf `python3 /patches/pNN_x.py` ist `sys.path[0]` bereits
  `/patches`, `from _patchlib import ...` funktioniert ohne Zutun.
* **DSA-Patches bleiben in Phase 3**, obwohl sie ungegatet sind (der Plan hatte sie als "gegatet"
  einsortiert, das stimmte nicht). Grund für die Verschiebung ist ein anderer: an DSA wird parallel
  aktiv gearbeitet (3 Commits am 2026-07-16), ein Umzug jetzt erzeugt nur Konflikte.
* **`_patchlib` brauchte mehr API als gedacht**: `replace_all()` (siehe unten), `prepend()` und einen
  `code`-Buffer für tolerante Edits. Letzteren brauchten zwei Konvertierungen unabhängig voneinander
  (`p10`, `p61`) — ohne ihn hätten sie auf private Interna zugegriffen.

## Was die Verifikation tatsächlich gefunden hat (der Grund, warum es sie gibt)

Der Tree-Diff hat in Phase 2 **drei echte Konversionsfehler** gefangen, alle drei bei **null
ANCHOR-DRIFT und "Patched"-Erfolgsmeldung im Log** — sie wären im Betrieb also stumm gewesen:

1. **`p50` / `p51` (qwen3_5.py):** die Originale ersetzten mit `s.replace(old, new)` **alle**
   Vorkommen (2 bzw. 4 im echten Image), `_patchlib.replace()` nur das erste. Ergebnis: halb
   gepatchte Datei, Log meldet Erfolg. Fix: neues `replace_all()`, plus ein Audit **aller**
   Konversionen auf die Ersetzungs-Semantik des Originals (`p24` war latent betroffen: heute nur
   1 Vorkommen, morgen vielleicht nicht).
2. **`p10` (weight_utils.py):** der Logger-Import des Originals hing an
   `if "\nlogger = " not in code`. Ohne die Bedingung wurde ein zweiter `import logging` +
   `logger = ...` injiziert.

Lehre für Phase 3: **Vor jeder Konversion die Ersetzungs-Semantik des Originals prüfen** (`replace`
mit/ohne `count`, `sed` ersetzt pro *Zeile*), und die Trefferzahl im echten Image zählen. Das wurde
in Phase 3 durchgehend gemacht (alle Anker dort: genau 1 Vorkommen, alle Originale `count=1`), die
Regel hat also gehalten.

### Und was Phase 3 zusätzlich gefunden hat: der Gruppen-Marker (p31)

Der Idempotenz-Test (Runner ZWEIMAL, wie bei jedem Pod-Restart) fing einen Fehler, den der
Tree-Diff **strukturell nicht sehen kann**, weil der nur einen Lauf vergleicht:

`p31` (flashinfer_gather) hat drei Edits in `dsa_backend.py`. Das Original prüfte **einen** Marker
**einmal vorab** und übersprang damit alle drei gemeinsam. Die Konversion verließ sich stattdessen
auf Pro-Edit-Proben (`new` als Probe). Das ist tödlich, weil **`p33` später genau den Text
umschreibt, den `p31` injiziert**: beim zweiten Lauf findet die Probe von B2/B3 ihren Text nicht
mehr, der Anker passt noch, also feuern sie erneut und zerlegen die Datei. Das Original war
idempotent, die Konversion nicht — eine reine Regression.

Regel daraus: **Wenn das Original eine Gruppe von Edits an EINEM Marker vorab gated, muss die
Konversion das auch tun.** Pro-Edit-Proben sind nur zulässig, solange kein späterer Patch den
injizierten Text anfasst. Und: **Idempotenz immer gegen ein Profil testen, in dem die Patches
wirklich feuern**, sowie gegen das Original gegenprüfen (`run_idem_old.sh`), um Regression von
Altlast zu unterscheiden.

## Verifikations-Harness (steht, spark5)

Beide Phasen laufen in je einem **frischen** podman-Container (Image ist unveränderlich, also kein
Restore nötig), die berührten Dateien werden über ein md5-Manifest über **alle** `.py` in
dist-packages ermittelt, nicht nur `sglang/`. Das ist keine Kosmetik: die Patch-Phase fasst auch
`flashinfer` (3 Dateien) und `transformers` (1) an, eine sglang-only-Momentaufnahme hätte `p60`/`p61`
komplett übersehen. Referenzmenge aktuell: **26 Dateien**.

    cd /root/patchtest && for P in old new; do
      podman run --rm -v /root/patchtest:/patchtest --entrypoint bash \
        xomoxcc/dgx-spark-sglang:0.5.15-sm121 /patchtest/run_phase.sh $P
    done && bash compare.sh
    # dazu: run_idem.sh (Runner zweimal, zweiter Lauf muss nichts ändern)

Ergebnis Phase 2: `touched-set: identical`, `TREE-DIFF: IDENTICAL`, 0 Drift auf beiden Seiten,
zweiter Lauf ändert nichts (25× "already applied", 0× "Patched").

## Ausgangslage

`roles/k8s_dgx/files/sglang_launch.sh` ist auf 3476 Zeilen / 193 KB gewachsen. Der Löwenanteil sind
Runtime-Patches gegen `/usr/local/lib/python3.12/dist-packages/sglang/...`:

* ca. 25 Python-Heredocs (`python3 - <<'PATCH_*_EOF'`), z. B. `PATCH_HUNYUAN_SHARED_EOF`,
  `PATCH_MLLAMA4_LOADER_EOF`, `PATCH_DSA_TORCH_*` (5 Stück), `PATCH_DSA_FLASHINFER_GATHER_EOF`
  (270 Zeilen allein), `PATCH_MIXED_NVFP4_*`, `PATCH_QWEN35_*`, `PATCH_VLM_IGNORE_EOF`,
  `PATCH_NEMOTRONH_OMNI_WRAPPER_EOF`, `PATCH_FI_*`.
* ca. 8 sed-/grep-basierte Blöcke (`WEIGHT_UTILS`, `LOADER`, `MOE_WNA16`, `MODELOPT_QUANT`,
  `CUTLASS_MOE`, `MINIMAX_M2`, `DEEPSEEK_V3_CFG`, `FST_F`, `HF_UTILS`, `TOKPY`).
* echter Launcher-Anteil (apt-Bootstrap, `.pth`-Installation, Flag-Zusammenbau, `exec`):
  geschätzt unter 400 Zeilen.

Probleme daraus:

1. Jede Patch-Änderung ist ein Diff mitten in einer 3.5k-Zeilen-Datei, Review ist schwer.
2. Der Heredoc-Inhalt ist für Editor/Tooling kein Python: keine Syntaxprüfung, kein black, kein mypy.
3. Gate-Logik (Bash-`if` außen) und Patch-Logik (Python innen) sind getrennt, das Gate ist beim Lesen
   des Patches oft 50 Zeilen weiter oben.
4. Boilerplate wird copy-pasted: Datei lesen, Marker-Guard, `replace(..., 1)`, `ANCHOR-DRIFT`-print.
   Genau dort saß der Idempotenz-Bug vom 2026-07-16 (`old_buffered` ist Präfix von `new_buffered`).
5. Ein Patch, der nur ein Modell betrifft, rollt trotzdem jeden Pod neu (ein Checksum über die
   ganze Datei).

## Zielbild

```
roles/k8s_dgx/files/sglang_patches/
  _patchlib.py                        # gemeinsame Helfer, kein Patch
  10_weight_utils_tqdm_logger.py
  10_loader_shard_progress.py
  20_modelopt_mixed_nvfp4_dispatch.py
  20_modelopt_mixed_nvfp4_variant.py
  20_linear_nvfp4_scale.py
  20_vlm_should_ignore_layer.py
  20_moe_wna16_qzeros_ep.py
  30_dsa_torch_backend.py
  30_dsa_flashinfer_gather.py
  40_hy3_nextn_bf16.py
  40_ds_nextn_mixed_mtp.py
  50_hunyuan_token_suffix.py
  50_mllama4_loader.py
  ...
```

`sglang_launch.sh` schrumpft auf Bootstrap + Patch-Runner + Flag-Bau, realistisch 400 bis 500 Zeilen.

### Ein Patch = eine Datei = ein Python-Modul, self-gating

Jeder Patch entscheidet **selbst** anhand von `os.environ`, ob er zutrifft, statt von einem
Bash-`if` umschlossen zu werden. Das hält Gate und Patch beieinander und macht den Runner dumm.
Der Preis (ein `python3`-Start pro Patch, ca. 0,2 s, also ~5 s gesamt) ist gegen die 7 bis 8 Minuten
Head-Startup irrelevant.

```python
"""[dgxarley] hunyuan_v3.py: remap .shared_experts. -> .shared_mlp. in load_weights.

Grund: HYV3-Checkpoints benennen den Shared Expert `mlp.shared_experts.*`, SGLangs Modul
heißt `shared_mlp`; ohne Remap werden die (echten, FP4) Gewichte still verworfen -> NaN.
Upstream: noch nicht eingereicht.
Re-Sync: bei Image-Bump prüfen, ob load_weights den Remap schon hat (Guard no-opt dann).
"""
from _patchlib import Patch, gate_model

patch = Patch(
    name="hunyuan-shared-experts",
    target="sglang/srt/models/hunyuan_v3.py",
    when=gate_model("Hy3", "Hunyuan"),
)

@patch.run
def apply(p):
    p.insert_after(
        anchor="        for name, loaded_weight in weights:\n",
        text='            name = name.replace(".shared_experts.", ".shared_mlp.")\n',
        marker='replace(".shared_experts.", ".shared_mlp.")',
    )
```

`_patchlib.py` liefert genau das, was heute pro Block dupliziert wird:

* `Patch(name, target, when=...)`: löst `target` gegen `dist-packages` auf, meldet
  `ANCHOR-DRIFT: <name>: target file missing` statt zu crashen, überspringt bei `when=False`
  mit einer Zeile Log.
* `p.replace(old, new, marker=...)` / `p.insert_after(anchor, text, marker=...)`:
  Marker-Guard **zuerst** (der 2026-07-16-Bug ist damit strukturell ausgeschlossen), exakt eine
  Ersetzung, einheitliches `Patched <file>: <name>` bzw.
  `ANCHOR-DRIFT: <file>: <name> (SGLang version drift; re-check anchor)`.
* `p.write_new_file(relpath, content)` für die Fälle wie `PATCH_DSA_TORCH_NEWFILE_EOF`.
* Gate-Helfer: `gate_model(*substrings)`, `gate_env("SGLANG_SPECULATIVE_ENABLED", "true")`,
  `gate_always()`.
* Rückgabe-Konvention: Exit 0 immer (auch bei Drift), damit `set -e` im Launcher nicht den Pod
  killt. Genau das heutige Verhalten, aber an einer Stelle statt 30-mal.

### Was NICHT in `sglang_patches/` gehört

* apt-Bootstrap, `pip install accelerate`, Transformers-Upgrade.
* `.pth`-Installation (`zz_dsv4_autopatch.pth`, `zz_dsv4_memprobe.pth`), das sind Deployments,
  keine Source-Patches.
* Image-Pattern-Check, Flag-Zusammenbau, `exec`.
* `SGLANG_HUNYUAN_TOKEN_SUFFIX`-Ermittlung: liest `tokenizer_config.json` und **exportiert eine
  Env-Var** für den Serverprozess, ist also Launcher-Logik. Bleibt in der `.sh`, die beiden
  Detector-Patches wandern ins Patch-Verzeichnis und lesen die Var.

### Auslieferung

Neue ConfigMap `{{ inst.prefix }}-patch-scripts`, gemountet auf `/patches` (eigener Top-Level-Pfad,
kein Nested-Mount unter dem bestehenden `/scripts`). `sglang_launch.sh` bekommt:

```bash
SGLANG_PATCH_DIR="${SGLANG_PATCH_DIR:-/patches}"
if [ -d "$SGLANG_PATCH_DIR" ]; then
  export PYTHONPATH="$SGLANG_PATCH_DIR:${PYTHONPATH:-}"
  for _p in "$SGLANG_PATCH_DIR"/[0-9][0-9]_*.py; do
    [ -e "$_p" ] || continue
    python3 "$_p" || echo "[launch] WARNING: patch $(basename "$_p") exited non-zero, continuing"
  done
fi
```

Sortierung über das `NN_`-Präfix, also deterministisch und ohne Registry-Datei. Präfixgruppen:
`10` Loader/Progress, `20` Quant, `30` Attention/DSA, `40` Spekulativ/MTP, `50` Modelle/Parser,
`60` Flashinfer/Env.

Ansible-Seite in `roles/k8s_dgx/tasks/sglang_instance.yml`:

```yaml
- name: Create SGLang patch-scripts ConfigMap ({{ inst.prefix }})
  kubernetes.core.k8s:
    definition:
      apiVersion: v1
      kind: ConfigMap
      metadata:
        name: "{{ inst.prefix }}-patch-scripts"
        namespace: "{{ sglang_namespace }}"
      data: >-
        {{ dict(_sglang_patch_files | map('basename') | zip(
                _sglang_patch_files | map('_file_content'))) }}
  vars:
    _sglang_patch_files: "{{ query('fileglob', role_path ~ '/files/sglang_patches/*.py') | sort }}"
```

`lookup('file')` in einem `map()` geht nicht direkt, praktikabel ist stattdessen eine
`ansible.builtin.set_fact`-Schleife über `query('fileglob', ...)` mit
`combine({ item | basename: lookup('file', item) })`. Basenames sind gültige ConfigMap-Keys
(`[-._a-zA-Z0-9]`), Unterverzeichnisse gibt es bewusst nicht.

**Größenbudget:** ConfigMaps sind hart auf 1 MiB begrenzt. Heute liegen 193 KB (`launch.sh`)
plus `dsv4_memprobe.py` in einer ConfigMap. Nach dem Split etwa 30 KB `launch.sh` plus ca. 160 KB
verteilt auf die Patch-ConfigMap. Reserve bleibt reichlich, aber die Aufteilung auf zwei ConfigMaps
verdoppelt den Puffer, statt ihn zu verbrauchen.

### Der Checksum-Fallstrick (wichtig)

Heute:

```yaml
checksum/launch-script: "{{ (lookup('file', .../sglang_launch.sh) ~ lookup('file', .../dsv4_memprobe.py)) | hash('sha256') }}"
```

Zieht man die Patches raus, ohne die Annotation zu erweitern, rollt eine reine Patch-Änderung die
Pods **nicht** mehr neu, die ConfigMap-Änderung propagiert zwar in den Mount, aber der Patch läuft
nur beim Container-Start. Ergebnis wäre ein stiller Nicht-Effekt, der aussieht wie ein wirkungsloser
Patch. Die Annotation muss also (an beiden Stellen, Zeile 406 und 717) um den Verzeichnis-Hash
erweitert werden:

```yaml
checksum/launch-script: "{{ (lookup('file', .../sglang_launch.sh)
                            ~ lookup('file', .../dsv4_memprobe.py)
                            ~ _sglang_patches_blob) | hash('sha256') }}"
```

mit `_sglang_patches_blob` = Konkatenation der sortierten Patch-Dateien (ein `set_fact` vor den
Deployment-Tasks, einmal berechnet, von Head und Worker geteilt).

Merke außerdem: `lookup('file')` strippt das schließende `\n`, ein Vergleich ConfigMap-Inhalt gegen
Quelldatei mismatcht deshalb immer (siehe `reference_ansible_file_lookup_trailing_newline`). Für den
Hash ist das egal, solange beide Seiten denselben Weg gehen.

## Migration in Phasen (jede Phase ist einzeln deploybar und rückrollbar)

**Phase 0, Gerüst, kein Verhaltens-Delta. ERLEDIGT (Commit 6b77d64).**
`_patchlib.py` + Runner-Loop + ConfigMap + Mount + Checksum-Erweiterung.

**Phase 1, ein Pilot-Patch. ERLEDIGT (Commit 6b77d64).**
`p20_moe_wna16_qzeros_ep.py` (klein, unkonditioniert, gut getestet) raus aus der `.sh`.

**Phase 2, die unkonditionierten Patches. ERLEDIGT.**
23 Patches: mllama4 (2), weight_utils (2) + loader-Progress, linear NVFP4-Scale, VLM-ignore,
Nemotron-Wrapper, Transformers-topk, FP8-out-dtype, Flashinfer (2), modelopt (3), cutlass, minimax,
deepseek-cfg, get_config, mistral-tokenizer, qwen3_5 (2), mixed-NVFP4 (2).
Gotcha, der fast zugeschlagen hätte: der `MODELOPT_QUANT=`-Bash-Variable wurde vom *nachfolgenden*
Block mitbenutzt — eine wörtliche Löschung des einen hätte den anderen gebrochen. Beide sind jetzt
gemeinsam weg (`p23` + `p28`). Bei Phase 3 dieselbe Prüfung fahren: `grep` auf jede im Löschbereich
deklarierte Variable.

**Phase 3, die restlichen Patches. ERLEDIGT.**
12 Patches: `p13` mem_fallback (GLM-5), `p29` CUTLASS-mma (siehe unten), `p30` DSA-torch (5 Blöcke,
eine Datei), `p31`/`p32`/`p33` DSA-flashinfer-gather + prefill + cuda-graph-split, `p40`/`p41`
HY3-NEXTN, `p42` DS-NEXTN-mixed-MTP, `p62`/`p63`/`p64` Hunyuan.
Bash-`if` -> `when=`: das war wie erwartet der Schritt mit echtem Logik-Umzug.
Zwei Dinge, die beim Schneiden zählten:
* Wo der `if` nach dem Umzug LEER zurückbliebe, muss er mit raus (HY3-NEXTN, DSNEXTN). Wo noch
  Launcher-Logik drinsteht, muss er bleiben: der Hunyuan-`if` enthält den
  `SGLANG_HUNYUAN_TOKEN_SUFFIX`-Export, der GLM-5-`if` den pip-Install und den `else`-Zweig.
* **`p29` fehlte in jeder Inventur**: der CUTLASS-`mma.py`-Patch nutzt kein `python3`-Heredoc,
  sondern eine `sed`-Schleife, und wurde von jedem `grep` auf `PATCH_*_EOF` übersehen. Gefunden
  erst durch `grep dist-packages` auf dem Rest-Script. Wer hier weitermacht: nach `sed -i`,
  `python3 -c` und `dist-packages` greppen, nicht nur nach Heredoc-Markern.

**Phase 4, Aufräumen.**
sed-Blöcke, die noch übrig sind, nach Python konvertieren (sie sind ohnehin Anchor-Replacements),
`.sh` durchlesen, Reste an Kommentar-Kontext zu den Patch-Docstrings verschieben, CLAUDE.md-Abschnitt
"SGLang ConfigMap scripts" um das Patch-Verzeichnis ergänzen.

## Verifikation: Tree-Diff statt Hoffnung

Der Refactor ist genau dann korrekt, wenn der **gepatchte dist-packages-Baum identisch** ist. Das ist
direkt messbar, ohne SGLang überhaupt zu starten:

1. Debug-Pod auf einem Spark (`tail -f /dev/null`, kein `app=sglang`-Label, siehe
   `feedback_debug_pod_no_sglang_label`), mit demselben Image und denselben `SGLANG_*`-Env-Vars wie
   der Head.
2. `cp -a /usr/local/lib/python3.12/dist-packages/sglang /tmp/base`
3. Alte `launch.sh` bis vor den `exec` laufen lassen (`SGLANG_PATCH_ONLY=1`-Guard einbauen, oder
   schlicht die Patch-Sektion per `sed -n` extrahieren), Baum nach `/tmp/old` sichern, aus
   `/tmp/base` zurückrollen.
4. Neue `launch.sh` + Runner, Baum nach `/tmp/new`.
5. `diff -r /tmp/old /tmp/new` muss leer sein.

Pro Phase einmal, mit den Env-Kombinationen der real genutzten Profile (mindestens: GLM-5.2-DSA,
Hy3-NVFP4-W4A4, ein mixed-NVFP4-Modell, ein Nicht-Gate-Modell). Das deckt die Gates ab, die der
Tree-Diff sonst nicht anfasst.

Zusätzlich zwei billige Dauerchecks:

* **Idempotenz-Test:** Runner zweimal laufen lassen, der zweite Lauf darf keine Datei mehr ändern
  (`diff -r` gegen den Zwischenstand leer) und muss für jeden Patch "already applied" loggen. Genau
  der Bug-Typ vom 2026-07-16, jetzt automatisch geprüft.
* **Lint:** die Patch-Dateien liegen als echtes Python im Repo, also greifen `make lint` (black,
  line-length 120) und `make tcheck` künftig darauf. `_patchlib.py` bekommt Typannotationen,
  kein `from __future__ import annotations` (Python 3.14+).

## Was der Split nicht löst

* Die Patches bleiben anchor-basiert und driften bei Image-Bumps weiterhin. Der Split macht die
  Drift nur sichtbarer (ein Patch = ein Dateiname im `ANCHOR-DRIFT`-Log statt einer Zeilennummer).
* Der Kommentar-Kontext (das eigentliche Wissen: warum, welcher Upstream-PR, wann löschbar) ist
  wertvoll und darf beim Umzug **nicht** verloren gehen, er wandert 1:1 in den Modul-Docstring.
  Kein Patch ohne Docstring mit Grund + Upstream-Status + Re-Sync-Hinweis.
* Startzeit ändert sich praktisch nicht (~5 s Interpreter-Starts gegen 7 bis 8 min Head-Boot).

## Offene Entscheidungen

1. `/patches` als eigener Mount (Vorschlag) oder zusätzliche Keys in der bestehenden
   `-launch-script`-ConfigMap mit `items[].path: patches/x.py`. Ersteres ist sauberer getrennt,
   letzteres spart einen Volume-Eintrag.
2. Ob `sglang_embed_launch.sh` denselben Runner bekommt (aktuell patcht es nichts, könnte aber von
   `20_*` profitieren) oder bewusst patchfrei bleibt.
3. Ob Patches pro Profil selektierbar werden sollen (`sglang_patches_disabled: [...]` als
   Profil-Knopf, der einzelne Dateien aus der ConfigMap auslässt). Nützlich zum Bisecten bei
   Image-Bumps, aber ein neuer Konfig-Knopf. Vorschlag: erst nach Phase 4, wenn überhaupt.

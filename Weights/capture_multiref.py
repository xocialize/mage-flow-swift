"""Multi-reference (2 and 3 refs) oracle capture.

DURABLE COPY — the live copy runs from mlxengine-image/WIP/mage-flow-oracle/ (paths below are
relative to THAT directory: upstream/, goldens/, .venv/). Copy it back there to re-run.

Original docstring: Multi-reference (2 and 3 refs) oracle capture for the mage-flow-swift multi-ref gate
(AB-A-0047). Same shape as capture_e2e.py: DiT INPUTS per step (img_in pre-hook = the
packed [target, ref_1..ref_N] tokens, txt_norm pre-hook = the VL conditioning features)
plus the decoded render, all at 512² Turbo (4 steps, cfg 1.0, seed 42).

Also runs the REAL screen_edit once per case and records the verdict, so the port's
multi-image filter path (spatial M-RoPE over N images) has an oracle verdict to match.

Outputs: goldens/multiref/<N>/{img_in_XX.npy, txt_XX.npy, render.png, verdict.json, case.json}
E2EGate consumes them with `--edit-refs N`.
"""
import os, sys, json, time, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "upstream"))
os.environ.setdefault("HF_HOME", "/Volumes/Satechi/hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VF_HF_ATTN_IMPL", "sdpa")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from mage_flow.pipeline import load_from_repo, generate_edits
from mage_flow.models.modules._attn_backend import set_attn_backend
from types import SimpleNamespace
from PIL import Image

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

CACHE = "/Volumes/Satechi/hf-cache/hub/models--microsoft--Mage-Flow-Edit-Turbo"
rev = open(os.path.join(CACHE, "refs", "main")).read().strip()
REPO = os.path.join(CACHE, "snapshots", rev)
A = os.path.join(HERE, "upstream", "mage_flow", "assets")
CASES = {
    2: dict(refs=[f"{A}/multiref_000000_0.jpg", f"{A}/multiref_000000_1.png"],
            prompt="Place the headphones from Image 1 on the marble table from Image 2, keeping the neon lighting"),
    3: dict(refs=[f"{A}/multiref_000000_0.jpg", f"{A}/multiref_000000_1.png", f"{A}/dog.jpg"],
            prompt="Put the headphones from Image 1 on the dog from Image 3, standing on the marble table from Image 2"),
}
SEED, STEPS, CFG, SIZE = 42, 4, 1.0, 512
only = [int(a) for a in sys.argv[1:]] or [2, 3]

torch.set_grad_enabled(False)
torch.set_num_threads(max(1, os.cpu_count() - 2))
log(f"loading {REPO} on CPU …")
model = load_from_repo(REPO, device="cpu")
set_attn_backend("sdpa")
tr = model.transformer

for n in only:
    case = CASES[n]
    OUT = os.path.join(HERE, "goldens", "multiref", str(n)); os.makedirs(OUT, exist_ok=True)
    pils = [Image.open(p).convert("RGB") for p in case["refs"]]

    # real filter verdict (the multi-image M-RoPE path), then stub it for the render
    t0 = time.time()
    v = model.txt_enc.screen_edit(case["prompt"], pils)
    verdict = {"violates": bool(v.violates), "categories": list(getattr(v, "categories", []) or []),
               "reason": getattr(v, "reason", None), "raw": getattr(v, "raw", None)}
    json.dump(verdict, open(os.path.join(OUT, "verdict.json"), "w"), indent=1, default=str)
    log(f"[{n}-ref] screen_edit {time.time()-t0:.0f}s -> violates={verdict['violates']} {verdict['categories']}")
    real_screen = model.txt_enc.screen_edit
    model.txt_enc.screen_edit = lambda *a, **k: SimpleNamespace(violates=False)

    cnt = {"img": 0, "txt": 0}
    def save(name, t): np.save(os.path.join(OUT, name + ".npy"), t.detach().cpu().float().numpy())
    h1 = tr.img_in.register_forward_pre_hook(
        lambda m, i: (save(f"img_in_{cnt['img']:02d}", i[0]), cnt.__setitem__("img", cnt["img"] + 1)) and None)
    h2 = tr.txt_norm.register_forward_pre_hook(
        lambda m, i: (save(f"txt_{cnt['txt']:02d}", i[0]), cnt.__setitem__("txt", cnt["txt"] + 1)) and None)
    t0 = time.time()
    imgs = generate_edits(model, [case["prompt"]], [case["refs"]], seeds=[SEED], steps=STEPS, cfg=CFG,
                          heights=[SIZE], widths=[SIZE], device="cpu")
    h1.remove(); h2.remove()
    model.txt_enc.screen_edit = real_screen
    imgs[0].save(os.path.join(OUT, "render.png"))
    json.dump(dict(case, seed=SEED, steps=STEPS, cfg=CFG, size=SIZE, repo=REPO),
              open(os.path.join(OUT, "case.json"), "w"), indent=1)
    log(f"[{n}-ref] render {(time.time()-t0)/60:.1f} min; img_in x{cnt['img']} txt x{cnt['txt']} -> {OUT}")
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".npy"): print(f"  {f:<14} {np.load(os.path.join(OUT, f)).shape}", flush=True)
log("DONE")

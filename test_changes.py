import pickle
import numpy as np

fluxtype = input("Enter fluxtype (SFR/no): ")
exp      = input("Enter experiment (poemma/ICgen2radio/grand200k): ")
d        = int(input("Enter dimension: "))
print((fluxtype, exp, d))

path_a = f"data/param/flux{fluxtype}_{exp}_dim{d}.pkl"
path_b = f"data/param_neuz_equals_injz/flux{fluxtype}_{exp}_dim{d}.pkl"

with open(path_a, "rb") as f: a = pickle.load(f)
with open(path_b, "rb") as f: b = pickle.load(f)

print(f"\n{'─'*60}")
print(f"  A: {path_a}")
print(f"  B: {path_b}")
print(f"{'─'*60}")

def summarize(label, obj, indent=2):
    pad = " " * indent
    if isinstance(obj, dict):
        print(f"{pad}{label}: dict with keys {list(obj.keys())}")
        for k, v in obj.items():
            summarize(k, v, indent + 2)
    elif isinstance(obj, np.ndarray):
        print(f"{pad}{label}: array shape={obj.shape} dtype={obj.dtype}  "
              f"min={obj.min():.4g}  max={obj.max():.4g}  mean={obj.mean():.4g}")
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{label}: {type(obj).__name__} len={len(obj)}")
    # else:
    #     print(f"{pad}{label}: {type(obj).__name__} = {obj}")

print("\n── A ──")
summarize("root", a)
print("\n── B ──")
summarize("root", b)

# ── numeric comparison ────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  COMPARISON")
print(f"{'─'*60}")

def compare(a_obj, b_obj, key="root"):
    if isinstance(a_obj, dict) and isinstance(b_obj, dict):
        for k in a_obj:
            if k in b_obj:
                compare(a_obj[k], b_obj[k], key=k)
            else:
                print(f"  [{k}] missing in B")
        for k in b_obj:
            if k not in a_obj:
                print(f"  [{k}] missing in A")
    elif isinstance(a_obj, np.ndarray) and isinstance(b_obj, np.ndarray):
        if a_obj.shape != b_obj.shape:
            print(f"  [{key}] shape mismatch: A={a_obj.shape}  B={b_obj.shape}")
            return
        diff = np.abs(a_obj - b_obj)
        rel  = diff / (np.abs(a_obj) + 1e-300)
        print(f"  [{key}]  max_abs_diff={diff.max():.4g}  "
              f"mean_abs_diff={diff.mean():.4g}  "
              f"max_rel_diff={rel.max():.4g}  "
              f"allclose={np.allclose(a_obj, b_obj)}")
    elif isinstance(a_obj, (int, float, np.floating, np.integer)):
        diff = abs(a_obj - b_obj)
        print(f"  [{key}]  A={a_obj:.6g}  B={b_obj:.6g}  diff={diff:.4g}")
    else:
        match = a_obj == b_obj
        print(f"  [{key}]  equal={match}  (A={a_obj!r}  B={b_obj!r})")

compare(a, b)
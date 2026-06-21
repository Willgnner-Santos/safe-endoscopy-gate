# -*- coding: utf-8 -*-
"""Authoritative audit from the frozen splits (source of truth for A4)."""
import pandas as pd, numpy as np, glob, os, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SPL = r"E:\Anonymous-Classificacao-Classes\splits"
LABELS = ["NORMAL","ALTERADO","SALIVA","LUZ","ENANTEMA","PÓLIPO","ÚLCERA",
          "EROSÃO","MICRONODULARIDADE","ECTASIA VASCULAR","NEOPLASIA"]
PATHO = ["ENANTEMA","PÓLIPO","ÚLCERA","EROSÃO","MICRONODULARIDADE","ECTASIA VASCULAR","NEOPLASIA"]

def load_fold(k):
    d = {}
    for part in ["train","val","test"]:
        d[part] = pd.read_csv(os.path.join(SPL, f"fold_{k}_train.csv".replace("train",part)))
    return d

# ---- 1. Union of all images (dedup by image_name) using fold_0 (train+val+test) ----
f0 = load_fold(0)
allrows = pd.concat([f0["train"], f0["val"], f0["test"]], ignore_index=True)
print("=== FOLD 0: sizes ===")
for p in ["train","val","test"]:
    print(f"  {p}: {len(f0[p])}")
print(f"  sum train+val+test: {len(allrows)}")
print(f"  unique image_name: {allrows['image_name'].nunique()}")
print(f"  unique ID: {allrows['ID'].nunique()}")

# ---- 2. Total distinct images in the ENTIRE dataset (sweeps all folds) ----
uni = set()
for k in range(5):
    fk = load_fold(k)
    for p in ["train","val","test"]:
        uni |= set(fk[p]["image_name"].tolist())
print(f"\n=== Distinct images summing the 5 folds: {len(uni)} ===")

# ---- 3. Binary: NORMAL x ALTERED complementarity ----
df = allrows.drop_duplicates("image_name").copy()
print(f"\n=== Deduplicated dataset (fold_0 union): {len(df)} images ===")
ct = df.groupby(["NORMAL","ALTERADO"]).size()
print("Cross-tab NORMAL x ALTERADO:")
print(ct)
nem = df[(df["NORMAL"]==0)&(df["ALTERADO"]==0)]
both = df[(df["NORMAL"]==1)&(df["ALTERADO"]==1)]
print(f"  neither-nor (0,0): {len(nem)}  | both (1,1): {len(both)}")
print(f"  ALTERADO=1: {int(df['ALTERADO'].sum())} ({100*df['ALTERADO'].mean():.2f}%)")
print(f"  NORMAL=1:   {int(df['NORMAL'].sum())} ({100*df['NORMAL'].mean():.2f}%)")

# ---- 4. has_file: how many images actually have a file ----
if "has_file" in df.columns:
    print(f"\n=== has_file ===\n{df['has_file'].value_counts(dropna=False)}")

# ---- 5. LOCAL: what is it? ----
print(f"\n=== LOCAL (values) ===\n{df['LOCAL'].value_counts(dropna=False)}")
print("LOCAL x NORMAL/ALTERADO:")
print(df.groupby("LOCAL")[["NORMAL","ALTERADO"]].mean().round(3))

# ---- 6. group_id: structure ----
print(f"\n=== group_id ===")
print(f"  unique groups: {df['group_id'].nunique()} over {len(df)} images")
gsz = df.groupby("group_id").size()
print(f"  group size: min {gsz.min()}, max {gsz.max()}, mean {gsz.mean():.2f}")
print(f"  groups with >1 image: {(gsz>1).sum()}")
print(f"  group sizes distribution:\n{gsz.value_counts().sort_index().head(15)}")

# ---- 7. Leakage check: does group_id cross train/test in any fold? ----
print(f"\n=== LEAKAGE by group_id (each fold) ===")
for k in range(5):
    fk = load_fold(k)
    gtr = set(fk["train"]["group_id"]); gva = set(fk["val"]["group_id"]); gte = set(fk["test"]["group_id"])
    print(f"  fold {k}: groups tr-te={len(gtr&gte)}, tr-va={len(gtr&gva)}, va-te={len(gva&gte)} | "
          f"img tr-te={len(set(fk['train']['image_name'])&set(fk['test']['image_name']))}")

# ---- 8. Prevalence by label (in the dedup dataset) ----
print(f"\n=== Prevalence by label (dedup, n={len(df)}) ===")
rows=[]
for L in LABELS:
    pos=int(df[L].sum()); n=len(df); prev=100*pos/n
    ir = (n-pos)/pos if pos>0 else float('inf')
    rows.append((L,pos,round(prev,2),round(ir,1)))
    print(f"  {L:<20} pos={pos:<5} prev={prev:5.2f}%  IR={ir:.1f}")

# ---- 9. Binary target: pathology among 'NORMAL' (base miss-rate) ----
print(f"\n=== Pathology present in images NORMAL=1 (sanity for gate) ===")
norm = df[df["NORMAL"]==1]
patho_in_normal = norm[PATHO].sum(axis=1)
print(f"  NORMAL images with ≥1 labeled pathology: {(patho_in_normal>0).sum()} / {len(norm)}")
print(f"=== ALTERADO=1 without any labeled pathology ===")
alt = df[df["ALTERADO"]==1]
patho_in_alt = alt[PATHO].sum(axis=1)
print(f"  ALTERADO without pathology: {(patho_in_alt==0).sum()} / {len(alt)}")

# save summary
summary = {
    "n_fold0_union": int(len(allrows)),
    "n_unique_image_name_fold0": int(allrows['image_name'].nunique()),
    "n_unique_all_folds": int(len(uni)),
    "n_dedup": int(len(df)),
    "altered_pos": int(df['ALTERADO'].sum()),
    "normal_pos": int(df['NORMAL'].sum()),
    "nem_nem": int(len(nem)),
    "both": int(len(both)),
    "local_counts": {str(k):int(v) for k,v in df['LOCAL'].value_counts().items()},
    "n_groups": int(df['group_id'].nunique()),
    "prevalence": {L:{"pos":int(df[L].sum()),"prev_pct":round(100*df[L].mean(),2)} for L in LABELS},
}
with open(os.path.join(SPL,"..","_audit_summary.json"),"w",encoding="utf-8") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=2)
print("\n[OK] summary saved to _audit_summary.json")

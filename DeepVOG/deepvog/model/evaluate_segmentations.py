# evaluate_segmentations.py
# ------------------------------------------------------------
# Realizado por: Stefanny Arboleda
# ------------------------------------------------------------
# Archivo que sirve para evaluar el desempeño de los modelos, si se
# elige traditional recuerdese pasar las carpetas con las máscaras ya generadas 
# es decir, ejecútese SAC_Traditional_Segmentation.py primero,
# si se elige deepvog recuérdese ejecutar primero batch_segment_pupil.py 
# ------------------------------------------------------------
import argparse
from pathlib import Path
import numpy as np
import skimage.io as ski
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, roc_auc_score
from collections import defaultdict
import csv
from skimage.transform import resize
import re
from typing import Optional, Tuple, List
import os

# ------------------------
# Canonical stem helpers
# ------------------------
STEM_RE = re.compile(
    r"^S1(?P<xxx>\d{3})(?P<eye>[LR])(?P<zz>\d{2})(?:[ _-]?(?:GT))?$",
    re.IGNORECASE
)

def canonical_stem_from_name(name_wo_ext: str) -> Optional[str]:
    m = STEM_RE.fullmatch(name_wo_ext)
    if not m: return None
    xxx = m.group("xxx")
    eye = m.group("eye").upper()
    zz  = m.group("zz")
    return f"S1{xxx}{eye}{zz}"

def read_mask_binary(p: Path):
    m = ski.imread(str(p))
    if m.ndim == 3:
        m = m[...,0]
    # >127 => 1
    m = (m.astype(np.float32) > 127.5).astype(np.uint8)
    return m

def read_prob_float01(p: Path):
    x = ski.imread(str(p)).astype(np.float32)
    if x.ndim == 3:
        x = x[...,0]
    if x.max() > 1.0:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)

def specificity_score(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    denom = (tn + fp)
    return tn / denom if denom > 0 else 0.0

# ------------------------
# Pair collectors
# ------------------------

def collect_pairs_deepvog(gt_root: Path, pred_root: Path,
                          exts=(".png",".jpg",".jpeg",".bmp",".tif",".tiff")) -> List[Tuple[Path, Path, Optional[Path], str]]:
    """
    GT: acepta S1xxxyzz(.+ opcional como _GT) con cualquier extensión en gt_root.
    Pred (DeepVOG): busca pred_root/bin/S1xxxyzz.png y pred_root/prob/S1xxxyzz.png
    Devuelve (gt_path, pred_bin_path, pred_prob_path | None, stem)
    """
    # 1) Indexa GT por stem canónico
    gt_by_stem = {}
    skipped = 0
    for p in gt_root.rglob("*"):
        if p.suffix.lower() in exts:
            stem = canonical_stem_from_name(p.stem)
            if stem is None:
                skipped += 1
                continue
            gt_by_stem.setdefault(stem, p)

    # 2) Construye pares
    pairs = []
    missing_bin = 0; missing_prob = 0
    for stem, gt_path in sorted(gt_by_stem.items()):
        pred_bin  = pred_root / "bin"  / f"{stem}.png"
        pred_prob = pred_root / "prob" / f"{stem}.png"
        if not pred_bin.exists():
            missing_bin += 1
        if not pred_prob.exists():
            pred_prob_path = None
            missing_prob += 1
        else:
            pred_prob_path = pred_prob
        pairs.append((gt_path, pred_bin, pred_prob_path, stem))

    print(f"[collect_pairs_deepvog] GT válidos: {len(gt_by_stem)} | GT ignorados: {skipped}")
    print(f"[collect_pairs_deepvog] pares listados: {len(pairs)} | bin faltantes: {missing_bin} | prob faltantes: {missing_prob}")
    return pairs

# Para TRADICIONAL: pred_root/xxx/yy/*_mask.png
NN_RE_TDT = re.compile(r".*?(\d{1,2})(?!.*\d)")  # último grupo de 1-2 dígitos en el nombre

def stem_from_traditional_pred(pred_path: Path, pred_root: Path) -> Optional[str]:
    """
    Construye 'S1xxxyzz' desde pred_path bajo pred_root/xxx/yy/*_mask.png
    - xxx = carpeta -3 (exactamente 3 dígitos)
    - yy  = carpeta -2 (L o R)
    - zz  = último grupo numérico (1–2 dígitos) del nombre (sin extensión), zfill a 2
    """
    try:
        rel = pred_path.relative_to(pred_root)
    except ValueError:
        rel = pred_path
    parts = rel.parts
    if len(parts) < 3:
        return None

    xxx = parts[-3]           # '001'..'249'
    eye = parts[-2].upper()   # 'L' o 'R'
    stem_name = pred_path.stem

    if not re.fullmatch(r"\d{3}", xxx):
        return None
    if eye not in {"L", "R"}:
        return None

    m = NN_RE_TDT.fullmatch(stem_name)
    if not m:
        # como fallback, busca último grupo numérico en cualquier parte
        m = re.search(r"(\d{1,2})(?!.*\d)", stem_name)
        if not m:
            return None
    zz = m.group(1)
    if len(zz) == 1:
        zz = "0" + zz
    elif len(zz) > 2:
        return None

    return f"S1{xxx}{eye}{zz}"

def collect_pairs_traditional(gt_root: Path, pred_root: Path,
                              exts_gt=(".png",".jpg",".jpeg",".bmp",".tif",".tiff")) -> List[Tuple[Path, Path, None, str]]:
    """
    GT: como arriba (indexado por S1xxxyzz).
    Pred (Tradicional): recorre pred_root/**/**/*_mask.png y construye S1xxxyzz desde la ruta.
    Devuelve (gt_path, pred_bin_path, None, stem)
    """
    # 1) Indexa GT por stem
    gt_by_stem = {}
    skipped = 0
    for p in gt_root.rglob("*"):
        if p.suffix.lower() in exts_gt:
            stem = canonical_stem_from_name(p.stem)
            if stem is None:
                skipped += 1
                continue
            gt_by_stem.setdefault(stem, p)

    # 2) Recoge predicciones binarias tradicionales
    pred_bin_files = [p for p in pred_root.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp") and p.stem.endswith("_mask")]
    pairs = []
    no_stem = 0; no_gt = 0
    for pb in pred_bin_files:
        st = stem_from_traditional_pred(pb, pred_root)
        if not st:
            no_stem += 1
            continue
        if st not in gt_by_stem:
            no_gt += 1
            continue
        pairs.append((gt_by_stem[st], pb, None, st))

    print(f"[collect_pairs_traditional] GT válidos: {len(gt_by_stem)} | GT ignorados: {skipped}")
    print(f"[collect_pairs_traditional] pred bin encontradas: {len(pred_bin_files)} | pares válidos: {len(pairs)} | sin stem: {no_stem} | sin GT: {no_gt}")
    return pairs

# ------------------------
# Evaluación
# ------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_root", required=True, help="Carpeta con GT (máscaras binarias)")
    ap.add_argument("--pred_root", required=True, help="Carpeta raíz de predicciones")
    ap.add_argument("--pred_layout", choices=["deepvog","traditional"], default="deepvog",
                    help="Estructura de predicciones: 'deepvog' o 'traditional'")
    # NUEVO: carpeta de salida
    ap.add_argument("--out_dir", default=".", help="Carpeta donde guardar métricas (se crea si no existe)")
    ap.add_argument("--out_csv", default="metrics_per_image.csv")
    ap.add_argument("--out_summary", default="metrics_summary.csv")
    args = ap.parse_args()

    gt_root = Path(args.gt_root)
    pred_root = Path(args.pred_root)

    # Asegura carpeta de salida
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if args.pred_layout == "deepvog":
        pairs = collect_pairs_deepvog(gt_root, pred_root)
    else:
        pairs = collect_pairs_traditional(gt_root, pred_root)

    if len(pairs) == 0:
        print("No se encontraron pares GT/Pred.")
        return

    rows = []
    agg = defaultdict(list)

    for gt_path, pred_bin_path, pred_prob_path, stem in pairs:
        if not pred_bin_path.exists():
            # si no hay binaria, saltamos
            continue

        gt = read_mask_binary(gt_path)
        pred_bin = read_mask_binary(pred_bin_path)

        # Ajuste de tamaño si difiere
        if gt.shape != pred_bin.shape:
            pred_bin = resize(pred_bin.astype(np.float32), gt.shape, order=0, preserve_range=True, anti_aliasing=False)
            pred_bin = (pred_bin > 0.5).astype(np.uint8)

        y_true = gt.flatten()
        y_pred = pred_bin.flatten()

        # Métricas binarias
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        se  = (np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)) if np.any(y_true == 1) else 0.0
        sp  = specificity_score(y_true, y_pred)
        iou = jaccard_score(y_true, y_pred, zero_division=0)

        # AUC (solo si hay mapa de probas)
        auc = None
        if (pred_prob_path is not None) and np.any(y_true == 1) and np.any(y_true == 0):
            if pred_prob_path.exists():
                prob = read_prob_float01(pred_prob_path)
                if prob.shape != gt.shape:
                    prob = resize(prob.astype(np.float32), gt.shape, order=1, preserve_range=True, anti_aliasing=True)
                    prob = np.clip(prob, 0.0, 1.0)
                try:
                    auc = roc_auc_score(y_true, prob.flatten())
                except ValueError:
                    auc = None

        rows.append({"id": stem, "ACC": acc, "F1": f1, "SE": se, "SP": sp, "IoU": iou, "AUC": (float(auc) if auc is not None else None)})
        agg["ACC"].append(acc); agg["F1"].append(f1); agg["SE"].append(se); agg["SP"].append(sp); agg["IoU"].append(iou)
        if auc is not None: agg["AUC"].append(float(auc))

    # RUTAS de salida ahora dentro de out_dir
    out_csv_path = out_dir / args.out_csv
    with out_csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","ACC","F1","SE","SP","IoU","AUC"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "ACC": float(np.mean(agg["ACC"])) if agg["ACC"] else None,
        "F1":  float(np.mean(agg["F1"])) if agg["F1"] else None,
        "SE":  float(np.mean(agg["SE"])) if agg["SE"] else None,
        "SP":  float(np.mean(agg["SP"])) if agg["SP"] else None,
        "IoU": float(np.mean(agg["IoU"])) if agg["IoU"] else None,
        "AUC": float(np.mean(agg["AUC"])) if agg["AUC"] else None,
        "N_images": len(rows),
        "N_with_AUC": len(agg["AUC"]),
    }

    out_sum_path = out_dir / args.out_summary
    with out_sum_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    print(f"Per-imagen: {out_csv_path}")
    print(f"Resumen:    {out_sum_path}")
    print("Promedios:", summary)

if __name__ == "__main__":
    main()
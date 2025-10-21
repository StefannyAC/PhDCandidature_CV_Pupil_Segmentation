# batch_segment_pupil.py
# ------------------------------------------------------------
# Realizado por: Stefanny Arboleda
# ------------------------------------------------------------
# Este archivo sirve para crear las evaluaciones de las segmentaciones
# correspondientes al método DeepLearning (DeepVOG), además debe estar
# en la misma carpeta que los archivos pupil_analysis.py (helper para 
# guardar csv con los círculos), DeepVOG_model.py (modelo)
# pues son archivos base con funciones que se estarán usando aquí.
# ------------------------------------------------------------
import argparse
from pathlib import Path
import csv
import numpy as np
import skimage.io as ski
from skimage.transform import resize
from tqdm import tqdm
from DeepVOG_model import load_DeepVOG
import re
from typing import Optional
from pupil_analysis import process_batch_analysis, ensure_dir

EXTS_IMG = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
EXTS_GT  = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

def load_image_as_float01(p: Path):
    img = ski.imread(str(p)).astype(np.float32)
    if img.ndim == 3 and img.shape[2] > 1:
        # usa el canal 0 para mantener compatibilidad con tu prueba
        img = img[:, :, :1]
    if img.ndim == 2:
        img = img[..., None]
    if img.max() > 1.0:
        img = img / 255.0
    return img  # HxWx1

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

STEM_GT_RE = re.compile(
    r"^S1(?P<xxx>\d{3})(?P<eye>[LR])(?P<zz>\d{2})(?:[ _-]?(?:GT))?$",
    re.IGNORECASE
)

def stem_from_casia_path_S1(img_path: Path, images_root: Path) -> Optional[str]:
    """
    Convierte CASIA '.../xxx/y/zz.tiff' -> 'S1xxxyzz' (xxx=3 dígitos, zz=2 dígitos).
    """
    try:
        rel = img_path.relative_to(images_root)
    except ValueError:
        rel = img_path

    parts = rel.parts
    # Esperamos .../<xxx>/<L|R>/<zz>.<ext>
    if len(parts) < 3:
        return None

    xxx = parts[-3]            # '001'..'249'
    eye = parts[-2].upper()    # 'L' o 'R'
    zz  = Path(parts[-1]).stem # '01', '02', etc.

    if not re.fullmatch(r"\d{1,3}", xxx):
        return None
    if eye not in {"L", "R"}:
        return None
    if not re.fullmatch(r"\d{1,3}", zz):
        # si hay algo raro, intenta tomar dígitos finales
        m = re.search(r"(\d{1,3})$", zz)
        if not m:
            return None
        zz = m.group(1)

    xxx = xxx.zfill(3)
    zz  = zz.zfill(2)
    return f"S1{xxx}{eye}{zz}"

def stem_from_gt_filename_S1(gt_path: Path) -> Optional[str]:
    """
    Acepta:
      S1xxxyzz.png
      S1xxxyzz_GT.png
      S1xxxyzz-mask.tif
    y devuelve exactamente 'S1xxxyzz' preservando ceros.
    """
    name_wo_ext = gt_path.stem
    m = STEM_GT_RE.fullmatch(name_wo_ext)
    if not m:
        return None
    xxx = m.group("xxx")                # siempre 3 dígitos, p.ej. '001', '011', '111'
    eye = m.group("eye").upper()        # 'L' o 'R'
    zz  = m.group("zz")                 # siempre 2 dígitos, p.ej. '01', '02'
    return f"S1{xxx}{eye}{zz}"

def collect_pairs(images_root: Path, gt_root: Path):
    """
    Empareja SOLO casos con GT usando el stem 'S1xxxyzz'.
    Guarda (img_path, gt_path, stem).
    """
    EXTS_IMG = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    EXTS_GT  = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

    imgs = [p for p in images_root.rglob("*") if p.suffix.lower() in EXTS_IMG]
    gts  = [p for p in gt_root.rglob("*")    if p.suffix.lower() in EXTS_GT]

    # indexa GT por stem S1xxxyzz
    gt_by_stem = {}
    for gp in gts:
        st = stem_from_gt_filename_S1(gp)
        if st:
            gt_by_stem[st] = gp

    pairs = []
    no_stem_imgs, no_match_imgs = 0, 0

    for ip in imgs:
        st = stem_from_casia_path_S1(ip, images_root)
        if not st:
            no_stem_imgs += 1
            continue
        if st not in gt_by_stem:
            no_match_imgs += 1
            continue
        pairs.append((ip, gt_by_stem[st], st))

    print(f"[collect_pairs] imágenes: {len(imgs)} | GT: {len(gts)}")
    print(f"[collect_pairs] pares válidos: {len(pairs)}")
    print(f"[collect_pairs] imgs sin stem CASIA: {no_stem_imgs}")
    print(f"[collect_pairs] imgs sin GT (stem no encontrado): {no_match_imgs}")

    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", required=True, help="Carpeta raíz con imágenes")
    ap.add_argument("--gt_root", required=True, help="Carpeta con GT (máscaras binarias)")
    ap.add_argument("--out_dir", required=True, help="Carpeta de salida")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--resize_h", type=int, default=240)
    ap.add_argument("--resize_w", type=int, default=320)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--save_prob_png", action="store_true")
    ap.add_argument("--pairs_csv", default="pairs.csv", help="CSV de pares (se genera aquí)")
    args = ap.parse_args()

    images_root = Path(args.images_root)
    gt_root     = Path(args.gt_root)
    out_dir     = Path(args.out_dir)
    prob_dir = out_dir / "prob"
    bin_dir  = out_dir / "bin"
    ensure_dir(prob_dir); ensure_dir(bin_dir)
    coord_csv_path = out_dir / "pupil_coords_deepvog.csv"

    pairs = collect_pairs(images_root, gt_root)
    if not pairs:
        print("No se encontraron pares (imagen,GT) por stem.")
        return

    # Guarda los pares para reproducibilidad
    pairs_csv_path = out_dir / args.pairs_csv
    with pairs_csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path","gt_path","stem"])
        for ip, gp, stem in pairs:
            w.writerow([str(ip), str(gp), stem])

    print(f"Pares encontrados: {len(pairs)} (CSV: {pairs_csv_path})")

    with coord_csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "status", "cx_dl", "cy_dl", "r_dl"]) # Encabezado del CSV de coordenadas

    model = load_DeepVOG()
    H, W = args.resize_h, args.resize_w
    B = args.batch_size
    thr = float(args.threshold)

    # Lotes por eficiencia
    for i in tqdm(range(0, len(pairs), B), desc="Segmentando solo pares con GT"):
        batch = pairs[i:i+B]
        batch_imgs, orig_shapes, stems, img_paths_orig = [], [], [], []

        for ip, _, stem in batch:
            img = load_image_as_float01(ip)
            orig_shapes.append(img.shape[:2])
            img_resized = resize(img, (H, W, 1), anti_aliasing=True, preserve_range=True).astype(np.float32)
            img_3 = np.repeat(img_resized, 3, axis=2)  # HxWx3
            batch_imgs.append(img_3)
            stems.append(stem)
            img_paths_orig.append(ip)

        x = np.stack(batch_imgs, axis=0)  # BxHxWx3
        pred = model.predict(x)           # esperado (B,H,W,C)
        # Soporta C=1 (sigmoid) o C=2 (softmax canal 1 = pupila)
        if pred.shape[-1] == 1:
            prob_pupil = pred[..., 0]
        else:
            prob_pupil = pred[..., 1]

        batch_analysis_data = []
        for j, stem in enumerate(stems):
            h0, w0 = orig_shapes[j]
            prob_resized = resize(prob_pupil[j], (h0, w0), anti_aliasing=True, preserve_range=True).astype(np.float32)
            bin_mask = (prob_resized >= thr).astype(np.uint8) * 255

            if args.save_prob_png:
                prob_u8 = np.clip(prob_resized * 255.0, 0, 255).astype(np.uint8)
                ski.imsave(str(prob_dir / f"{stem}.png"), prob_u8)

            ski.imsave(str(bin_dir / f"{stem}.png"), bin_mask)
            # Recolectar datos para el análisis
            batch_analysis_data.append({
                'stem': stem,
                'bin_mask': bin_mask,
                'img_path': img_paths_orig[j]
            })

        # 4. EXTRACCIÓN DE COORDENADAS Y GUARDADO CSV
        csv_results = process_batch_analysis(batch_analysis_data, out_dir)
        
        with coord_csv_path.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerows(csv_results)

    print(f"Listo. Guardado en:\n - Prob: {prob_dir} (si activaste --save_prob_png)\n - Bin:  {bin_dir}\n - Pares: {pairs_csv_path}")

if __name__ == "__main__":
    main()
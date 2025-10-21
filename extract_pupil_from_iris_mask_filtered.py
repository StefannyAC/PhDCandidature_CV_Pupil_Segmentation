# extract_pupil_from_iris_mask_filtered.py
# ------------------------------------------------------------
# Realizado por: Stefanny Arboleda
# ------------------------------------------------------------
# Archivo que genera las máscaras de GT para todo el dataset, eligiendo como 
# GT válido por defecto aquellos que tienen OperatorB en el nombre. Se debe 
# ejecutar primero antes de cualquier otro script, pues es necesario construir
# primero el dataset y sus GT's. El archivo evaluate_segmentations.py ya hace
# un filtrado sobre los casos en que un archivo original no tiene GT.  
# ------------------------------------------------------------
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import re

MASK_RE = re.compile(r"""
    Operator\s*                # 'Operator' + espacios
    (?P<op>[ABab])             # A o B (case-insensitive)
    [\s_-]*                    # separadores varios
    (?:S\s*1)?                 # 'S1' opcional con espacio
    [\s_-]*                    # separadores varios
    (?P<xxx>\d+)               # primer bloque numérico
    \s*
    (?P<y>[LR])                # L o R
    \s*
    (?P<zz>\d+)                # segundo bloque numérico
    [^.\n]*                    # sufijos opcionales como '_mask'
    \.(?:png|jpe?g|bmp|tif|tiff)$  # extensiones comunes
""", re.IGNORECASE | re.VERBOSE)

def parse_mask_name(p: Path):
    m = MASK_RE.search(p.name)   
    if not m:
        return None
    op  = m.group("op").upper()
    xxx = m.group("xxx").lstrip("0") or "0"
    y   = m.group("y").upper()
    zz  = m.group("zz").lstrip("0") or "0"
    return op, xxx, y, zz

def binarize(img):
    img = np.asarray(img).astype(np.float32)
    thr = 127.5 if img.max() > 1.0 else 0.5
    return (img > thr).astype(np.uint8)

def morph(mask, op="open", it=1):
    s = ndi.generate_binary_structure(2,1)
    out = mask.astype(bool)
    for _ in range(max(0,it)):
        out = getattr(ndi, f"binary_{op}")(out, structure=s)
    return out.astype(np.uint8)

def disk(r):
    Y,X = np.ogrid[-r:r+1, -r:r+1]
    return ((X*X + Y*Y) <= r*r).astype(np.uint8)

def extract_pupil(iris, max_r=3, ratio_min=0.01, ratio_max=0.60):
    iris = iris.astype(np.uint8)
    iris_area = iris.sum()

    # intento directo
    filled = ndi.binary_fill_holes(iris.astype(bool)).astype(np.uint8)
    pupil  = (filled & (~iris.astype(bool))).astype(np.uint8)
    if pupil.sum() > 0:
        ratio = pupil.sum() / max(iris_area, 1)
        if ratio_min <= ratio <= ratio_max:
            return pupil

    # cierres leves r=1..max_r para sellar roturas pequeñas
    for r in range(1, max_r+1):
        closed = ndi.binary_closing(iris, structure=disk(r)).astype(np.uint8)
        filled = ndi.binary_fill_holes(closed.astype(bool)).astype(np.uint8)
        pupil  = (filled & (~closed.astype(bool))).astype(np.uint8)
        if pupil.sum() > 0:
            ratio = pupil.sum() / max(iris_area, 1)
            if ratio_min <= ratio <= ratio_max:
                return True

    # si llegamos aquí: no hay pupila confiable -> descartar
    return pupil

def is_nonempty_mask(mask, thr=0.5):
    """Devuelve True si la máscara tiene algún píxel > thr (no negra)."""
    return np.any(mask > thr)

def main(inp, out):
    inp, out = Path(inp), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ims = [p for p in inp.rglob("*") if p.suffix.lower() in [".png",".jpg",".bmp",".tif",".tiff"]]

    saved, skipped_nonmatch, skipped_empty = 0, 0, 0

    for p in ims:
        parsed = parse_mask_name(p)
        if not parsed:
            skipped_nonmatch += 1
            continue

        op, _, _,_ = parsed
        if op != "B":
            # Por si cambiamos el regex a [AB] en el futuro; ahora mismo el regex ya filtra B
            skipped_nonmatch += 1
            continue

        # Carga y binariza
        with Image.open(p) as im:
            im = im.convert("L")
        mask = binarize(im)

        # Extraer pupila
        pupil = extract_pupil(mask, max_r=3, ratio_min=0.01, ratio_max=0.60)

        if not is_nonempty_mask(pupil):
            skipped_empty += 1
            continue

        # Guarda con nombre claro
        clean_name = re.sub(r"^OperatorB[\s_-]*", "", p.stem, flags=re.IGNORECASE)
        out_name = f"{clean_name}_GT.png"
        Image.fromarray((pupil*255).astype(np.uint8)).save(out / out_name)
        saved += 1

    print(f"Guardadas: {saved}")
    print(f"Saltadas (nombre): {skipped_nonmatch}")
    print(f"Saltadas (pupila vacía): {skipped_empty}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    main(args.masks_dir, args.out_dir)
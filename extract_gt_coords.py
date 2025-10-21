# extract_gt_coords.py
# ------------------------------------------------------------
# Realizado por: Stefanny Arboleda
# ------------------------------------------------------------
# Archivo para crear el csv con los cx,cy,r del gt de cada imagen
# necesario para poder realizar los dibujos sobre las imágenes
# de los círculos base y así poder visualizarlos.
# ------------------------------------------------------------
import argparse
import csv
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from typing import Optional

def extract_pupil_circle_from_mask(mask_bool):
    """
    Extrae las coordenadas del círculo (cx, cy, r) a partir de una máscara binaria 
    usando análisis de contornos y minEnclosingCircle.
    """
    # Convertir máscara booleana a formato uint8 (0, 255)
    mask = (mask_bool * 255).astype(np.uint8)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None # Fallo en la segmentación
    
    # Seleccionar el contorno más grande (asumiendo que es la pupila)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calcular el círculo que mejor encierra (x, y, r)
    ((x, y), r) = cv2.minEnclosingCircle(largest_contour)
    
    # Devolvemos el centro y el radio estimados
    cx, cy = float(x), float(y)
    
    return cx, cy, r

# ----------------------------------------------------------------------
# FUNCION PRINCIPAL
# ----------------------------------------------------------------------
def process_gt_folder(gt_root: Path, output_csv: Path):
    """
    Lee máscaras de GT, extrae coordenadas circulares y guarda en CSV.

    Args:
        gt_root: Carpeta con los archivos de máscara de Ground Truth.
        output_csv: Ruta completa para guardar el archivo CSV de coordenadas.
    """
    EXTS_GT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    
    # Buscar todos los archivos que coincidan con las extensiones de imagen
    gt_paths = [p for p in gt_root.rglob("*") if p.suffix.lower() in EXTS_GT]
    
    if not gt_paths:
        print(f"Error: No se encontraron archivos de máscara de GT en {gt_root}")
        return

    # Asegurar que el directorio de salida exista
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Encontradas {len(gt_paths)} máscaras de GT. Procesando...")

    # Abrir el archivo CSV para escritura en streaming
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "cx_gt", "cy_gt", "r_gt", "status"])

        for gt_path in tqdm(gt_paths, desc="Extrayendo coordenadas GT"):
            
            # 1. Obtener el STEM (nombre de la imagen sin _GT y sin extensión)
            # Ejemplo: S1001L01_GT.png -> S1001L01
            stem = gt_path.stem.upper()
            if stem.endswith("_GT"):
                final_stem = stem[:-3]
            else:
                final_stem = stem
            
            # 2. Cargar la máscara
            # cv2.imread(IMREAD_GRAYSCALE) asegura que cargamos un canal de 0-255
            mask_u8 = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            
            if mask_u8 is None:
                writer.writerow([final_stem, "read_error", "", "", ""])
                continue

            # Convertir a booleano (True/False). Asumimos que la pupila es el píxel blanco (255)
            mask_bool = (mask_u8 > 127)
            
            # 3. Extraer coordenadas del círculo
            cx, cy, r = extract_pupil_circle_from_mask(mask_bool)
            
            # 4. Escribir fila CSV
            if cx is None:
                row = [final_stem, "fail", "", "", ""]
            else:
                row = [final_stem, f"{cx:.2f}", f"{cy:.2f}", f"{r:.2f}", "ok"]
            
            writer.writerow(row)

    print(f"\n Listo. Coordenadas de Ground Truth guardadas en: {output_csv}")

# ----------------------------------------------------------------------
# PUNTO DE ENTRADA
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrae coordenadas circulares (cx, cy, r) de las máscaras de Ground Truth.")
    parser.add_argument("--gt_root", type=str, required=True, help="Ruta a la carpeta raíz que contiene las máscaras de GT (ej: Casia-Iris-Interval-masks-pupils).")
    parser.add_argument("--output_csv", type=str, default="gt_pupil_coordinates.csv", help="Nombre del archivo CSV de salida.")
    args = parser.parse_args()

    process_gt_folder(Path(args.gt_root), Path(args.output_csv))
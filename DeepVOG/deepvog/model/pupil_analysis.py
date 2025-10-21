# pupil_analysis.py
# ------------------------------------------------------------
# Realizado por: Stefanny Arboleda
# ------------------------------------------------------------
import cv2
import numpy as np
from pathlib import Path
import csv

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

def process_batch_analysis(batch_data, out_dir: Path):
    """
    Procesa los datos de un lote: guarda originales y extrae coordenadas.

    Args:
        batch_data: Lista de diccionarios que contienen 'stem', 'bin_mask' y 'img_path'.
        out_dir: Directorio base de salida.
    
    Returns:
        Lista de filas CSV con (stem, cx, cy, r, status).
    """
    csv_rows = []
    
    # Definir subdirectorios
    orig_dir = out_dir / "circles"
    ensure_dir(orig_dir)
    
    for item in batch_data:
        stem = item['stem']
        mask = item['bin_mask']
        img_path = item['img_path']
        
        # 1. Extraer coordenadas
        cx, cy, r = extract_pupil_circle_from_mask(mask)
        
        # 2. Guardar imagen original para referencia (copiando el archivo)
        # La imagen original se carga como RGB/BGR para poder dibujar sobre ella si es necesario
        try:
            img_orig = cv2.imread(str(img_path))
            # Opcional: dibujar el círculo para inspección visual
            if cx is not None:
                 cv2.circle(img_orig, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
                 cv2.circle(img_orig, (int(cx), int(cy)), 2, (0, 0, 255), -1)
            
            cv2.imwrite(str(orig_dir / f"{stem}.png"), img_orig)
        except Exception as e:
            print(f"Error al guardar imagen original/overlay {stem}: {e}")

        # 3. Preparar fila CSV
        if cx is None:
            row = (stem, "fail", "", "", "")
        else:
            row = (stem, "ok", f"{cx:.2f}", f"{cy:.2f}", f"{r:.2f}")
        
        csv_rows.append(row)
        
    return csv_rows

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
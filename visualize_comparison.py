# visualize_comparison.py
# ------------------------------------------------------------
# Realizado por: Stefanny Arboleda
# ------------------------------------------------------------
# Archivo para generar las visualizaciones de los círculos para cada método
# Es necesario tener los archivos csv de cada método (incluido el GT) para
# su correcto funcionamiento.
# ------------------------------------------------------------
import argparse
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# ======================================================================
# CONFIGURACIÓN
# ======================================================================
# Columnas esperadas en cada CSV
COLS_TDT = {'stem': str, 'relative_path': str, 'cx': float, 'cy': float, 'r': float}
COLS_DL = {'stem': str, 'cx_dl': float, 'cy_dl': float, 'r_dl': float}
COLS_GT = {'stem': str, 'cx_gt': float, 'cy_gt': float, 'r_gt': float}

# Definiciones de colores para los círculos (BGR de OpenCV)
COLOR_TDT = (0, 0, 255)  # Rojo - TDT (Tradicional)
COLOR_DL  = (255, 0, 0)  # Azul - DL (Deep Learning / DeepVOG)
COLOR_GT  = (0, 255, 0)  # Verde - GT (Ground Truth)

# ======================================================================
# FUNCIONES AUXILIARES
# ======================================================================

def load_data(csv_path: Path, col_defs: dict, prefix: str):
    """Carga un CSV, crea/renombra columnas, limpia fallos y establece el índice 'stem'."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Archivo CSV no encontrado en {csv_path}")
        return None

    # --- 1. Creación del STEM para el CSV TDT ---
    if prefix == 'tdt':
        if 'relative_path' not in df.columns:
            print(f"Error TDT: Columna 'relative_path' no encontrada en {csv_path}. ¡Necesaria para crear el 'stem'!")
            return None
            
        # Extraer 'S1xxxyzz' del path (ej: '001/L/S1001L03.jpg' -> 'S1001L03')
        # Utilizamos una función lambda para extraer el nombre del archivo y luego el stem (sin extensión).
        # Esto asume que el nombre del archivo ya contiene el stem S1...
        df['stem'] = df['relative_path'].apply(
            lambda p: Path(p).stem.upper()
        )
        
    # --- 2. Renombrar y Conversión de Tipos (General) ---
    rename_map = {}
    for col, dtype in col_defs.items():
        # Lógica para renombrar columnas sin prefijo (ej: 'cx' a 'cx_dl') en DL/GT
        if prefix != 'tdt' and col.startswith('cx_') and col[3:] in df.columns:
             rename_map[col[3:]] = col 
        
        # Si la columna 'stem' no está definida en col_defs, no la intentamos renombrar

    if rename_map:
         df = df.rename(columns=rename_map)

    # Convertir a tipos numéricos, forzando NaN en errores
    # Creamos la lista de columnas a mantener, incluyendo 'stem' y 'relative_path'
    cols_to_keep = ['stem', 'relative_path'] if prefix == 'tdt' else ['stem']
    cols_to_keep.extend([c for c in col_defs if c not in ['stem', 'relative_path']])
    
    # Filtrar solo las columnas que realmente existen
    df = df[[c for c in cols_to_keep if c in df.columns]].copy()

    for col, dtype in col_defs.items():
        if col in df.columns and dtype == float:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Asegurarse del tipo del stem y establecerlo como índice para la unión
    df['stem'] = df['stem'].astype(str)
    
    # Retornar el DataFrame indexado por 'stem'
    return df.set_index('stem')

def draw_circle_on_image(img, cx, cy, r, color, label=""):
    """Dibuja un círculo y una etiqueta en la imagen."""
    if pd.notna(cx) and pd.notna(cy) and pd.notna(r) and r > 0:
        center = (int(cx), int(cy))
        radius = int(r)
        
        # Dibujar círculo
        cv2.circle(img, center, radius, color, 2)
        # Dibujar centro
        cv2.circle(img, center, 3, color, -1)
        
        # Opcional: Escribir etiqueta (Ej: 'GT')
        if label:
            cv2.putText(img, label, (center[0] + radius + 5, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_comparison(data, img_path, out_path, prefix1, prefix2):
    """Carga la imagen y dibuja los círculos de dos métodos para una imagen."""
    # 1. Cargar imagen original (en color para dibujar)
    img = cv2.imread(str(img_path))
    if img is None:
        print(f" No se pudo cargar la imagen: {img_path}")
        return

    # 2. Dibujar Círculo 1 (Método 1)
    cx1 = data[f'cx_{prefix1}'] if prefix1 != 'tdt' else data['cx']
    cy1 = data[f'cy_{prefix1}'] if prefix1 != 'tdt' else data['cy']
    r1  = data[f'r_{prefix1}'] if prefix1 != 'tdt' else data['r']
    
    color1 = COLOR_DL if prefix1 == 'dl' else (COLOR_GT if prefix1 == 'gt' else COLOR_TDT)
    draw_circle_on_image(img, cx1, cy1, r1, color1, prefix1.upper())

    # 3. Dibujar Círculo 2 (Método 2)
    cx2 = data[f'cx_{prefix2}'] if prefix2 != 'tdt' else data['cx']
    cy2 = data[f'cy_{prefix2}'] if prefix2 != 'tdt' else data['cy']
    r2  = data[f'r_{prefix2}'] if prefix2 != 'tdt' else data['r']

    color2 = COLOR_DL if prefix2 == 'dl' else (COLOR_GT if prefix2 == 'gt' else COLOR_TDT)
    draw_circle_on_image(img, cx2, cy2, r2, color2, prefix2.upper())

    # 4. Guardar imagen resultante
    cv2.imwrite(str(out_path), img)


# ======================================================================
# PROGRAMA PRINCIPAL
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Compara visualmente la segmentación de pupila (TDT vs DL vs GT).")
    parser.add_argument("--images_root", type=str, required=True, help="Raíz donde se encuentran las imágenes (ej: carpeta que contiene '001/L/...').")
    parser.add_argument("--tdt_csv", type=str, required=True, help="Ruta al CSV de resultados del método tradicional (pupil_segmentation_summary.csv).")
    parser.add_argument("--dl_csv", type=str, required=True, help="Ruta al CSV de resultados de Deep Learning (pupil_coords_deepvog.csv).")
    parser.add_argument("--gt_csv", type=str, required=True, help="Ruta al CSV de coordenadas de Ground Truth (gt_pupil_coordinates.csv).")
    parser.add_argument("--out_dir", type=str, default="visual_comparisons", help="Carpeta de salida para las imágenes de comparación.")
    args = parser.parse_args()

    images_root = Path(args.images_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. CARGAR Y UNIR DATOS ---
    df_tdt = load_data(Path(args.tdt_csv), COLS_TDT, 'tdt')
    df_dl = load_data(Path(args.dl_csv), COLS_DL, 'dl')
    df_gt = load_data(Path(args.gt_csv), COLS_GT, 'gt')

    if df_tdt is None or df_dl is None or df_gt is None:
        print("Finalizando debido a errores en la carga de datos.")
        return
    df_tdt
    # Unir todos los DataFrames por el índice 'stem'
    df_merged = df_tdt.join(df_dl, how='inner').join(df_gt, how='inner').reset_index()
    
    if df_merged.empty:
        print(" No se encontraron stems comunes en los tres CSVs. Revise sus identificadores 'stem'.")
        return
        
    print(f" Encontrados {len(df_merged)} imágenes con datos completos para TDT, DL y GT.")

    # --- 2. DIBUJAR COMPARACIONES ---
    comparisons = [
        ('tdt', 'gt', 'TDT_vs_GT'),
        ('dl', 'gt', 'DL_vs_GT'),
        ('tdt', 'dl', 'TDT_vs_DL'),
    ]

    for prefix1, prefix2, name in comparisons:
        comp_dir = out_dir / name
        comp_dir.mkdir(exist_ok=True)
        
        print(f"\n Generando comparación: {name} (Guardando en {comp_dir})")

        for _, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc=f"Dibujando {name}"):
            stem = row['stem']
            # Construir la ruta de la imagen original a partir de la columna 'relative_path'
            rel_path_tdt = row['relative_path']
            
            # Reconstruir la ruta completa: images_root / rel_path_tdt
            # (Ej: 'CASIA/001/L/S1001L03.jpg')
            img_path = images_root / rel_path_tdt
            
            out_path = comp_dir / f"{stem}_{name}.png"
            
            draw_comparison(row, img_path, out_path, prefix1, prefix2)
            
    print("\n Proceso de visualización finalizado.")

if __name__ == "__main__":
    main()
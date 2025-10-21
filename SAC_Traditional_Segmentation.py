# SAC_Traditional_Segmentation.py
# ------------------------------------------------------------
# Realizado por: Stefanny Arboleda
# Basado en el artículo: Robust and swift iris recognition at distance based on novel pupil segmentation
# ------------------------------------------------------------
# Implementación del pipeline de segmentación de pupila propuesto en el artículo
# ------------------------------------------------------------ 
import os # Operaciones de sistema de archivos
import cv2 # OpenCV para procesamiento de imágenes
import csv # Escritura del resumen en CSV
import time # Medición de tiempos por imagen
import argparse # CLI: parseo de argumentos
import multiprocessing as mp # Paralelismo por procesos
import numpy as np # Operaciones numéricas y matriciales
from pathlib import Path # Manejo de rutas
from tqdm import tqdm # Barra de progreso para batch

# =========================
#  FASE 1: PRE-PROCESADO
# =========================
def gaussian_blur(img_u8, sigma=3.5):
    """
    Función que aplica desenfoque gaussiano. Se usa para suavizar ruido de alta frecuencia antes de la extracción de texturas/contornos, preservando, en lo posible, la estructura global.

    Args:
        img_u8: Imagen en escala de grises uint8 de forma (H, W).
        sigma: Desviación estándar del kernel gaussiano (en píxeles).

    Returns:
        Imagen suavizada (uint8) de la misma forma que la entrada.
    """
    # ksize=(0,0) delega a OpenCV el tamaño según sigma; sigmaX==sigmaY.
    return cv2.GaussianBlur(img_u8, (0, 0), sigmaX=sigma, sigmaY=sigma)

def contrast_stretching_piecewise(img_u8, p_lo=1.0, p_hi=99.0):
    """
    Función para aumentar el contraste descartando extremos (p_lo, p_hi) para evitar saturación por outliers. Además, reescala a rango [0, 255] de forma lineal entre los percentiles.

    Args:
        img_u8: Imagen en escala de grises uint8 de forma (H, W).
        p_lo: Percentil inferior (0-100) usado como negro.
        p_hi: Percentil superior (0-100) usado como blanco.

    Returns:
        Imagen uint8 con contraste estirado; si hi<=lo, devuelve copia original.
    """
    # Cálculo de percentiles bajo distribución de intensidades.
    lo = np.percentile(img_u8, p_lo)
    hi = np.percentile(img_u8, p_hi)
    # Evita división por cero o inversión de contraste (datos patológicos).
    if hi <= lo:
        return img_u8.copy()
    # Convertimos a float32 para la operación lineal estable; mantiene precisión.
    x = img_u8.astype(np.float32)
    # Reescalado lineal: [lo, hi] -> [0, 255].
    x = (x - lo) * (255.0 / (hi - lo))
    # Saturación y retorno a uint8.
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x

def preprocess_exact(img_gray):
    """
    Función que aplica el preprocesado exacto usado por el pipeline. Aplica (1) filtro gaussiano y (2) stretching por percentiles. Esta etapa reduce ruido y mejora contraste para la MS-GLCM.

    Args:
        img_gray: Imagen uint8 en escala de grises (H, W).

    Returns:
        Imagen uint8 preprocesada (H, W).
    """
    # Suavizado para estabilizar co-ocurrencias.
    g = gaussian_blur(img_gray, sigma=1.5)
    # Estirado de contraste robusto a outliers.
    g = contrast_stretching_piecewise(g, p_lo=1.0, p_hi=99.0)
    return g

# =========================
#  FASE 2: MS-GLCM (dos escalas)
# =========================
def _quantize(img_u8, S):
    """
    Función que cuantiza intensidades a S niveles uniformes en [0, 255].

    Args:
        img_u8: Imagen uint8 (H, W).
        S: Número de niveles (típicamente 7 u 11), de acuerdo a los experimentos realizados.

    Returns:
        Imagen uint8 (H, W) con valores en {0, 1, ..., S-1}.
    """
    # Escalado a [0, S-1] y cast a uint8. Usamos float32 para precisión intermedia.
    return (img_u8.astype(np.float32) * (S - 1) / 255.0).astype(np.uint8)

def ms_glcm_picture_u8(img_u8, S: int):
    """
    'MS-GLCM picture' fiel al paper: imagen reescalada a S niveles (posterizada).
    Devuelve uint8 con bandas 0..255 en S niveles.
    """
    # cuantiza a clases 0..S-1
    q = _quantize(img_u8, S)  # uint8 en {0..S-1}
    # remapea a 0..255 para visualizar/guardar como en Fig. 7(b)
    if S <= 1: 
        return np.zeros_like(img_u8, dtype=np.uint8)
    pic = (q.astype(np.float32) * (255.0 / (S - 1))).astype(np.uint8)
    return pic

def seed_from_ms_picture(pic_u8):
    """
    ZOA según el paper: f(x,y)=0 si I(x,y)=0; 1 en otro caso.

    Returns:
      - seed01: ZOA para el pipeline (pupila=1, resto=0) en {0,1}.
    """
    # Paper: f(x,y)=0 si I(x,y)=0; 1 en otro caso
    seed01 = (pic_u8 == 0).astype(np.uint8)            # 1 fuera de pupila, 0 en pupila
    return seed01

# =========================
#  FASE 3: POST-PROCESO
# =========================
def fill_holes_flood(binary01, min_hole_area=500, min_obj_area=150):
    """
    Función que cierra agujeros pequeños y filtra objetos diminutos.

    Estrategia:
        1) Cierre morfológico suave (k=3 elipse) para unir bordes finos.
        2) Inversión y eliminación de componentes conectados pequeños (agujeros).
        3) Re-inversión y filtrado de objetos finales por área mínima.

    Args:
        binary01: Máscara {0,1}.
        min_hole_area: Área máxima para considerar un agujero y rellenarlo.
        min_obj_area: Área mínima para conservar un objeto final.

    Returns:
        Máscara {0,1} sin agujeros pequeños y sin microcomponentes.
    """
    # Asegurar formato uint8 0/255 para operaciones morfológicas en OpenCV.
    m = (binary01 * 255).astype(np.uint8)
    # Cierre morfológico para puentear gaps finos.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    # Invertimos para detectar agujeros como componentes conectados.
    inv = 255 - m
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    # Rellenar agujeros pequeños (área < min_hole_area).
    for i in range(1, n): # i=0 es el fondo
        if stats[i, cv2.CC_STAT_AREA] < min_hole_area:
            inv[labels == i] = 0
    # Revertimos la inversión: agujeros pequeños quedan rellenos.
    m = 255 - inv

    # Segunda pasada: suprimir objetos diminutos en la máscara final.
    n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n2):
        if stats2[i, cv2.CC_STAT_AREA] >= min_obj_area:
            out[labels2 == i] = 255
    return (out > 0).astype(np.uint8)

# =========================
#  FASE 4: MRR-CHT (Detección de círculo por Hough sobre bordes)
# =========================
def hough_best_circle(edge_src_u8, ranges, dp=1.2, param1=120, param2=18):
    """
    Función que selecciona el mejor círculo en rangos de radio usando CHT.

    Procedimiento:
        - Extrae bordes con Canny sobre la imagen de entrada.
        - Para cada rango (rmin, rmax), invoca HoughCircles.
        - Puntúa cada círculo por la intensidad media de bordes en un anillo de +-2 px.
        - Devuelve el círculo con mejor puntuación.

    Args:
        edge_src_u8: Imagen uint8 (H, W); aquí se aplica Canny internamente.
        ranges: Lista de tuplas (minRadius, maxRadius) en píxeles.
        dp: Inverso de la resolución acumulada de Hough (OpenCV).
        param1: Umbral superior de Canny usado por HoughCircles.
        param2: Umbral de acumulación para centros (más alto = menos falsos).

    Returns:
        (cx, cy, r) en float si se detecta; None si no hay círculos válidos.
    """
    # Inicializa mejor candidato.
    best = None; best_score = -1.0
    # Bordes prismáticos sobre la fuente (aun si es máscara dilatada).
    edges = cv2.Canny(edge_src_u8, 100, 200)
    # Explora rangos de radios predefinidos.
    for (rmin, rmax) in ranges:
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=dp,
                                   minDist=max(8, rmin//2),
                                   param1=param1, param2=param2,
                                   minRadius=rmin, maxRadius=rmax)
        # Si no hay círculos en este rango, continúa.
        if circles is None:
            continue
        # Recorre candidatos y puntúa por energía de borde en anillo.
        for (x, y, r) in circles[0]:
            x, y, r = float(x), float(y), float(r)
            H, W = edges.shape
            yy, xx = np.ogrid[:H, :W]
            dist = np.sqrt((xx - x)**2 + (yy - y)**2)
            ring = (dist >= max(1, int(r-2))) & (dist <= int(r+2))
            score = edges[ring].mean() if ring.any() else 0.0
            if score > best_score:
                best_score = score; best = (x, y, r)
    return best

# =========================
#  WRAPPER EXACTO (sin detección previa, parte MS-GLCM -> Hough)
# =========================
def run_hough_on_seed(seed01, ranges, param1=120, param2=18, dp=1.2):
    """
    Función que aplica dilatación a la ZOA de esta rama y corre Hough en el/los rangos dados, para estimar el mejor círculo.

    Args:
        seed01: Máscara binaria {0,1} de la ZOA (semilla de pupila).
        ranges: Rangos de radio (rmin, rmax) para HoughCircles.
        param1: Umbral alto de Canny para HoughCircles.
        param2: Umbral de acumulación para HoughCircles.
        dp: Parámetro de resolución de Hough (OpenCV).


    Returns:
        (cx, cy, r) del mejor círculo o None si falla.
    """
    # Convertimos la semilla a 0/255 y aplicamos una dilatación suave.
    roi = (seed01 * 255).astype(np.uint8)
    roi = cv2.dilate(roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)), iterations=1)
    # Buscamos círculo sobre la ROI dilatada.
    return hough_best_circle(roi, ranges=ranges, dp=dp, param1=param1, param2=param2)

def segment_pupil(img_gray_u8):
    """
    Pipeline completo de segmentación de pupila usando el método tradicional para una imagen.

    Secuencia:
        1) Preprocesado (gauss + stretching).
        2) Rama S1: MS-GLCM con S1=7. Si falla, intenta Rama S2: S2=11.
        3) Postproceso: binarización + relleno de agujeros + filtro de área.
        4) Hough (MRR-CHT) sobre la ZOA dilatada.
        5) Construcción de la máscara circular final.

    Args:
        img_gray_u8: Imagen de entrada en escala de grises uint8 (H, W).

    Returns:
        center: (cx, cy, r) o None si no se detecta la pupila.
        mask: Máscara booleana (H, W) de la pupila (None si falla).
        used_seed: Semilla binaria booleana (H, W) usada para Hough.
        ms_picture_u8: imagen u8 tipo 'MS-GLCM picture' con el S usado.
    """
    # Preprocesado robusto a ruido y variaciones de iluminación.
    g = preprocess_exact(img_gray_u8)

    # Parámetros paper (CASIA): niveles y rangos de radio.
    S1, S2 = 7, 11
    R1 = [(24, 75)]
    R2 = [(22, 75)]

    # ----- Rama S1:  picture a S1 niveles -> ZOA (nivel 0) -> morph -> Hough -----
    pic1 = ms_glcm_picture_u8(g, S1)
    seed1 = seed_from_ms_picture(pic1)
    seed1 = fill_holes_flood(seed1, 500, 150)
    circle = run_hough_on_seed(seed1, R1, param1=120, param2=18, dp=1.2)

    used_seed = seed1
    used_pic = pic1

    if circle is None:
        # ----- Rama S2 (solo si falla la S1) -----
        pic2 = ms_glcm_picture_u8(g, S2)
        seed2 = seed_from_ms_picture(pic2)
        seed2 = fill_holes_flood(seed2, 500, 150)
        circle = run_hough_on_seed(seed2, R2, param1=120, param2=18, dp=1.2)
        used_seed = seed2
        used_pic = pic2

    # Si aún no hay círculo, reporta fallo.
    if circle is None:
        return None, None, used_seed.astype(bool), used_pic

    # Construcción de máscara circular (centro y radio en float -> booleano).
    cx, cy, r = circle
    H, W = img_gray_u8.shape
    yy, xx = np.ogrid[:H, :W]
    mask = (((xx - cx)**2 + (yy - cy)**2) <= (r*r)).astype(bool)
    return (cx, cy, r), mask, used_seed.astype(bool), used_pic


# =========================
#  BATCH (estructura 000..999/L,R/*.jpg)
# =========================
def process_one(args):
    """
    Procesa una imagen: segmenta la pupila y guarda outputs.

    Args:
        args: Tupla (img_path, input_root, output_root, dataset, overwrite, save_seed).

    Returns:
        Tupla CSV: (relative_path, status, cx, cy, r, elapsed_ms).
                    - status pertenece a {"ok", "fail", "skip", "read_error"}, para validación
                    - Si falla, cx, cy, r serán "" (vacío).
                    - Guarda overlay con el círculo y centro.
                    - Guarda máscara binaria de pupila.
    """
    # Desempaquetar argumentos.
    img_path, input_root, output_root, overwrite, save_seed = args
    # Ruta relativa para replicar estructura de carpetas.
    rel = Path(img_path).relative_to(input_root)
    out_dir = (output_root / rel.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rutas de salida.
    stem = Path(img_path).stem
    out_overlay = out_dir / f"{stem}_seg.png"
    out_mask    = out_dir / f"{stem}_mask.png"
    out_seed    = out_dir / f"{stem}_seed.png"
    out_ms_pic = out_dir / f"{stem}_msglcm_picture_S.png"

    # Respeta archivos previos salvo que se pida overwrite.
    if (not overwrite) and out_overlay.exists() and out_mask.exists() and (not save_seed or out_seed.exists()):
        return (str(rel).replace("\\","/"), "skip", "", "", "", 0)

    # Lectura robusta en escala de grises.
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (str(rel).replace("\\","/"), "read_error", "", "", "", 0)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Segmentación y conteo del tiempo.
    t0 = time.time()
    center, mask, seed, ms_picture = segment_pupil(img)
    elapsed = (time.time() - t0) * 1000.0

    # Manejo de fallos: opcionalmente guarda la semilla para análisis.
    if center is None:
        if save_seed and seed is not None:
            cv2.imwrite(str(out_seed), (seed.astype(np.uint8) * 255))
        return (str(rel).replace("\\","/"), "fail", "", "", "", f"{elapsed:.1f}")

    # Dibujo de círculo y centro sobre la imagen original.
    cx, cy, r = center
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.circle(out, (int(cx),int(cy)), int(r), (0,255,0), 2)
    cv2.circle(out, (int(cx),int(cy)), 2, (0,0,255), 2)
    cv2.imwrite(str(out_overlay), out)
    # Guardar máscara binaria de la pupila.
    cv2.imwrite(str(out_mask), (mask.astype(np.uint8)*255))
    # Guardar la MS-GLCM
    cv2.imwrite(str(out_ms_pic), ms_picture)
    # Guardar semilla si se solicita.
    if save_seed and seed is not None:
        cv2.imwrite(str(out_seed), (seed.astype(np.uint8)*255))

    return (str(rel).replace("\\","/"), "ok", f"{cx:.2f}", f"{cy:.2f}", f"{r:.2f}", f"{elapsed:.1f}")

def process_folder(input_root, output_root, dataset="CASIA", overwrite=False, save_seed=True):
    """
    Procesa recursivamente una carpeta de imágenes .jpg. 
    De la siguiente manera:
        - Recorre input_root buscando *.jpg.
        - Ejecuta segmentación en paralelo (mp.Pool) y escribe un CSV resumen.
        - Replica la estructura de carpetas de entrada en la salida.

    Args:
        input_root: Raíz del dataset (contiene 000..999/L,R/*.jpg).
        output_root: Carpeta donde se guardan resultados y el CSV.
        dataset: Nombre del dataset (informativo).
        overwrite: Si True, sobrescribe outputs existentes.
        save_seed: Si True, guarda la imagen de la semilla (ZOA) usada.

    Returns:
        Crea archivos en disco y escribe un CSV.
    """
    # Normaliza rutas a Path.
    input_root  = Path(input_root)
    output_root = Path(output_root)
    # Busca imágenes .jpg recursivamente y las ordena para reproducibilidad.
    jpgs = sorted(str(p) for p in input_root.rglob("*.jpg"))
    if not jpgs:
        print(f"No se encontraron .jpg en {input_root}")
        return

    # Ruta del CSV resumen y aseguro carpeta existente.
    csv_path = output_root / "pupil_segmentation_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Optimización OpenCV + explícito de hilos para no interferir con mp.
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)

    # Preparo lista de argumentos para map paralelo.
    args_list = [(p, input_root, output_root, overwrite, save_seed) for p in jpgs]

    # Escribo CSV en streaming a medida que llegan resultados del pool.
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "status", "cx", "cy", "r", "elapsed_ms"])

        # Pool del tamaño del número de CPUs disponibles.
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for row in tqdm(pool.imap_unordered(process_one, args_list), total=len(args_list), unit="img", desc="Segmentando pupila"):
                writer.writerow(row)

    print(f"\nListo. Resumen en: {csv_path}")

def main():
    """
    Punto de entrada por consola. Define y parsea argumentos, y lanza el procesamiento de la carpeta.

    Flags principales:
        --input_root Raíz del dataset CASIA-Iris-Interval (000..999).
        --output_root Carpeta de salida (replica estructura).
        --dataset Nombre del dataset (por ahora informativo).
        --overwrite Reescribe si ya existen outputs.
        --no-seed No guardar imagen de semilla (ZOA).

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Segmentación de pupila por batches para CASIA-Iris-Interval.")
    parser.add_argument("--input_root", type=str, required=True, help="Raíz de CASIA-Iris-Interval (contiene 000..999).")
    parser.add_argument("--output_root", type=str, required=True, help="Carpeta salida (replica estructura).")
    parser.add_argument("--dataset", type=str, default="CASIA", help='\"CASIA\" (rangos 24–75/22–75).')
    parser.add_argument("--overwrite", action="store_true", help="Reescribe archivos existentes.")
    parser.add_argument("--no-seed", action="store_true", help="No guardar imagen de semilla (ZOA).")
    args = parser.parse_args()

    process_folder(args.input_root, Path(args.output_root),
                   dataset=args.dataset, overwrite=args.overwrite, save_seed=(not args.no_seed))

if __name__ == "__main__":
    main()
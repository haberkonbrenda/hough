import numpy as np
import math
import matplotlib.pyplot as plt
import os

# ============================================================
# Crear carpeta outputs automáticamente
# ============================================================

os.makedirs("outputs", exist_ok=True)

def guardar_fig(nombre):
    """Guarda la figura actual en la carpeta outputs/"""
    plt.savefig(f"outputs/{nombre}", dpi=140, bbox_inches='tight')


# ============================================================
# Generación de imágenes sintéticas
# ============================================================

def dibujar_linea(img, x0, y0, x1, y1, valor=255):
    """
    Dibuja una línea usando el algoritmo de Bresenham.
    img: matriz 2D (uint8)
    (x0, y0) -> (x1, y1) en coordenadas (col, fila)
    """
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 >= x0 else -1
    sy = 1 if y1 >= y0 else -1
    x, y = x0, y0

    if dy <= dx:
        err = dx // 2
        while x != x1:
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                img[y, x] = valor
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
        img[y1, x1] = valor
    else:
        err = dy // 2
        while y != y1:
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                img[y, x] = valor
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
        img[y1, x1] = valor


def dibujar_circunferencia(img, cx, cy, radio, valor=255):
    """
    Dibuja una circunferencia aproximada con el algoritmo de mid-point.
    (cx, cy) centro en coordenadas (col, fila).
    """
    x = radio
    y = 0
    d = 1 - x

    def plot(cx, cy, x, y):
        pts = [
            (cx + x, cy + y), (cx - x, cy + y),
            (cx + x, cy - y), (cx - x, cy - y),
            (cx + y, cy + x), (cx - y, cy + x),
            (cx + y, cy - x), (cx - y, cy - x),
        ]
        for px, py in pts:
            if 0 <= py < img.shape[0] and 0 <= px < img.shape[1]:
                img[int(py), int(px)] = valor

    while y <= x:
        plot(cx, cy, x, y)
        y += 1
        if d <= 0:
            d += 2*y + 1
        else:
            x -= 1
            d += 2*(y - x) + 1


def agregar_ruido_gaussiano(img, sigma=10):
    ruido = np.random.normal(0, sigma, img.shape)
    out = img.astype(float) + ruido
    return np.clip(out, 0, 255).astype(np.uint8)


def generar_imagen_rectas(size=(256, 256)):
    img = np.zeros(size, dtype=np.uint8)
    # Algunas rectas cruzadas de prueba
    dibujar_linea(img, 10, 20, 240, 220, 255)
    dibujar_linea(img, 20, 230, 230, 50, 255)
    dibujar_linea(img, 0, 128, 255, 128, 255)
    return agregar_ruido_gaussiano(img, sigma=15)


def generar_imagen_circunferencias(size=(256, 256), centro=(140, 110), radio=45, espesor=2):
    img = np.zeros(size, dtype=np.uint8)
    # Aro principal: "anillo" de cierto espesor alrededor del radio
    for r in range(radio - espesor, radio + espesor + 1):
        dibujar_circunferencia(img, centro[0], centro[1], r, 255)

    # Algunas circunferencias pequeñas para "molestar"
    dibujar_circunferencia(img, 40, 40, 12, 255)
    dibujar_circunferencia(img, 200, 200, 16, 255)
    return agregar_ruido_gaussiano(img, sigma=10)


# ============================================================
# Detector de Bordes - Sobel
# ============================================================

def sobel_bordes(img, umbral=50):
    """
    Aplica Sobel en X e Y, calcula magnitud de gradiente
    y devuelve una imagen binaria de bordes.
    """
    imgf = img.astype(np.float32)

    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Gy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

    h, w = img.shape
    bordes = np.zeros_like(imgf)

    padded = np.pad(imgf, 1, mode='edge')

    for y in range(h):
        for x in range(w):
            region = padded[y:y+3, x:x+3]
            gx = np.sum(region * Gx)
            gy = np.sum(region * Gy)
            mag = np.sqrt(gx*gx + gy*gy)
            bordes[y, x] = mag

    return (bordes > umbral).astype(np.uint8) * 255, bordes


# ============================================================
# Consigna 2 - Hough rectas
# ============================================================

def hough_rectas(bordes, res_theta=1, res_rho=1, umbral=120):
    """
    Hough para rectas en espacio (rho, theta).
    bordes: imagen binaria con 0 / 255.
    Devuelve: acumulador, vector rhos, vector thetas, lista de picos.
    """
    ys, xs = np.nonzero(bordes)
    h, w = bordes.shape
    diag = int(np.ceil(np.sqrt(h*h + w*w)))

    rhos = np.arange(-diag, diag+1, res_rho)
    thetas = np.deg2rad(np.arange(-90, 90, res_theta))

    acumulador = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # Voto de cada píxel borde
    for y, x in zip(ys, xs):
        for i, theta in enumerate(thetas):
            rho = x*cos_t[i] + y*sin_t[i]
            ir = int(round(rho - rhos[0]))
            if 0 <= ir < len(rhos):
                acumulador[ir, i] += 1

    # Búsqueda de picos simples con umbral + no-max local
    picos = []
    for ir in range(1, acumulador.shape[0]-1):
        for it in range(1, acumulador.shape[1]-1):
            if acumulador[ir, it] >= umbral:
                ventana = acumulador[ir-1:ir+2, it-1:it+2]
                if acumulador[ir, it] == ventana.max():
                    picos.append((rhos[ir], thetas[it], acumulador[ir, it]))

    return acumulador, rhos, thetas, picos


def dibujar_rectas_detectadas(img, picos, max_lineas=5):
    """
    Dibuja las rectas correspondientes a los picos sobre la imagen.
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray')
    plt.title("Rectas detectadas")
    plt.axis('off')

    for rho, theta, _ in sorted(picos, key=lambda p: -p[2])[:max_lineas]:
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a*rho
        y0 = b*rho
        # Dos puntos lejanos sobre la recta
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        plt.plot([x1, x2], [y1, y2], linewidth=1)

    guardar_fig("rectas_detectadas.png")
    plt.show()


# ============================================================
# Consigna 3 - Hough circunferencias (radio fijo)
# ============================================================

def hough_circunferencias_radio_fijo(bordes, radio, paso_theta=2, umbral=90):
    """
    Hough para circunferencias con radio conocido.
    bordes: imagen binaria 0 / 255
    radio: radio conocido (en píxeles)
    Devuelve: acumulador de centros (a, b) y lista de picos.
    """
    ys, xs = np.nonzero(bordes)
    h, w = bordes.shape

    acumulador = np.zeros((h, w), dtype=np.int32)
    thetas = np.deg2rad(np.arange(0, 360, paso_theta))
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # Voto de centros
    for y, x in zip(ys, xs):
        # Para cada ángulo, calculo posible centro (a, b)
        a_vals = x - radio*cos_t
        b_vals = y - radio*sin_t
        for a, b in zip(a_vals, b_vals):
            ia = int(round(a))
            ib = int(round(b))
            if 0 <= ia < w and 0 <= ib < h:
                acumulador[ib, ia] += 1

    # Búsqueda de picos
    picos = []
    for y in range(1, h-1):
        for x in range(1, w-1):
            if acumulador[y, x] >= umbral:
                ventana = acumulador[y-1:y+2, x-1:x+2]
                if acumulador[y, x] == ventana.max():
                    picos.append((x, y, acumulador[y, x]))

    return acumulador, picos


def dibujar_circunferencias_detectadas(img, picos, radio, max_circulos=3):
    """
    Dibuja circunferencias a partir de centros encontrados.
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray')
    plt.title("Circunferencias detectadas")
    plt.axis('off')

    for cx, cy, _ in sorted(picos, key=lambda p: -p[2])[:max_circulos]:
        theta = np.linspace(0, 2*np.pi, 360)
        xs = cx + radio*np.cos(theta)
        ys = cy + radio*np.sin(theta)
        plt.plot(xs, ys, linewidth=1)
        plt.plot([cx], [cy], 'ro', markersize=3)

    guardar_fig("circunferencias_detectadas.png")
    plt.show()


# ============================================================
# DEMOS finales
# ============================================================

def demo_rectas():
    img = generar_imagen_rectas()

    # Imagen original
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray')
    plt.title("Imagen original (rectas)")
    plt.axis('off')
    guardar_fig("imagen_rectas_original.png")
    plt.show()

    # Bordes
    bordes, _ = sobel_bordes(img, umbral=60)

    plt.figure(figsize=(5, 5))
    plt.imshow(bordes, cmap='gray')
    plt.title("Bordes (Sobel)")
    plt.axis('off')
    guardar_fig("bordes_rectas.png")
    plt.show()

    # Hough rectas
    acc, rhos, thetas, picos = hough_rectas(bordes)

    plt.figure(figsize=(6, 4))
    extent = [np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]]
    plt.imshow(acc, aspect='auto', extent=extent)
    plt.title("Acumulador Hough – Rectas")
    plt.xlabel("θ")
    plt.ylabel("ρ")
    plt.colorbar(label="Votos")
    guardar_fig("acumulador_rectas.png")
    plt.show()

    dibujar_rectas_detectadas(img, picos)


def demo_circunferencias():
    centro_real = (140, 110)
    radio = 45
    img = generar_imagen_circunferencias(centro=centro_real, radio=radio)

    # Imagen original
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray')
    plt.title("Imagen original (circunferencias)")
    plt.axis('off')
    guardar_fig("imagen_circ_original.png")
    plt.show()

    # Bordes
    bordes, _ = sobel_bordes(img, umbral=50)

    plt.figure(figsize=(5, 5))
    plt.imshow(bordes, cmap='gray')
    plt.title("Bordes (Sobel)")
    plt.axis('off')
    guardar_fig("bordes_circunf.png")
    plt.show()

    # Hough circunferencias
    acc, picos = hough_circunferencias_radio_fijo(bordes, radio)

    plt.figure(figsize=(5, 5))
    plt.imshow(acc, cmap='hot')
    plt.title("Acumulador Hough – Centros")
    plt.axis('off')
    plt.colorbar(label="Votos")
    guardar_fig("acumulador_circunf.png")
    plt.show()

    dibujar_circunferencias_detectadas(img, picos, radio)


if __name__ == "__main__":
    # Prueba consigna 2
    print("=== Hough RECTAS ===")
    demo_rectas()

    # Prueba consigna 3
    print("=== Hough CIRCUNFERENCIAS ===")
    demo_circunferencias()

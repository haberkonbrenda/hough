# README – Transformada de Hough (Rectas y Circunferencias)

Este repositorio contiene un prototipo didáctico que implementa:

- Transformada de Hough para **rectas**
- Transformada de Hough para **circunferencias de radio conocido**
- Detector de bordes **Sobel**
- Generación de imágenes sintéticas para pruebas

---

## 🚀 Cómo ejecutar

1. Clonar el repositorio

2. Instalar dependencias mínimas:

```bash
pip install numpy matplotlib
```

3. Ejecutar el script principal:

```bash
python tp4_hough.py
```

---

## 📊 Qué genera el script

El programa ejecuta dos demos:

### 1) Hough para rectas
- Imagen sintética con líneas  
- Bordes detectados con Sobel  
- Acumulador de Hough (ρ, θ)  
- Rectas detectadas superpuestas  

### 2) Hough para circunferencias
- Imagen sintética con un aro de radio fijo  
- Bordes detectados  
- Acumulador de centros  
- Circunferencias detectadas y centro estimado  

Cada demo muestra los resultados en pantalla mediante `matplotlib`.

---

## 📁 Estructura del proyecto

```
tp4_hough.py      # Script principal con toda la lógica
README.md         # Instrucciones breves de uso
outputs/          # (Opcional) Carpeta para guardar figuras generadas
```



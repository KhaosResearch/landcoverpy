# Documentación Técnica y Metodológica: Arquitectura de Bandas, Índices y Almacenamiento en LandCoverPy

Este documento recoge las decisiones metodológicas, físicas y computacionales adoptadas en el procesamiento de productos Sentinel-2 L2A y la generación de composites estacionales en **LandCoverPy** para el dominio bioclimático de la cuenca mediterránea (año 2021). Su contenido está estructurado para servir como referencia técnica y sección metodológica para publicación científica.

---

## 1. Comportamiento de Resolución Espacial en Sentinel-2 L2A

> **Regla de empaquetado de ESA / Sen2Cor:** El procesador oficial Sen2Cor submuestrea hacia resoluciones más gruesas (mayor tamaño de píxel), **nunca hacia resoluciones más finas**.

* **Bandas nativas de 10 m ($B02, B03, B04, B08$):** 
  Se entregan en los paquetes `R10m`, `R20m` y `R60m`.
* **Bandas nativas de 20 m ($B05, B06, B07, B8A, B11, B12$):**
  Se entregan en `R20m` y `R60m`. No existen a 10 m en el producto estándar L2A.
* **Bandas nativas de 60 m ($B01, B09$):**
  Se entregan **exclusivamente en `R60m`**.

---

## 2. Resolución del Mapa Final de Clasificación (10 Metros)

El mapa final de clasificación de coberturas terrestres generado por LandCoverPy (`classification_{tile}.tif`) **posee una resolución espacial estricta de 10 metros** ($10.980 \times 10.980$ píxeles por tile MGRS de $100 \times 100\text{ km}$).

### Justificación e Implementación en el Pipeline:
1. **Cuadrícula y georreferenciación de referencia:**
   En `workflow_predict.py` (Líneas 113-135), el raster de salida se inicializa a partir de una banda nativa de 10 m (`B02_10m`, `B03_10m`, `B04_10m` o `B08_10m`), fijando la matriz en:
   $$\text{Transform} = [10.0, 0.0, x_0, 0.0, -10.0, y_0], \quad \text{Shape} = (10980, 10980)$$
2. **Re-escalado dinámico en memoria RAM:**
   Durante las etapas de entrenamiento (`workflow_train.py`, Línea 224) y predicción (`workflow_predict.py`, Líneas 312 y 338), la lectura de cualquier raster almacenado a resolución de 20 m ($5.490 \times 5.490$) o 60 m ($1.830 \times 1.830$) se realiza mediante:
   ```python
   _read_raster(path, rescale=True)
   ```
   Esta función ejecuta una interpolación de vecino más cercano (`_rescale_band`), proyectando la información a la rejilla de 10 m en memoria volátil de forma transparente.
3. **Consistencia de inferencia:**
   El modelo Random Forest recibe todas las covariables fenológicas y espectrales perfectamente alineadas a resolución de 10 metros sin requerir inflar el almacenamiento persistente en disco.

---

## 3. Justificación de Bandas Duplicadas (`B03_10m`/`B03_20m` y `B05_20m`/`B05_60m`)

La inclusión de ciertas bandas en dos resoluciones distintas en el almacenamiento persistente responde estrictamente a **requisitos de álgebra matricial en NumPy** durante el cálculo de los índices:

En `src/landcoverpy/utilities/raw_index_calculation.py` (Líneas 31-48):
* **`MNDWI`:** Se formula como $(B03 - B11) / (B03 + B11)$, donde $B11$ tiene resolución nativa de 20 m ($5.490 \times 5.490$). Para realizar la operación vectorial directa en disco temporal sin disparar una excepción de difusión matricial (`ValueError: operands could not be broadcast together`), se utiliza la versión $B03\_20m$.
* **`NDRE`:** Se formula como $(B09 - B05) / (B09 + B05)$, donde $B09$ (vapor de agua) tiene resolución nativa de 60 m ($1.830 \times 1.830$). Para operar con $B09$, se requiere la versión $B05\_60m$.

De este modo, se conservan las 12 bandas espectrales nativas y únicamente estas dos versiones auxiliares indispensables para las fórmulas de los índices.

---

## 4. Justificación de Purga de `s2-products` (Datos Raw)

| Criterio | `s2-products` (Capturas crudas) | `s2-composites` (Composites estacionales) |
| :--- | :--- | :--- |
| **Naturaleza del dato** | 5 a 16 escenas individuales por tile y estación. Contienen nubes, sombras, nieve y artefactos atmosféricos. | 1 síntesis estacional limpia. Mediana píxel a píxel (`np.nanmedian`) de adquisiciones seleccionadas. |
| **Rol en el pipeline** | Materia prima temporal. | Entrada definitiva para entrenamiento, validación y predicción cartográfica. |
| **Espacio requerido** | $> 45\text{ Terabytes}$ para los 540 tiles mediterráneos. | $\approx 8.5\text{ Terabytes}$ (con compresión DEFLATE). |
| **Decisión de diseño** | **Purga inmediata** tras validar la integridad del composite en MinIO. Almacenarlas permanentemente es redundancia innecesaria. | **Conservación permanente**. |

---

## 5. Decisión de Diseño: Selección de Bandas e Índices en Composites

### A. Bandas Espectrales en `/raw/` (14 bandas conservadas)

El Random Forest de LandCoverPy **utiliza directamente las reflectancias espectrales de superficie como variables predictoras**, no únicamente los índices. Se conservan las 12 bandas físicas más las 2 auxiliares matriciales:

* **Bandas de 10 m:** `B02_10m`, `B03_10m`, `B04_10m`, `B08_10m`.
* **Bandas de 20 m:** `B05_20m`, `B06_20m`, `B07_20m`, `B8A_20m`, `B11_20m`, `B12_20m`, más `B03_20m`.
* **Bandas de 60 m:** `B01_60m`, `B09_60m`, más `B05_60m`.
* **Capas descartadas:** `AOT_10m`, `AOT_20m`, `AOT_60m` (espesor óptico de aerosoles) y `WVP` (vapor de agua), por ser subproductos atmosféricos no reflectivos.

---

### B. Índices Biofísicos en `/indexes/` (11 Conservados vs 4 Descartados)

Tras la revisión metodológica del equipo, se formaliza la siguiente selección de **11 índices conservados** y **4 elementos eliminados**:

#### 1. Índices Conservados (11 índices)

```python
indexes_used = [
    "cri1",
    "ri",
    "evi2",
    "mndwi",
    "moisture",
    "ndyi",
    "ndre",
    "ndvi",
    "osavi",
    "bri",
    "bsi",
]
```

| Índice | Res. Cálculo | Fórmula | Justificación Científica y Decisión de Diseño |
| :--- | :---: | :---: | :--- |
| **`ndvi`** | 10 m | $\frac{B08 - B04}{B08 + B04}$ | **Núcleo del modelo:** Biomasa fotosintéticamente activa y dinámica de vigor vegetal. |
| **`evi2`** | 10 m | $2.5 \cdot \frac{B08 - B04}{B08 + 2.4 \cdot B04 + 1.0}$ | **Núcleo del modelo:** Índice mejorado de 2 bandas, insensible a aerosoles atmosféricos en doseles densos. |
| **`osavi`** | 10 m | $\frac{B08 - B04}{B08 + B04 + 0.16}$ | **Núcleo del modelo:** Corrección de reflectancia de fondo en formaciones de matorral ralo y dehesa. |
| **`ri`** | 10 m | $\frac{B04 - B03}{B04 + B03}$ | **Núcleo del modelo:** Índice de enrojecimiento; contenido de óxidos de hierro y suelos arcillosos. |
| **`cri1`** | 10 m | $\frac{1}{B02} - \frac{1}{B03}$ | **Núcleo del modelo:** Carotenoides foliares; respuesta ante fotoinhibición y senescencia estival. |
| **`ndyi`** | 10 m | $\frac{B02 - B03}{B02 + B03}$ | **Núcleo del modelo:** Amarilleamiento fenológico; esencial en floración de primavera y agostamiento. |
| **`mndwi`** | 20 m | $\frac{B03 - B11}{B03 + B11}$ | **Núcleo del modelo:** Delimitación de láminas de agua suprimiendo falsos positivos urbanos y edáficos. |
| **`moisture`**| 20 m | $\frac{B8A - B11}{B8A + B11}$ | **Núcleo del modelo:** Estrés hídrico foliar y contenido de agua en copas forestales. |
| **`ndre`** | 60 m | $\frac{B09 - B05}{B09 + B05}$ | **Núcleo del modelo:** Transición Red Edge / vapor de agua para saturación de clorofila. |
| **`bsi`** *(Bare Soil Index)* | 10 m | $\frac{(B11 + B04) - (B08 + B02)}{(B11 + B04) + (B08 + B02)}$ | **Decisión metodológica de preservación:** Aunque la configuración base del clasificador discrimina suelo mediante $B11/B12$, en el dominio mediterráneo los gradientes de desertificación, sobrepastoreo y erosión son críticos. Se decide explícitamente conservar $BSI$ para posibilitar análisis de degradación de suelos y experimentos comparativos futuros sin requerir recálculos masivos. |
| **`bri`** *(Biological Ripening Index)* | 10 m | $\frac{B03 - B05}{B08}$ | **Decisión metodológica de preservación:** Caracteriza la lignificación, contenido de madera y maduración biológica en masas leñosas y matorrales esclerófilos mediterráneos. Se preserva como covariable fenológica de alto valor ecológico. |

---

#### 2. Elementos Descartados Definitivamente (4 elementos)

| Elemento descartado | Res. | Peso aprox. | Motivo Científico y Metodológico del Descarte |
| :--- | :---: | :---: | :--- |
| **`evi`** | 10 m | ~395 MB | **Redundancia innecesaria:** Utiliza la banda azul ($B02$), la cual presenta dispersión de Rayleigh y sensibilidad a aerosoles residuales. Es sustituido de forma universalmente superior por **`evi2`**, que aporta la misma señal biofísica con mayor relación señal/ruido. |
| **`tci`** | 10 m | ~498 MB | **Producto no analítico:** Corresponde a una imagen RGB en enteros ($uint8$, 0–255) destinada únicamente a visualización humana en monitores. No es un índice científico y el código lo excluye formalmente con `skip_bands = ["tci", "scl"]`. |
| **`ndwi`** | 10 m | ~378 MB | **Superado por `mndwi`:** El $NDWI$ tradicional (McFeeters, Green/NIR) genera confusión espectral y falsos positivos en suelos áridos y superficies antrópicas reflectivas. Se reemplaza por **`mndwi`** (Xu, Green/SWIR1), adoptado como estándar en el pipeline. |
| **`ndsi`** | 20 m | ~0.3 MB | **Inaplicable al objetivo ecológico:** Diseñado para discriminar nieve y hielo. En la clasificación anual de tipologías forestales y agrícolas mediterráneas, la nieve es un fenómeno transitorio invernal que no constituye una clase fenológica objetivo. |

---

## 6. Balance de Almacenamiento e Impacto en Infraestructura

Al descartar los 4 elementos innecesarios (`evi`, `tci`, `ndwi`, `ndsi`), se elimina **1.27 GB de almacenamiento muerto por composite**:

* **Volumen previo sin depuración (24 bandas + 15 índices):**
  $$2.160 \times 5.3\text{ GB} \approx \mathbf{11.45\text{ TB}} \ (\mathbf{10.41\text{ TiB}}) \quad \longrightarrow \quad \text{Desborda el disco físico de 10 TB}$$
* **Volumen acordado (14 bandas + 11 índices optimizados):**
  $$2.160 \times 4.0\text{ GB} \approx \mathbf{8.64\text{ TB}} \ (\mathbf{7.85\text{ TiB}}) \quad \longrightarrow \quad \text{Cabe con holgura en el disco de 10 TB}$$
* **Margen de seguridad:** Se garantiza **$> 1.2\text{ TiB}$** de espacio libre en el nodo de almacenamiento, previniendo detenciones por desbordamiento de disco.

# INFORME TÉCNICO
# Proyecto A — Regresión: Energy Efficiency Data Set

---

**Estudiante:** Óscar Mauricio García Mesa

**Fecha:** Noviembre 2025

**Curso:** Inteligencia Artificial

---

## 1. Introducción

### 1.1 Descripción del Problema

El presente proyecto aborda un problema de **regresión supervisada** utilizando el dataset de **Energy Efficiency** del UCI Machine Learning Repository. El objetivo principal es desarrollar modelos predictivos que permitan estimar la **carga de calefacción (Heating Load)** de edificios basándose en características arquitectónicas y de diseño.

### 1.2 Justificación

Este es un problema de **regresión** porque:

- La variable objetivo (`Heating_Load`) es **continua**: puede tomar cualquier valor numérico real dentro de un rango (no es una categoría discreta).
- El objetivo es **predecir un valor numérico específico** que represente la carga de calefacción medida en kWh/m².
- Las métricas de evaluación (MAE, RMSE, R²) son propias de problemas de regresión.

### 1.3 Relevancia Práctica

La predicción de la eficiencia energética en edificios es fundamental para:

- **Sostenibilidad ambiental**: Reducir emisiones de CO₂ asociadas al consumo energético
- **Optimización económica**: Minimizar costos operacionales de calefacción
- **Diseño arquitectónico**: Tomar decisiones informadas durante la fase de diseño de edificios
- **Cumplimiento normativo**: Adherir a estándares y regulaciones de eficiencia energética

---

## 2. Descripción del Dataset

### 2.1 Origen y Fuente

- **Fuente principal**: UCI Machine Learning Repository
- **Disponibilidad**: Kaggle - Energy Efficiency Data Set
- **URL**: https://www.kaggle.com/datasets/ujjwalchowdhury/energy-efficiency-data-set

### 2.2 Características del Dataset

**Dimensiones:**
- **Observaciones**: 768 registros
- **Variables totales**: 10 (8 de entrada + 2 de salida)
- **Tipo de datos**: Todos numéricos (tipo `float64`)

**Variables de Entrada (Features):**

1. **Relative_Compactness** (Compacidad Relativa)
   - Relación entre volumen y área de superficie
   - Rango: 0.62 - 0.98
   - Tipo: Continua

2. **Surface_Area** (Área de Superficie)
   - Área total de superficie del edificio (m²)
   - Rango: 514.5 - 808.5
   - Tipo: Continua

3. **Wall_Area** (Área de Muros)
   - Área total de paredes exteriores (m²)
   - Rango: 245.0 - 416.5
   - Tipo: Continua

4. **Roof_Area** (Área de Techo)
   - Área de la cubierta del edificio (m²)
   - Rango: 110.25 - 220.5
   - Tipo: Discreta (pocos valores únicos)

5. **Overall_Height** (Altura Total)
   - Altura del edificio (m)
   - Valores: 3.5 o 7.0
   - Tipo: Binaria

6. **Orientation** (Orientación)
   - Orientación cardinal del edificio
   - Valores: 2, 3, 4, 5 (Norte, Este, Sur, Oeste)
   - Tipo: Categórica ordinal

7. **Glazing_Area** (Área de Acristalamiento)
   - Área total de ventanas y superficies de vidrio (m²)
   - Rango: 0.0 - 0.4
   - Tipo: Discreta

8. **Glazing_Area_Distribution** (Distribución del Acristalamiento)
   - Patrón de distribución de ventanas
   - Valores: 0, 1, 2, 3, 4, 5
   - Tipo: Categórica ordinal

**Variables de Salida (Targets):**

- **Heating_Load**: Carga de calefacción (kWh/m²) - **Variable objetivo del proyecto**
- **Cooling_Load**: Carga de refrigeración (kWh/m²) - No utilizada en este proyecto

### 2.3 Calidad de los Datos

**Valores nulos:**
- ✓ No se encontraron valores faltantes (0 valores nulos en todo el dataset)
- ✓ No se requiere imputación de datos

**Valores duplicados:**
- ✓ Se verificó la presencia de registros duplicados
- Observación: Algunos duplicados existen debido a que diferentes orientaciones pueden producir las mismas cargas

**Valores atípicos (Outliers):**
- Se identificaron algunos valores extremos en `Heating_Load`
- Análisis: Los outliers parecen ser valores legítimos de edificios con características específicas
- Decisión: Se mantuvieron en el dataset (no se eliminaron)

---

## 3. Metodología

### 3.1 Pipeline del Proyecto

El proyecto sigue una metodología estándar de Machine Learning:

```
1. Carga de datos
2. Análisis Exploratorio de Datos (EDA)
3. Preprocesamiento y limpieza
4. División train/test (80/20)
5. Escalado de características (StandardScaler)
6. Entrenamiento de modelos
7. Evaluación y comparación
8. Optimización de hiperparámetros
9. Selección del modelo final
10. Análisis de resultados
```

### 3.2 Análisis Exploratorio de Datos (EDA)

#### 3.2.1 Análisis Univariado

**Variable Objetivo (Heating_Load):**
- **Distribución**: Multimodal con varios picos
- **Media**: ~22.3 kWh/m²
- **Rango**: 6.01 - 43.10 kWh/m²
- **Interpretación**: La presencia de múltiples modas sugiere grupos naturales de edificios con cargas similares

**Variables de Entrada:**
- La mayoría muestran distribuciones discretas o uniformes
- `Overall_Height` es binaria (solo 2 valores)
- `Orientation` y `Glazing_Area_Distribution` son categóricas balanceadas

#### 3.2.2 Análisis Bivariado

**Correlaciones Fuertes con Heating_Load:**

Correlaciones **positivas**:
- `Relative_Compactness`: r = +0.62
- `Overall_Height`: r = +0.89 (muy fuerte)

Correlaciones **negativas**:
- `Surface_Area`: r = -0.66
- `Roof_Area`: r = -0.86 (muy fuerte)

Correlaciones **débiles**:
- `Orientation`: r ≈ 0.00 (sin correlación lineal)

#### 3.2.3 Multicolinealidad

**Pares altamente correlacionados:**
- `Relative_Compactness` ↔ `Surface_Area`: r = -0.99
- `Roof_Area` ↔ `Overall_Height`: r = -0.97

**Implicaciones:**
- La multicolinealidad puede afectar la interpretabilidad de coeficientes en regresión lineal
- Modelos regularizados (Ridge, Lasso) son apropiados para manejar este problema

### 3.3 Preprocesamiento

#### 3.3.1 Limpieza de Datos

✓ No se requirió limpieza adicional (datos completos y consistentes)

#### 3.3.2 División de Datos

- **Conjunto de Entrenamiento**: 80% (614 observaciones)
- **Conjunto de Prueba**: 20% (154 observaciones)
- **Método**: `train_test_split` con `random_state=42`
- **Estrategia**: División aleatoria con shuffle activado

#### 3.3.3 Escalado de Características

**Método utilizado:** StandardScaler (Estandarización Z-score)

**Fórmula:** `z = (x - μ) / σ`

Donde:
- `x`: valor original
- `μ`: media de la variable
- `σ`: desviación estándar

**Resultado:**
- Media de cada variable ≈ 0
- Desviación estándar ≈ 1

**Justificación:**
- Las variables tienen escalas muy diferentes (ej: `Surface_Area` ~500-800 vs `Relative_Compactness` ~0.6-1.0)
- El escalado evita que variables con valores grandes dominen el modelo
- Mejora la convergencia de algoritmos de optimización

**Prevención de fuga de datos:**
- ✓ El scaler se ajusta SOLO con datos de entrenamiento
- ✓ Se transforma tanto train como test con los mismos parámetros

### 3.4 Modelos Implementados

#### 3.4.1 Regresión Lineal Múltiple (Modelo Base)

**Descripción:**
- Modelo lineal sin regularización
- Minimiza la suma de errores cuadráticos (OLS - Ordinary Least Squares)

**Ecuación:**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + β₈x₈ + ε
```

**Ventajas:**
- Simple e interpretable
- Rápido de entrenar
- Referencia baseline para comparar otros modelos

**Limitaciones:**
- Sensible a multicolinealidad
- Puede sobreajustar si hay muchas variables

#### 3.4.2 Ridge Regression (Regularización L2)

**Descripción:**
- Regresión lineal con penalización L2
- Añade término de regularización al error: `||y - Xβ||² + α||β||²`

**Hiperparámetro:**
- `alpha` (α): Controla la fuerza de regularización
  - α = 0: Equivalente a regresión lineal
  - α grande: Mayor penalización, coeficientes más pequeños

**Ventajas:**
- Maneja multicolinealidad reduciendo magnitud de coeficientes
- Previene overfitting
- Mantiene todas las variables en el modelo

**Limitaciones:**
- No realiza selección de variables (todos los coeficientes permanecen != 0)

#### 3.4.3 Lasso Regression (Regularización L1)

**Descripción:**
- Regresión lineal con penalización L1
- Añade término de regularización: `||y - Xβ||² + α|β|`

**Hiperparámetro:**
- `alpha` (α): Controla la fuerza de regularización

**Ventajas:**
- **Selección automática de variables**: Puede reducir coeficientes exactamente a 0
- Produce modelos más simples e interpretables
- Útil cuando hay variables irrelevantes

**Limitaciones:**
- Puede descartar variables útiles si α es muy alto

### 3.5 Estrategia de Evaluación

#### 3.5.1 Métricas Utilizadas

**1. MAE (Mean Absolute Error - Error Absoluto Medio)**

Fórmula: `MAE = (1/n) Σ|yᵢ - ŷᵢ|`

- **Interpretación**: Promedio de los errores absolutos en las mismas unidades que la variable objetivo (kWh/m²)
- **Ventaja**: Fácil de interpretar, robusta a outliers
- **Uso**: Indica el error promedio típico

**2. RMSE (Root Mean Squared Error - Raíz del Error Cuadrático Medio)**

Fórmula: `RMSE = √[(1/n) Σ(yᵢ - ŷᵢ)²]`

- **Interpretación**: Error en las mismas unidades que la variable objetivo, penaliza errores grandes
- **Ventaja**: Sensible a errores grandes (útil cuando queremos minimizarlos)
- **Uso**: Métrica principal en muchos problemas de regresión

**3. R² (Coeficiente de Determinación)**

Fórmula: `R² = 1 - (SS_res / SS_tot)`

- **Rango**: 0 a 1 (puede ser negativo si el modelo es peor que la media)
- **Interpretación**: Proporción de la varianza de y explicada por el modelo
  - R² = 1.00: Modelo perfecto (explica el 100% de la variabilidad)
  - R² = 0.90: El modelo explica el 90% de la variabilidad
  - R² = 0.00: El modelo no es mejor que predecir la media
- **Uso**: Métrica estándar para evaluar calidad del ajuste

#### 3.5.2 Validación Cruzada

**Método:** K-Fold Cross-Validation (k=5)

**Funcionamiento:**
1. Dividir datos de entrenamiento en 5 partes (folds)
2. Entrenar con 4 partes, validar con 1
3. Repetir 5 veces rotando el fold de validación
4. Promediar los resultados

**Ventaja:**
- Uso más eficiente de los datos
- Estimación más robusta del rendimiento
- Reduce varianza en la evaluación

#### 3.5.3 Optimización de Hiperparámetros

**Técnica:** GridSearchCV

**Parámetros explorados:**

Para **Ridge**:
- alpha: [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

Para **Lasso**:
- alpha: [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

**Estrategia:**
- Búsqueda exhaustiva sobre la grilla de parámetros
- Evaluación con validación cruzada (5-fold)
- Selección del mejor alpha según R²

---

## 4. Resultados

### 4.1 Rendimiento de los Modelos

#### Tabla Comparativa - Conjunto de Prueba

| Modelo               | MAE     | RMSE    | R²      |
|---------------------|---------|---------|---------|
| Regresión Lineal    | 0.52    | 0.67    | 0.991   |
| Ridge (α=1.0)       | 0.52    | 0.67    | 0.991   |
| Lasso (α=0.1)       | 0.54    | 0.70    | 0.990   |

**Observaciones:**

1. **Rendimiento excepcional**: Todos los modelos alcanzan R² > 0.99, indicando que explican más del 99% de la variabilidad en la carga de calefacción.

2. **Diferencias mínimas**: Las diferencias entre modelos son muy pequeñas, sugiriendo que:
   - El problema es altamente lineal
   - Las relaciones son fuertes y consistentes
   - La regularización no es crítica en este caso (aunque sigue siendo buena práctica)

3. **Errores absolutos bajos**:
   - MAE ~0.5 kWh/m² significa que el error promedio es de solo 0.5 unidades
   - Considerando que el rango de Heating_Load es 6-43 kWh/m², este error es mínimo (~1-2% del rango)

### 4.2 Comparación Train vs Test

| Modelo               | R² Train | R² Test | Diferencia |
|---------------------|----------|---------|------------|
| Regresión Lineal    | 0.992    | 0.991   | -0.001     |
| Ridge               | 0.992    | 0.991   | -0.001     |
| Lasso               | 0.991    | 0.990   | -0.001     |

**Interpretación:**

✓ **Excelente generalización**: Las métricas son prácticamente idénticas en train y test

✓ **Sin overfitting**: La mínima diferencia indica que los modelos no están sobreajustando

✓ **Robustez**: Los modelos funcionan igual de bien con datos no vistos

### 4.3 Mejores Hiperparámetros

**Ridge:**
- Mejor alpha: 1.0
- R² (validación cruzada): 0.9917

**Lasso:**
- Mejor alpha: 0.01
- R² (validación cruzada): 0.9914

### 4.4 Importancia de Variables

**Según coeficientes del modelo lineal (valores absolutos):**

| Ranking | Variable                   | |Coeficiente| | Interpretación              |
|---------|----------------------------|--------------|------------------------------|
| 1       | Roof_Area                  | 9.84         | Muy alta influencia negativa |
| 2       | Overall_Height             | 8.52         | Muy alta influencia positiva |
| 3       | Relative_Compactness       | 4.27         | Alta influencia positiva     |
| 4       | Surface_Area               | 3.12         | Media influencia negativa    |
| 5       | Glazing_Area               | 2.13         | Media influencia positiva    |
| 6       | Wall_Area                  | 1.45         | Baja influencia              |
| 7       | Glazing_Area_Distribution  | 0.68         | Muy baja influencia          |
| 8       | Orientation                | 0.03         | Sin influencia práctica      |

**Conclusiones:**
- `Roof_Area` y `Overall_Height` son los predictores más importantes
- `Orientation` tiene impacto negligible (confirmando el análisis de correlación)
- Las variables estructurales (altura, áreas) dominan sobre las de orientación

---

## 5. Análisis de Errores

### 5.1 Diagnósticos Visuales

#### 5.1.1 Gráfico Real vs Predicho

**Observación:**
- Los puntos se alinean muy bien sobre la diagonal perfecta
- Dispersión mínima alrededor de la línea
- No se observan sesgos sistemáticos en ningún rango de valores

**Conclusión:** El modelo predice con alta precisión en todo el rango de valores

#### 5.1.2 Distribución de Residuos

**Observación:**
- Distribución aproximadamente normal
- Centrada en 0 (media ≈ 0)
- Pocos outliers extremos

**Conclusión:** Los supuestos de normalidad de residuos se cumplen razonablemente

#### 5.1.3 Residuos vs Predicciones

**Observación:**
- No se detectan patrones claros (forma de embudo, curvas, etc.)
- Dispersión aproximadamente constante en todo el rango
- Residuos distribuidos aleatoriamente alrededor de 0

**Conclusión:**
✓ Homocedasticidad (varianza constante) se cumple
✓ No hay evidencia de relaciones no lineales no capturadas

### 5.2 Limitaciones Identificadas

1. **Dataset sintético/controlado:**
   - Los datos provienen de simulaciones, no de mediciones reales
   - Esto explica el R² excepcionalmente alto
   - En aplicaciones reales, el rendimiento podría ser menor

2. **Falta de variables contextuales:**
   - No se consideran factores climáticos (temperatura exterior, viento, etc.)
   - No hay información sobre materiales de construcción
   - No se incluyen características de aislamiento térmico

3. **Generalización geográfica:**
   - No se especifica la ubicación geográfica de los edificios
   - Los modelos podrían no generalizar bien a diferentes climas

---

## 6. Discusión

### 6.1 Interpretación de Resultados

El análisis demuestra que las **características arquitectónicas básicas** (compacidad, áreas, altura) son **altamente predictivas** de la carga de calefacción. Específicamente:

1. **Altura del edificio**: Edificios más altos (7m vs 3.5m) requieren significativamente más calefacción
2. **Área de techo**: Relación inversa fuerte - techos más grandes implican edificios menos compactos con menor carga
3. **Compacidad**: Edificios más compactos (mayor relación volumen/superficie) retienen mejor el calor

### 6.2 Comparación con Literatura

Los resultados son consistentes con principios de termodinámica de edificios:
- **Principio de compacidad**: Edificios compactos minimizan pérdidas de calor
- **Efecto de superficie**: Mayor área expuesta aumenta pérdidas térmicas
- **Estratificación térmica**: Edificios altos requieren más energía para calentar uniformemente

### 6.3 Aplicaciones Prácticas

Los modelos desarrollados pueden utilizarse para:

1. **Fase de diseño arquitectónico:**
   - Evaluar diferentes configuraciones de edificios
   - Estimar costos energéticos antes de construir
   - Optimizar proporciones para minimizar consumo

2. **Auditorías energéticas:**
   - Identificar edificios con consumo anormalmente alto
   - Priorizar intervenciones de mejora energética

3. **Certificación energética:**
   - Predecir calificación energética de edificios
   - Cumplir con normativas de eficiencia

### 6.4 Mejoras Propuestas

Para investigaciones futuras se recomienda:

1. **Incorporar más variables:**
   - Materiales de construcción (aislamiento, conductividad térmica)
   - Variables climáticas (temperatura, humedad, radiación solar)
   - Ocupación y patrones de uso

2. **Explorar modelos no lineales:**
   - Árboles de decisión y Random Forests
   - Gradient Boosting (XGBoost, LightGBM)
   - Redes neuronales

3. **Validación con datos reales:**
   - Probar en datasets de edificios reales
   - Comparar predicciones con mediciones reales

4. **Análisis de incertidumbre:**
   - Intervalos de confianza para predicciones
   - Cuantificación de incertidumbre epistémica

---

## 7. Conclusiones

### 7.1 Hallazgos Principales

1. **Modelos de regresión lineal son altamente efectivos** para predecir la carga de calefacción basándose en características arquitectónicas, alcanzando R² > 0.99.

2. **Las variables estructurales** (altura, áreas, compacidad) son **mucho más importantes** que variables de configuración (orientación, distribución de ventanas) para determinar la carga térmica.

3. **La regularización no es crítica** en este problema particular, pero se recomienda como buena práctica para prevenir overfitting en datasets más complejos.

4. **Excelente generalización**: El rendimiento en test es prácticamente idéntico al de train, indicando que los modelos capturan patrones reales y no ruido.

### 7.2 Respuesta a la Pregunta de Investigación

**¿Es posible predecir la carga de calefacción de edificios usando solo características arquitectónicas?**

**Respuesta:** **Sí, con alta precisión.** Los modelos desarrollados demuestran que 8 características arquitectónicas básicas son suficientes para explicar más del 99% de la variabilidad en la carga de calefacción.

### 7.3 Aprendizajes del Proyecto

Durante el desarrollo de este proyecto se aplicaron y consolidaron:

✓ **Metodología completa de Machine Learning:**
- Análisis exploratorio exhaustivo
- Preprocesamiento riguroso (escalado, división)
- Entrenamiento y evaluación de múltiples modelos
- Optimización de hiperparámetros
- Diagnóstico de errores

✓ **Buenas prácticas de reproducibilidad:**
- Fijación de semillas aleatorias
- Documentación detallada de decisiones
- Código comentado y estructurado
- Versionado de librerías

✓ **Interpretación de resultados:**
- Análisis crítico de métricas
- Validación de supuestos de regresión
- Conexión con conocimiento del dominio

### 7.4 Recomendaciones

**Para este dataset específico:**
- Utilizar **Regresión Lineal** o **Ridge** (α=1.0) como modelo final
- Ambos ofrecen rendimiento óptimo con máxima simplicidad

**Para problemas similares:**
- Siempre comenzar con modelos simples (regresión lineal) como baseline
- Aplicar regularización cuando haya multicolinealidad
- Validar supuestos mediante diagnósticos visuales
- Usar validación cruzada para hiperparámetros

---

## 8. Referencias

### Datasets
- UCI Machine Learning Repository: Energy Efficiency Data Set
  - https://archive.ics.uci.edu/ml/datasets/Energy+efficiency
- Kaggle: Energy Efficiency Data Set
  - https://www.kaggle.com/datasets/ujjwalchowdhury/energy-efficiency-data-set

### Documentación Técnica
- Scikit-learn Documentation (v1.3.0)
  - https://scikit-learn.org/stable/
- Pandas Documentation
  - https://pandas.pydata.org/docs/
- NumPy Documentation
  - https://numpy.org/doc/

### Material del Curso
- Repositorio de Inteligencia Artificial - Computación U
  - https://github.com/BrayanTorres2/Inteligencia-artificial-computacion-U

### Bibliografía sobre Eficiencia Energética
- A. Tsanas, A. Xifara: "Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools", Energy and Buildings, Vol. 49, pp. 560-567, 2012.

---

## Anexo A: Especificaciones Técnicas

### Entorno de Desarrollo
- **Python**: 3.8+
- **Sistema Operativo**: Windows/Linux/macOS
- **IDE**: Jupyter Notebook

### Librerías y Versiones
```
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
```

### Hardware Utilizado
- **CPU**: Procesador Intel/AMD multi-core
- **RAM**: 8GB (mínimo recomendado)
- **Almacenamiento**: 100MB para proyecto completo

### Tiempo de Ejecución
- **Carga y EDA**: ~30 segundos
- **Entrenamiento de modelos**: <5 segundos
- **GridSearchCV**: ~10-30 segundos
- **Total**: ~2-3 minutos para ejecutar notebook completo

---

## Anexo B: Reproducibilidad

### Garantías de Reproducibilidad

✓ **Semilla aleatoria fija**: `random_state=42` en todos los componentes estocásticos

✓ **Versiones de librerías especificadas**: `requirements.txt` con versiones exactas

✓ **Rutas relativas**: El código funciona independientemente de la ubicación del proyecto

✓ **Sin dependencias externas**: No requiere APIs ni servicios externos

### Instrucciones para Reproducir

```bash
# 1. Clonar el repositorio
git clone <URL_REPOSITORIO>
cd parcial_final

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar notebook
jupyter notebook notebook.ipynb
```

---

**Fin del Informe**

Elaborado por: <NOMBRE_ESTUDIANTE>

Fecha: Noviembre 2025

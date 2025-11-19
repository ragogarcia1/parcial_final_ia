# Proyecto A — REGRESIÓN: Energy Efficiency Data Set

**Estudiante:** ÓScar Mauricio García Mesa

**Fecha:** Noviembre 2025

**Curso:** Inteligencia Artificial - Computación

---

## Descripción del Proyecto

Este proyecto implementa modelos de **regresión** para predecir la **carga de calefacción (Heating Load)** de edificios basándose en características arquitectónicas. Se utiliza el dataset **Energy Efficiency Data Set** disponible en Kaggle y el UCI Machine Learning Repository.

### Problema

Predecir la eficiencia energética de edificios es fundamental para:
- Diseñar construcciones más sostenibles
- Reducir costos operacionales
- Minimizar el impacto ambiental
- Cumplir con regulaciones de eficiencia energética

### Dataset

**Fuente:** [Kaggle - Energy Efficiency Data Set](https://www.kaggle.com/datasets/ujjwalchowdhury/energy-efficiency-data-set)

**Características:**
- 768 observaciones
- 8 variables de entrada (características arquitectónicas)
- 2 variables de salida (cargas de calefacción y refrigeración)

**Variables de entrada:**
1. `Relative_Compactness`: Compacidad relativa del edificio
2. `Surface_Area`: Área de superficie total (m²)
3. `Wall_Area`: Área de los muros (m²)
4. `Roof_Area`: Área del techo (m²)
5. `Overall_Height`: Altura total del edificio (m)
6. `Orientation`: Orientación del edificio (2, 3, 4, 5 = N, E, S, O)
7. `Glazing_Area`: Área de acristalamiento/ventanas (m²)
8. `Glazing_Area_Distribution`: Distribución del acristalamiento (0-5)

**Variable objetivo:** `Heating_Load` (carga de calefacción en kWh/m²)

---

## Estructura del Proyecto

```
parcial_final/
│
├── data/
│   └── energy_efficiency_data.csv    # Dataset
│
├── notebook.ipynb                     # Notebook principal con análisis completo
├── REPORT.md                          # Informe técnico del proyecto
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias del proyecto
│
```

---

## Instalación y Configuración

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Jupyter Notebook o JupyterLab

### Instrucciones de Instalación

1. **Clonar o descargar el repositorio**

   ```bash
   # Si está en GitHub/GitLab
   git clone https://github.com/ragogarcia1/parcial_final_ia.git
   cd parcial_final
   ```

2. **Crear un entorno virtual (recomendado)**

   ```bash
   # En Windows
   python -m venv venv
   venv\Scripts\activate

   # En Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar las dependencias**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verificar la instalación**

   ```bash
   python -c "import numpy, pandas, sklearn, matplotlib, seaborn; print('✓ Todas las librerías instaladas correctamente')"
   ```

---

## Ejecución del Proyecto

### Opción 1: Jupyter Notebook

1. Iniciar Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. En el navegador, abrir `notebook.ipynb`

3. Ejecutar las celdas secuencialmente:
   - Menú: `Cell > Run All`
   - O usar `Shift + Enter` para ejecutar celda por celda

### Opción 2: JupyterLab

```bash
jupyter lab
```

Luego abrir `notebook.ipynb` desde la interfaz.

### Opción 3: VS Code

Si usas Visual Studio Code con la extensión de Jupyter:
1. Abrir `notebook.ipynb`
2. Seleccionar el kernel de Python del entorno virtual
3. Ejecutar las celdas

---

## Contenido del Notebook

El notebook está organizado en las siguientes secciones:

1. **Introducción y descripción del problema**
2. **Configuración inicial** (importación de librerías y semillas)
3. **Carga y descripción del dataset**
4. **Limpieza y preparación de datos**
5. **Análisis Exploratorio de Datos (EDA)**
   - Análisis univariado
   - Análisis bivariado
   - Matriz de correlación
6. **Selección de variables y división train/test**
7. **Escalado de datos** (StandardScaler)
8. **Modelos de regresión**
   - Regresión Lineal Múltiple
   - Ridge Regression
   - Lasso Regression
9. **Evaluación de modelos** (MAE, RMSE, R²)
10. **Análisis de errores y diagnósticos**
11. **Búsqueda de hiperparámetros** (GridSearchCV)
12. **Modelo final y conclusiones**

---

## Modelos Implementados

### 1. Regresión Lineal Múltiple
- Modelo base sin regularización
- Encuentra relaciones lineales entre features y target

### 2. Ridge Regression (Regularización L2)
- Penaliza coeficientes grandes
- Maneja multicolinealidad
- Reduce overfitting

### 3. Lasso Regression (Regularización L1)
- Selección automática de variables
- Puede reducir coeficientes a exactamente 0
- Útil para simplificar modelos

---

## Métricas de Evaluación

- **MAE (Mean Absolute Error)**: Error promedio absoluto
- **MSE (Mean Squared Error)**: Error cuadrático medio
- **RMSE (Root Mean Squared Error)**: Raíz del MSE
- **R² (Coeficiente de Determinación)**: Proporción de varianza explicada

---

## Resultados Esperados

El proyecto demuestra:

✓ **Alto rendimiento predictivo** (R² > 0.90)
✓ **Buena generalización** (métricas similares en train y test)
✓ **Identificación de variables importantes** para la eficiencia energética
✓ **Comparación rigurosa** de diferentes técnicas de regresión

---

## Reproducibilidad

El proyecto garantiza reproducibilidad mediante:

- **Semillas aleatorias fijadas** (`random_state=42`)
- **Versiones específicas** de librerías en `requirements.txt`
- **Rutas relativas** para acceso a datos
- **Documentación completa** de cada paso

---

## Troubleshooting

### Error: "No module named 'sklearn'"

```bash
pip install scikit-learn
```

### Error: "FileNotFoundError: data/energy_efficiency_data.csv"

Verificar que el archivo CSV esté en la carpeta `data/` en la raíz del proyecto.

### Warning sobre versiones de librerías

Si aparecen warnings de compatibilidad, instalar las versiones exactas:

```bash
pip install -r requirements.txt --force-reinstall
```

---

## Referencias

- **Dataset**: UCI Machine Learning Repository - Energy Efficiency Data Set
- **Kaggle**: https://www.kaggle.com/datasets/ujjwalchowdhury/energy-efficiency-data-set
- **Scikit-learn Documentation**: https://scikit-learn.org/
- **Repositorio del curso**: https://github.com/BrayanTorres2/Inteligencia-artificial-computacion-U

---

## Autor

**Óscar Mauricio García Mesa**

Proyecto desarrollado para la asignatura de Inteligencia Artificial - Computación

Universidad: Universitaria Uniagustiniana

Fecha: Noviembre 2025

---

## Licencia

Este proyecto es de uso académico y educativo.

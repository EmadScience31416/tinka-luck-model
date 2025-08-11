## 📓 Análisis de La Tinka

Se realiza un estudio exhaustivo de los resultados históricos de **La Tinka**, un sorteo de lotería en Perú, con el objetivo de identificar patrones y desarrollar estrategias más informadas para la selección de números.  

El flujo de trabajo incluye:

1. **Carga y descripción de datos históricos**  
   - Fechas de sorteo, números principales ordenados de menor a mayor (*B1* a *B6*) y número adicional (*Boliyapa*).  
   - Rango de fechas analizadas: 03/01/2024 a 10/08/2025.

2. **Exploración de patrones numéricos**  
   - Distribución de números **pares** e **impares**.  
   - Distribución de **números bajos** (≤25) y **altos** (>25).  
   - Frecuencia de aparición de **números consecutivos**.  
   - Análisis de **transiciones** entre sorteos (qué números suelen seguir a otros).  
   - Estudio de **repeticiones** de un sorteo a otro.

3. **Análisis por posición de bola**  
   - Rangos típicos por cada posición (*B1* a *B6*).  
   - Rangos óptimos para cada posición.  
   - Histogramas de frecuencia por posición.

4. **Definición de reglas de selección**  
   - Establecimiento de condiciones basadas en hallazgos estadísticos.  
   - Objetivo: reducir el espacio de combinaciones posibles priorizando las más frecuentes.

5. **Simulación de Montecarlo**  
   - Comparación entre seleccionar números aleatoriamente y seleccionar números con reglas basadas en datos.  
   - Estimación de probabilidades de acierto bajo diferentes estrategias.

---

🔗 **Ver notebook en GitHub:**  
[📂 `tinka_analysis.ipynb`](Analysis/tinka_analysis.ipynb)  

🔗 **Ver notebook en nbviewer (mejor visualización):**  
[🌐 Abrir en nbviewer](https://nbviewer.org/github/EmadScience31416/tinka-luck-model/blob/main/Analysis/tinka_analysis.ipynb)


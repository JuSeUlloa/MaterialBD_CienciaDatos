# Actividad: Modelos de clasificación para detección temprana de fraude en entidades bancarias


En esta actividad se analiza un problema real de detección de fraude en transacciones bancarias, un reto crítico para las entidades financieras debido a su impacto económico, operativo y reputacional. La institución dispone de un conjunto de datos históricos, el cual reúne información asociada a transacciones realizadas por sus clientes, incluyendo variables que describen patrones de uso, comportamiento y características de las operaciones.

El objetivo del reto es formular este escenario como un problema de clasificación, en el cual se busca determinar si una transacción corresponde o no a un caso de fraude. Este contexto permite aplicar modelos de aprendizaje supervisado para identificar patrones anómalos, evaluar su desempeño mediante métricas adecuadas y analizar el equilibrio entre detección oportuna y reducción de falsas alarmas, aspectos fundamentales en sistemas de prevención de fraude en el sector financiero.


El objetivo del reto es formular este escenario como un problema de clasificación, en el cual se busca determinar si una transacción corresponde o no a un caso de fraude. Este contexto permite aplicar modelos de aprendizaje supervisado para identificar patrones anómalos, evaluar su desempeño mediante métricas adecuadas y analizar el equilibrio entre detección oportuna y reducción de falsas alarmas, aspectos fundamentales en sistemas de prevención de fraude en el sector financiero.

Desrrollo de la actividad:

*   Carga de conjuntos de datos de entrenamiento y prueba.
*   Realice la exploración y limpieza de los datos.
*   Aplique ingeniería de variables cuando sea pertinente.
*   Implemente modelos de clasificación supervisada.
*   Evalúe los modelos usando la métrica AUC-ROC.
*   Seleccione el mejor modelo según desempeño y estabilidad.
*   Realice el cargue de predicciones en la plataforma  Kaggle.


link de las predicciones : https://www.kaggle.com/competitions/modelos-de-fraude/submissions

## Pipelines 

Modelo XgBoosting y Ensamble 

[modelo Xgboots Ensamble.py](./modelos-Xgboots-Ensamble.py)

Modelo Random Forest 

[Modelo Random Forest.py](./modelo-RForest.py)

Modelo Regresión Logística

[Modelo Regresión Logística.py](./modelo-logistica.py)
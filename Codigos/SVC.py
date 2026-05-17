import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve,auc, ConfusionMatrixDisplay

# Semilla para reproducibilidad
ra = 12

# Configuracion general de estilo para matplotlib
plt.rcParams.update({'font.family': 'DejaVu Sans','axes.spines.top': False,'axes.spines.right': False, 'axes.labelsize': 11, 'axes.titlesize': 13, 'figure.facecolor': 'white'})

# Carga y preprocesamiento de datos

# Se carga el dataset desde un archivo Excel
df = pd.read_excel('diabetes_procesado.xlsx')

# Variable objetivo que contiene la clasificacion
TARGET = 'Resultado'

# Se seleccionan todas las columnas excepto la variable objetivo
features = [c for c in df.columns if c != TARGET]

# Matriz de caracteristicas
X = df[features].values

# Vector de etiquetas
y = df[TARGET].values

# Division del dataset en entrenamiento y prueba
# test_size=0.20 -> 20% para pruebas
# stratify=y -> mantiene la proporcion de clases
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=ra, stratify=y)

# Escalado de datos
# SVM es sensible a diferencias de escala entre variables,
# por ello se utiliza StandardScaler para normalizar

scaler = StandardScaler()

# Ajusta y transforma los datos de entrenamiento
X_train_sc = scaler.fit_transform(X_train)

# Solo transforma los datos de prueba
X_test_sc = scaler.transform(X_test)

# Entrenamiento del modelo SVC

# Se crea un clasificador SVC con:
# kernel='rbf'  -> separacion no lineal
# C=1e6         -> penalizacion muy alta al error
# gamma='scale' -> ajuste automatico del kernel
# probability=True -> habilita probabilidades

svc = SVC(kernel='rbf', C=1e6, gamma='scale', probability=True, random_state=ra)

# Entrenamiento del modelo
svc.fit(X_train_sc, y_train)

# Predicciones y evaluacion

# Predicciones binarias
y_pred_svc = svc.predict(X_test_sc)

# Probabilidad estimada de pertenecer a la clase positiva
y_prob_svc = svc.predict_proba(X_test_sc)[:, 1]

# Accuracy manual
acc_svc = np.mean(y_pred_svc == y_test)

# Matriz de confusion
cm_svc = confusion_matrix(y_test, y_pred_svc)

# Reporte completo de metricas
report_svc = classification_report(y_test, y_pred_svc, target_names=['No Diabetes', 'Diabetes'], output_dict=True)

# Calculo de curva ROC
fpr_svc, tpr_svc, _ = roc_curve(y_test, y_prob_svc)

# Area bajo la curva ROC
auc_svc = auc(fpr_svc, tpr_svc)

# Impresion de resultados

# Paleta de colores utilizada en las graficas
BLUE   = "#1a3c6e"
RED    = "#c0392b"
GRAY   = "#ecf0f1"
ACCENT = "#2980b9"
GREEN  = "#27ae60"
ORANGE = "#e67e22"

print(f"\nSVC con Accuracy: {acc_svc:.4f} y AUC: {auc_svc:.4f}")

# Numero de vectores de soporte por clase
print(f"Vectores de soporte: {svc.n_support_} (No Diab. / Diab.)")

# Total de vectores de soporte
print(f"Total vectores de soporte: {sum(svc.n_support_)}")

print("\nReporte de clasificacion:")

# Muestra precision, recall y F1-score
print(classification_report(y_test, y_pred_svc, target_names=['No Diabetes', 'Diabetes']))

# Creacion de figura general

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')

# Se crea una cuadricula de 2 filas x 3 columnas
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# Titulo general de la figura
fig.text(0.5, 0.97, "Support Vector Classifier (SVC) para Diabetes Dataset", ha='center', va='top', fontsize=16, fontweight='bold',color=BLUE)

# Curva ROC

ax = fig.add_subplot(gs[0, 0])

# Curva ROC
ax.plot(fpr_svc, tpr_svc, color=ACCENT, lw=2.5, label=f'ROC (AUC = {auc_svc:.3f})')

# Linea diagonal de referencia
ax.plot([0, 1], [0, 1], 'k--', lw=1)

# Area sombreada bajo la curva
ax.fill_between(fpr_svc, tpr_svc, alpha=0.15, color=ACCENT)

ax.set_title("Curva ROC", fontweight='bold')
ax.set_xlabel("Tasa de Falsos Positivos")
ax.set_ylabel("Tasa de Verdaderos Positivos")

ax.legend()
ax.grid(alpha=0.3)

# Matriz de confusion

ax = fig.add_subplot(gs[0, 1])

# Visualizacion de matriz de confusion
disp = ConfusionMatrixDisplay(cm_svc, display_labels=['No Diabetes', 'Diabetes'])

disp.plot(ax=ax, colorbar=False, im_kw={'cmap': plt.cm.Blues})

ax.set_title("Matriz de Confusion", fontweight='bold')

# Rotacion de etiquetas
ax.tick_params(axis='x', labelrotation=15)

# Grafica de metricas por clase

ax = fig.add_subplot(gs[0, 2])

# Clases
classes = ['No Diabetes', 'Diabetes']

# Metricas a comparar
metrics_names = ['precision', 'recall', 'f1-score']
metrics_labels = ['Precision', 'Recall', 'F1-score']

# Posiciones en eje X
x = np.arange(len(classes))

# Ancho de barras
w = 0.25

# Colores para cada metrica
colors_bar = [BLUE, GREEN, ORANGE]

# Construccion de barras
for i, (m, lbl) in enumerate(zip(metrics_names, metrics_labels)):

    # Obtiene los valores de cada metrica
    vals = [report_svc[c][m] for c in classes]

    # Grafica las barras
    ax.bar(x + i * w, vals, w, label=lbl, color=colors_bar[i], alpha=0.85)

ax.set_xticks(x + w)
ax.set_xticklabels(classes)

ax.set_ylim(0, 1.15)

ax.set_ylabel("Valor")

ax.set_title("Metricas por Clase", fontweight='bold')

ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Curva de aprendizaje

ax = fig.add_subplot(gs[1, 0])

# Calcula desempeno del modelo con diferentes tamanos
# de entrenamiento para detectar overfitting

train_sizes, train_scores, test_scores = learning_curve(SVC(kernel='rbf', C=1e6, gamma='scale', probability=True, random_state=ra), X_train_sc, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)

# Promedios
train_mean = train_scores.mean(axis=1)
test_mean  = test_scores.mean(axis=1)

# Desviaciones estandar
train_std  = train_scores.std(axis=1)
test_std   = test_scores.std(axis=1)

# Curva de entrenamiento
ax.plot(train_sizes, train_mean, color=RED, lw=2,label='Entrenamiento')

# Curva de validacion
ax.plot(train_sizes, test_mean, color=GREEN, lw=2, label='Validacion cruzada')

# Area de incertidumbre entrenamiento
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15,color=RED)

# Area de incertidumbre validacion
ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15,color=GREEN)

ax.set_title("Curva de Aprendizaje", fontweight='bold')

ax.set_xlabel("Tamano del conjunto de entrenamiento")
ax.set_ylabel("Exactitud")

ax.legend()
ax.grid(alpha=0.3)

# Distribucion de probabilidades
ax = fig.add_subplot(gs[1, 1])

# Probabilidades reales clase negativa
prob_neg = y_prob_svc[y_test == 0]

# Probabilidades reales clase positiva
prob_pos = y_prob_svc[y_test == 1]

# Histograma clase negativa
ax.hist(prob_neg, bins=20, color=BLUE, alpha=0.65, label='No Diabetes (real)')

# Histograma clase positiva
ax.hist(prob_pos, bins=20, color=RED, alpha=0.65, label='Diabetes (real)')

# Umbral de decision
ax.axvline(0.5, color='black', linestyle='--', lw=1.5, label='Umbral 0.5')

ax.set_title("Distribucion de Probabilidades\nPredichas", fontweight='bold')

ax.set_xlabel("P(Diabetes)")
ax.set_ylabel("Frecuencia")

ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel final vacio para texto o resumen

ax = fig.add_subplot(gs[1, 2])

# Se ocultan los ejes para utilizar el espacio como resumen
ax.axis('off')

# Mostrar todas las graficas
plt.show()
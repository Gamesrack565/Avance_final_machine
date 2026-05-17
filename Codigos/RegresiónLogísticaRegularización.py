import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Cargar y separar los datos
df = pd.read_excel('diabetes_procesado.xlsx')
X = df.drop(columns=['Resultado'])
y = df['Resultado']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 2. Escalar los datos
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Definir los valores de C que queremos probar
parametros = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100]
}

print("Buscando la mejor configuración para Regresión Logística...")

# 4. Crear y ejecutar GridSearchCV
grid_log = GridSearchCV(
    estimator=LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    param_grid=parametros,
    scoring='accuracy',
    cv=5,
    n_jobs=1
)

grid_log.fit(X_train_scaled, y_train)

# AQUÍ ESTÁ EL MODELO GANADOR
mejor_log_reg = grid_log.best_estimator_

print(f"\n¡Búsqueda terminada! El mejor parámetro encontrado fue: {grid_log.best_params_}")

# 5. Evaluar el modelo optimizado
y_pred_mejor_log = mejor_log_reg.predict(X_test_scaled)

print("\n--- EVALUACIÓN: REGRESIÓN LOGÍSTICA OPTIMIZADA ---")
print(f"Exactitud (Accuracy): {accuracy_score(y_test, y_pred_mejor_log) * 100:.2f}%\n")
print(classification_report(y_test, y_pred_mejor_log))

# =========================================================
# 8. VISUALIZACIONES
# =========================================================
print("\nGenerando gráficos...")

# --- Gráfico 1: Matriz de Confusión ---
plt.figure(figsize=(8, 6))
matriz_log = confusion_matrix(y_test, y_pred_mejor_log)
sns.heatmap(matriz_log, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Negativo', 'Positivo'],
            yticklabels=['Negativo', 'Positivo'])
plt.title('Matriz de Confusión - Regresión Logística')
plt.xlabel('Predicción del Modelo')
plt.ylabel('Valor Real (Paciente)')
plt.show()

# --- Gráfico 2: Importancia de las Variables (Coeficientes) ---
# CORRECCIÓN: Extraemos los coeficientes de 'mejor_log_reg', no de 'log_reg'
coeficientes = mejor_log_reg.coef_[0]

df_coeficientes = pd.DataFrame({
    'Variable': X.columns,
    'Coeficiente': coeficientes
}).sort_values(by='Coeficiente', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coeficiente', y='Variable', data=df_coeficientes, hue='Variable', palette='coolwarm', legend=False)
plt.title('Impacto de cada variable en el diagnóstico (Coeficientes L2)')
plt.xlabel('Peso (Hacia la derecha aumenta el riesgo, hacia la izquierda lo disminuye)')
plt.ylabel('Variables Clínicas')
plt.axvline(0, color='black', linestyle='--')
plt.tight_layout()
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar el dataset original
df = pd.read_excel('diabetes_procesado.xlsx')

print(f"Filas iniciales: {len(df)}")

# 2. ELIMINAR filas solo si Glucosa es 0
df = df[df['Glucose'] != 0]

# 3. REEMPLAZAR los ceros por las medias en las otras columnas
df['BloodPressure'] = df['BloodPressure'].replace(0, 69)
df['SkinThickness'] = df['SkinThickness'].replace(0, 21)
df['Insulin'] = df['Insulin'].replace(0, 80)
df['BMI'] = df['BMI'].replace(0, 31.9)

# 4. (Opcional) Aplicar los cambios de nombre y texto que vimos en el paso anterior
df.rename(columns={'Outcome': 'Resultado'}, inplace=True)

print(f"Filas finales (después de borrar solo glucosa=0): {len(df)}")

# 5. Guardar el resultado
archivo_salida = 'diabetes_procesado.xlsx'
df.to_excel(archivo_salida, index=False)

print(f"Archivo guardado como '{archivo_salida}'")

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 1. Cargar el dataset que limpiamos en el paso anterior
archivo_procesado = 'diabetes_procesado.xlsx'
df = pd.read_excel(archivo_procesado)

# 2. Separar características (X) y la variable a predecir (y)
# X contendrá todas las columnas numéricas (Glucosa, BMI, etc.)
X = df.drop(columns=['Resultado'])
# y contendrá únicamente si es 'Positivo' o 'Negativo'
y = df['Resultado']

# 3. Dividir los datos: 80% entrenamiento (train) y 20% prueba (test)
# random_state=42 asegura que la división sea siempre igual si vuelves a correr el código
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Datos de entrenamiento: {len(X_train)} filas")
print(f"Datos de prueba: {len(X_test)} filas\n")

# 4. Inicializar el modelo de Random Forest
# n_estimators=80 significa que el modelo creará 80 "árboles" de decisión diferentes
rf_model = RandomForestClassifier(n_estimators=80, random_state=42, class_weight='balanced')

# 5. Entrenar el modelo con los datos de entrenamiento
rf_model.fit(X_train, y_train)

# 6. Hacer predicciones con los datos de prueba
y_pred = rf_model.predict(X_test)

# 7. Evaluar los resultados
print("--- EVALUACIÓN DEL MODELO ---")

# --- 8. VISUALIZACIONES ---

print("\nGenerando gráficos...")

# 8.1 Visualizar la Matriz de Confusión
# Creamos una figura
plt.figure(figsize=(8, 6))

# Usamos seaborn para crear un mapa de calor (heatmap) atractivo
matriz = confusion_matrix(y_test, y_pred)
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negativo (0)', 'Positivo (1)'],
            yticklabels=['Negativo (0)', 'Positivo (1)'])

plt.title('Matriz de Confusión - Random Forest')
plt.xlabel('Predicción del Modelo')
plt.ylabel('Valor Real (Paciente)')

# Mostrar el primer gráfico
plt.show()

# Exactitud general
exactitud = accuracy_score(y_test, y_pred)
print(f"Exactitud (Accuracy): {exactitud * 100:.2f}%\n")

# Reporte detallado de clasificación
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))



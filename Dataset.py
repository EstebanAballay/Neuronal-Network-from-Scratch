import matplotlib.pyplot as plt

dataset = { (1,5):1,
            (2,3):0,
            (6,7):0,
            (4,6):1,
            (2,5):1,
            (3,5):0,
            (2,10):1,
            (5,8):1,
            (7,15):1,
            (4,5):0,
            (7,9):0,
            (8,16):0
            }

# Extraer coordenadas y colores
x_values = []
y_values = []
colors = []

for (x, y), status in dataset.items():
    x_values.append(x)
    y_values.append(y)
    colors.append('green' if status == 1 else 'red')

# Graficar puntos
plt.figure(figsize=(6,6))
plt.scatter(x_values, y_values, c=colors, edgecolors='black', s=100)  # s controla el tamaño del punto

# Etiquetas y título
plt.xlabel("Lectures")
plt.ylabel("Study Hours")
plt.title("Pass/Fail Visualization")
plt.grid(True)

# Mostrar gráfico
plt.show()

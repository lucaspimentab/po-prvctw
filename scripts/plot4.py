import matplotlib.pyplot as plt
import numpy as np

# Estilo acadêmico minimalista
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'serif'

# Configuração da figura (3 subplots lado a lado)
fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)

np.random.seed(42) # Semente fixa para o mapa ficar sempre igual
num_clientes = 100

# ==========================================
# 1. Família C (Clustered / Aglomerada)
# ==========================================
# Cria 4 "bolhas" de clientes
centros = [(20, 20), (80, 80), (20, 80), (80, 20)]
x_c, y_c = [], []
for cx, cy in centros:
    x_c.extend(np.random.normal(cx, 5, 25))
    y_c.extend(np.random.normal(cy, 5, 25))

axes[0].scatter(x_c, y_c, c='#3498db', edgecolor='black', s=40, alpha=0.8)
axes[0].scatter(50, 50, c='red', marker='s', s=100, edgecolor='black', label='Depósito')
axes[0].set_title('Família C (Clustered)', fontweight='bold')

# ==========================================
# 2. Família R (Random / Aleatória)
# ==========================================
# Distribuição puramente uniforme
x_r = np.random.uniform(0, 100, num_clientes)
y_r = np.random.uniform(0, 100, num_clientes)

axes[1].scatter(x_r, y_r, c='#2ecc71', edgecolor='black', s=40, alpha=0.8)
axes[1].scatter(50, 50, c='red', marker='s', s=100, edgecolor='black')
axes[1].set_title('Família R (Random)', fontweight='bold')

# ==========================================
# 3. Família RC (Random-Clustered / Mista)
# ==========================================
# Metade do mapa em bolhas, metade aleatória
x_rc = list(np.random.normal(25, 6, 25)) + list(np.random.normal(75, 6, 25))
y_rc = list(np.random.normal(50, 6, 50))
x_rc.extend(np.random.uniform(0, 100, 50))
y_rc.extend(np.random.uniform(0, 100, 50))

axes[2].scatter(x_rc, y_rc, c='#9b59b6', edgecolor='black', s=40, alpha=0.8)
axes[2].scatter(50, 50, c='red', marker='s', s=100, edgecolor='black')
axes[2].set_title('Família RC (Mista)', fontweight='bold')

# Formatação limpa para todos os eixos
for ax in axes:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([]) # Remove números dos eixos para focar na forma
    ax.set_yticks([])
    ax.grid(True, linestyle=':', alpha=0.4)

fig.suptitle('Distribuição Geográfica Clássica de Solomon (1987)', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('grafico_mapas_solomon.png', bbox_inches='tight')
print("Imagem dos mapas gerada com sucesso: 'grafico_mapas_solomon.png'")
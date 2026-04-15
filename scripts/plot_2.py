import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuração de estilo para artigo acadêmico
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def generate_plots():
    # Lê os resultados
    df = pd.read_csv("results/resultados_rapidos.csv")
    
    # Cria uma coluna com o nome da família (ex: C1, R2, RC1) pegando as 2/3 primeiras letras
    df['Family'] = df['instance'].apply(lambda x: x[:-2])

    # ---------------------------------------------------------
    # Gráfico 1: Boxplot de Tempo Computacional por Família
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=300)
    sns.boxplot(data=df, x='Family', y='runtime_seconds', palette='Set2', width=0.6)
    sns.stripplot(data=df, x='Family', y='runtime_seconds', color=".3", size=5, alpha=0.6)
    
    plt.title("Tempo de Execução da Matheurística por Classe de Instância", fontweight='bold')
    plt.xlabel("Família de Instâncias (Solomon)", fontweight='bold')
    plt.ylabel("Tempo de Execução (segundos)", fontweight='bold')
    plt.axhline(y=df['runtime_seconds'].mean(), color='r', linestyle='--', alpha=0.7, label=f'Média Geral ({df["runtime_seconds"].mean():.1f}s)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/runtime_boxplot.png")
    plt.close()
    
    # ---------------------------------------------------------
    # Gráfico 2: Trade-off de Veículos (Seu modelo vs BKS)
    # ---------------------------------------------------------
    # Nota: Esse gráfico só ficará perfeito depois que você consertar o BKS!
    plt.figure(figsize=(8, 5), dpi=300)
    sns.histplot(data=df, x='vehicle_gap', discrete=True, color='#4a90e2', edgecolor='black')
    
    plt.title("Diferença na Frota Utilizada (Proposto vs BKS)", fontweight='bold')
    plt.xlabel("Veículos a mais/a menos que a literatura (Vehicle Gap)", fontweight='bold')
    plt.ylabel("Contagem de Instâncias", fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.savefig("results/vehicle_gap_histogram.png")
    plt.close()

    print("Gráficos gerados com sucesso na pasta 'results'!")

if __name__ == "__main__":
    generate_plots()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def extrair_familia(instance):
    match = re.match(r"([A-Z]+\d)", instance)
    return match.group(1) if match else instance

def analise_modelo_classico():
    df_lento = pd.read_csv('resultados_oficiais/resultados_lentos_pc_tunado.csv')
    df_burro = pd.read_csv('resultados_oficiais/resultados_burros_pc_tunado.csv')

    df_lento['Family'] = df_lento['instance'].apply(extrair_familia)
    df_burro['Family'] = df_burro['instance'].apply(extrair_familia)

    # Merge para parear os dados instância a instância
    df_comp = df_lento[['instance', 'Family', 'total_distance', 'runtime_seconds']].merge(
        df_burro[['instance', 'total_distance', 'runtime_seconds']], 
        on='instance', 
        suffixes=('_matheuristica', '_classico')
    )

    # =================================================================
    # GRÁFICO 1: Distribuição de Tempo de CPU (KDE / Histograma)
    # Mostra o gargalo (tailing-off) do modelo Clássico
    # =================================================================
    plt.figure(figsize=(9, 5), dpi=300)
    
    sns.kdeplot(df_comp['runtime_seconds_classico'], fill=True, color='#e74c3c', label='Modelo Clássico', alpha=0.5, linewidth=2)
    sns.kdeplot(df_comp['runtime_seconds_matheuristica'], fill=True, color='#2ecc71', label='Matheurística Proposta', alpha=0.5, linewidth=2)
    
    plt.axvline(x=300, color='black', linestyle='--', linewidth=1.5, label='Tempo Limite (300s)')
    
    plt.title('Estudo de Ablação: Gargalo Computacional (Tailing-off)', fontweight='bold')
    plt.xlabel('Tempo de Execução (segundos)', fontweight='bold')
    plt.ylabel('Densidade de Instâncias', fontweight='bold')
    
    # Anotação mostrando o gargalo
    plt.annotate('70% das instâncias\ntravam no tempo limite', xy=(295, 0.005), xytext=(150, 0.008),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5), fontsize=10, fontweight='bold', ha='center')

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('grafico_gargalo_computacional.png')
    plt.close()

    # =================================================================
    # GRÁFICO 2: Resgate de Distância nas Instâncias Críticas
    # Foca só nas 39 que bateram 300s no clássico
    # =================================================================
    df_critico = df_comp[df_comp['runtime_seconds_classico'] >= 300].copy()
    df_critico['melhoria_pct'] = ((df_critico['total_distance_classico'] - df_critico['total_distance_matheuristica']) / df_critico['total_distance_classico']) * 100
    
    # Agrupa por família
    ordem = ['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2']
    agrupado = df_critico.groupby('Family')['melhoria_pct'].mean().reindex(ordem).fillna(0).reset_index()

    plt.figure(figsize=(9, 5), dpi=300)
    ax = sns.barplot(x='Family', y='melhoria_pct', data=agrupado, palette='Reds_d', edgecolor='black')
    
    plt.title('Ganho de Qualidade nas Instâncias Críticas (Timeout do Clássico)', fontweight='bold')
    plt.xlabel('Famílias de Instâncias (Solomon)', fontweight='bold')
    plt.ylabel('Redução Média na Distância (%)', fontweight='bold')
    
    for p in ax.patches:
        val = p.get_height()
        if val > 0: 
            ax.annotate(f"-{val:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', fontweight='bold')

    plt.tight_layout()
    plt.savefig('grafico_resgate_critico.png')
    plt.close()

    print("Gráficos gerados com sucesso: 'grafico_gargalo_computacional.png' e 'grafico_resgate_critico.png'")

if __name__ == "__main__":
    analise_modelo_classico()
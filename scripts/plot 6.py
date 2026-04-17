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

def analise_honesta():
    df_lento = pd.read_csv('resultados_oficiais/resultados_lentos_pc_tunado.csv')
    df_burro = pd.read_csv('resultados_oficiais/resultados_burros_pc_tunado.csv')

    df_lento['Family'] = df_lento['instance'].apply(extrair_familia)
    df_burro['Family'] = df_burro['instance'].apply(extrair_familia)

    # Merge
    df_comp = df_lento[['instance', 'Family', 'total_distance', 'runtime_seconds']].merge(
        df_burro[['instance', 'total_distance', 'runtime_seconds']], 
        on='instance', 
        suffixes=('_matheuristica', '_classico')
    )

    # =================================================================
    # GRÁFICO 1: STRIP PLOT (A verdade nua e crua sobre o tempo)
    # =================================================================
    # Prepara os dados no formato longo
    df_long = pd.DataFrame({
        'Tempo de Execução (s)': np.concatenate([df_comp['runtime_seconds_classico'], df_comp['runtime_seconds_matheuristica']]),
        'Abordagem': ['Modelo Clássico (Burro)']*56 + ['Matheurística (Lenta)']*56
    })

    plt.figure(figsize=(10, 4.5), dpi=300)
    
    # Desenha os pontos exatos, com um pouco de "jitter" (espalhamento) no eixo Y para não encavalar
    sns.stripplot(x='Tempo de Execução (s)', y='Abordagem', data=df_long, 
                  palette=['#e74c3c', '#2ecc71'], jitter=0.25, size=7, alpha=0.7, edgecolor='black', linewidth=0.8)
    
    # Linhas de limite e fases
    plt.axvline(x=300, color='red', linestyle='--', linewidth=1.5, label='Timeout Global (300s)')
    plt.axvline(x=180, color='grey', linestyle=':', linewidth=2, label='Fim da Fase 1 (180s)')
    
    plt.title('Estudo de Ablação: Distribuição Real do Tempo de Execução', fontweight='bold')
    plt.xlabel('Tempo de Execução Computacional (segundos)', fontweight='bold')
    plt.ylabel('')
    
    # Anotações para explicar o fenômeno que você notou
    plt.annotate('70% trava no\ntempo limite', xy=(295, 0.1), xytext=(240, -0.2),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5), fontsize=10, fontweight='bold', ha='center')
    
    plt.annotate('Fase 2 é tão rápida que\nraramente usa seus 120s', xy=(185, 1), xytext=(250, 1.3),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5), fontsize=10, fontweight='bold', ha='center', color='#27ae60')

    plt.legend(loc='center left')
    plt.tight_layout()
    plt.savefig('grafico_gargalo_stripplot.png')
    plt.close()

    # =================================================================
    # GRÁFICO 2: RESGATE DE DISTÂNCIA (Barplot das 39 instâncias falhas)
    # =================================================================
    df_critico = df_comp[df_comp['runtime_seconds_classico'] >= 300].copy()
    df_critico['melhoria_pct'] = ((df_critico['total_distance_classico'] - df_critico['total_distance_matheuristica']) / df_critico['total_distance_classico']) * 100
    
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
    
    print("Gráficos limpos e honestos gerados com sucesso!")

if __name__ == "__main__":
    analise_honesta()
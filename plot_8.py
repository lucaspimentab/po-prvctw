import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'

def extrair_familia(instance):
    match = re.match(r"([A-Z]+\d)", instance)
    return match.group(1) if match else instance

def extrair_tipo(family):
    if family in ['C1', 'R1', 'RC1']:
        return 'Tipo 1 (Janelas Restritas)'
    else:
        return 'Tipo 2 (Janelas Frouxas)'

def gerar_graficos_novos():
    df_rapido = pd.read_csv('resultados_oficiais/resultados_rapidos_pc_tunado.csv')
    df_lento = pd.read_csv('resultados_oficiais/resultados_lentos_pc_tunado.csv')
    df_burro = pd.read_csv('resultados_oficiais/resultados_burros_pc_tunado.csv')

    df_rapido['Family'] = df_rapido['instance'].apply(extrair_familia)
    df_lento['Family'] = df_lento['instance'].apply(extrair_familia)
    df_burro['Family'] = df_burro['instance'].apply(extrair_familia)

    # =================================================================
    # GRÁFICO 1: TIPO 1 VS TIPO 2 (Evolução Interna)
    # =================================================================
    df_comp_interna = df_rapido[['instance', 'Family', 'total_distance']].merge(
        df_lento[['instance', 'total_distance']], on='instance', suffixes=('_rapido', '_lento')
    )
    df_comp_interna['Type'] = df_comp_interna['Family'].apply(extrair_tipo)
    df_comp_interna['melhoria_pct'] = ((df_comp_interna['total_distance_rapido'] - df_comp_interna['total_distance_lento']) / df_comp_interna['total_distance_rapido']) * 100

    plt.figure(figsize=(7, 5), dpi=300)
    
    # Boxplot para mostrar a distribuição e a mediana
    sns.boxplot(x='Type', y='melhoria_pct', data=df_comp_interna, palette='Set2', width=0.4, boxprops=dict(alpha=0.7))
    sns.stripplot(x='Type', y='melhoria_pct', data=df_comp_interna, color='black', alpha=0.5, jitter=True)
    
    plt.title("Sensibilidade do Fix-and-Optimize por Topologia", fontweight='bold')
    plt.xlabel("")
    plt.ylabel("Melhoria na Distância: Dinâmico vs Tático (%)", fontweight='bold')
    
    # Anotações das médias
    media_t1 = df_comp_interna[df_comp_interna['Type'] == 'Tipo 1 (Janelas Restritas)']['melhoria_pct'].mean()
    media_t2 = df_comp_interna[df_comp_interna['Type'] == 'Tipo 2 (Janelas Frouxas)']['melhoria_pct'].mean()
    plt.text(0, media_t1 + 0.3, f"Média: {media_t1:.2f}%", ha='center', fontweight='bold', color='darkblue')
    plt.text(1, media_t2 + 0.3, f"Média: {media_t2:.2f}%", ha='center', fontweight='bold', color='darkgreen')

    plt.tight_layout()
    plt.savefig('grafico_tipo1_vs_tipo2.png')
    plt.close()

    # =================================================================
    # GRÁFICO 2: DOMINÂNCIA MATHEURÍSTICA VS CLÁSSICO
    # =================================================================
    df_comp_ablasao = df_lento[['instance', 'Family', 'total_distance']].merge(
        df_burro[['instance', 'total_distance']], on='instance', suffixes=('_matheuristica', '_classico')
    )

    plt.figure(figsize=(7, 6), dpi=300)
    
    # Plota a nuvem de pontos (Distância do Clássico vs Nossa Distância)
    sns.scatterplot(x='total_distance_classico', y='total_distance_matheuristica', 
                    hue='Family', data=df_comp_ablasao, palette='tab10', s=80, edgecolor='black', alpha=0.8)
    
    # Reta de Empate (y = x)
    max_val = max(df_comp_ablasao['total_distance_classico'].max(), df_comp_ablasao['total_distance_matheuristica'].max())
    min_val = min(df_comp_ablasao['total_distance_classico'].min(), df_comp_ablasao['total_distance_matheuristica'].min())
    plt.plot([min_val-50, max_val+50], [min_val-50, max_val+50], 'r--', lw=2, label='Linha de Empate (y = x)')
    
    # Destaca a área de dominância
    plt.fill_between([min_val-50, max_val+50], 0, [min_val-50, max_val+50], color='green', alpha=0.05, label='Matheurística é Superior')

    plt.title("Dominância de Qualidade: Matheurística vs Clássico", fontweight='bold')
    plt.xlabel("Distância Obtida pelo Modelo Clássico (300s)", fontweight='bold')
    plt.ylabel("Distância Obtida pela Matheurística (300s)", fontweight='bold')
    
    # Define limites e legenda
    plt.xlim(min_val-50, max_val+50)
    plt.ylim(min_val-50, max_val+50)
    plt.legend(loc='lower right', frameon=True, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('grafico_dominancia_classico.png')
    plt.close()

    print("Novos gráficos gerados com sucesso!")

if __name__ == "__main__":
    gerar_graficos_novos()
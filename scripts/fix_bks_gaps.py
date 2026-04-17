import pandas as pd
from pathlib import Path

# Dados oficiais do SINTEF (Solomon 100) limpos [Veículos, Distância]
SINTEF_BKS = {
    "C101": [10, 828.94], "C102": [10, 828.94], "C103": [10, 828.06], "C104": [10, 824.78],
    "C105": [10, 828.94], "C106": [10, 828.94], "C107": [10, 828.94], "C108": [10, 828.94], "C109": [10, 828.94],
    "C201": [3, 591.56], "C202": [3, 591.56], "C203": [3, 591.17], "C204": [3, 590.60],
    "C205": [3, 588.88], "C206": [3, 588.49], "C207": [3, 588.29], "C208": [3, 588.32],
    "R101": [19, 1650.80], "R102": [17, 1486.12], "R103": [13, 1292.68], "R104": [9, 1007.31],
    "R105": [14, 1377.11], "R106": [12, 1252.03], "R107": [10, 1104.66], "R108": [9, 960.88],
    "R109": [11, 1194.73], "R110": [10, 1118.84], "R111": [10, 1096.72], "R112": [9, 982.14],
    "R201": [4, 1252.37], "R202": [3, 1191.70], "R203": [3, 939.50], "R204": [2, 825.52],
    "R205": [3, 994.43], "R206": [3, 906.14], "R207": [2, 890.61], "R208": [2, 726.82],
    "R209": [3, 909.16], "R210": [3, 939.37], "R211": [2, 885.71],
    "RC101": [14, 1696.95], "RC102": [12, 1554.75], "RC103": [11, 1261.67], "RC104": [10, 1135.48],
    "RC105": [13, 1629.44], "RC106": [11, 1424.73], "RC107": [11, 1230.48], "RC108": [10, 1139.82],
    "RC201": [4, 1406.94], "RC202": [3, 1365.65], "RC203": [3, 1049.62], "RC204": [3, 798.46],
    "RC205": [4, 1297.65], "RC206": [3, 1146.32], "RC207": [3, 1061.14], "RC208": [3, 828.14]
}

def fix_results_csv():
    file_path = Path("results/resultados_rapidos_pc_tunado.csv")
    
    if not file_path.exists():
        print("Arquivo CSV não encontrado na pasta results.")
        return

    # Carrega o CSV que você acabou de gerar
    df = pd.read_csv(file_path)

    # Limpa nomes de instâncias para garantir que deem match no dicionário (ex: "Solomon_C101" -> "C101")
    df['clean_name'] = df['instance'].apply(lambda x: x.split('_')[-1].upper().replace('.TXT', ''))

    # Atualiza as colunas
    for idx, row in df.iterrows():
        inst = row['clean_name']
        if inst in SINTEF_BKS:
            bks_veh, bks_dist = SINTEF_BKS[inst]
            
            # Atualiza BKS
            df.at[idx, 'best_known_vehicles'] = bks_veh
            df.at[idx, 'best_known_distance'] = bks_dist
            
            # Recalcula Gaps
            df.at[idx, 'vehicle_gap'] = row['vehicles_used'] - bks_veh
            df.at[idx, 'distance_gap_pct'] = round(((row['total_distance'] - bks_dist) / bks_dist) * 100.0, 2)

    # Remove a coluna temporária e salva
    df.drop(columns=['clean_name'], inplace=True)
    df.to_csv(file_path, index=False)
    print("CSV corrigido e atualizado com sucesso! Gaps recalculados.")

if __name__ == "__main__":
    fix_results_csv()
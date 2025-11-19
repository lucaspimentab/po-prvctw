## Trabalho de Pesquisa Operacional - PRVC-TW

Repositório contendo o pipeline baseado em Gurobi para resolver o Problema de Roteamento de Veículos com Capacidade e Janelas de Tempo (PRVC-TW) nas instâncias clássicas de Solomon (C101, R101 e RC101).

### Estrutura
- `instances/` - arquivos `.txt` originais das instâncias de Solomon (100 clientes + depósito).
- `data/solomon_bks.csv` - tabela com os melhores valores conhecidos (BKS) usados para comparar distância/frota.
- `scripts/solve_vrptw_gurobi.py` - modelo MILP completo em Python + gurobipy
- `results/` - pasta onde o script salva `solomon_vrptw_gurobi.json` e `.csv`.
- `docs/` - relatório

### Como executar
1. Crie o ambiente e instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Certifique-se de ter uma licença válida do Gurobi configurada (`gurobi.lic`).
3. Execute o solver:
   ```bash
   python scripts/solve_vrptw_gurobi.py
   ```
4. Ao final consulte `results/solomon_vrptw_gurobi.csv` para comparar os resultados obtidos com o BKS citado na planilha.

### Dados e referências
- Instâncias: Solomon (1987), com coordenadas euclidianas, demandas, tempos de serviço e janelas `[a_i, b_i]` para cada cliente.
- BKS: Minocha & Tripathi (2013) – referência usada para montar `data/solomon_bks.csv`.
- Solver: Gurobi Optimizer (branch-and-cut para MILP), integração via `gurobipy`.
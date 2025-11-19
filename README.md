## Trabalho de Pesquisa Operacional - PRVC-TW (Gurobi)

Repositório com o pipeline baseado em Gurobi para resolver o Problema de Roteamento de Veículos com Capacidade e Janelas de Tempo (PRVC-TW) nas instâncias clássicas de Solomon (C101, R101 e RC101).

### Estrutura
- `instances/` – arquivos `.txt` originais de Solomon com 100 clientes + depósito.
- `data/solomon_bks.csv` – tabela com os melhores valores conhecidos (BKS) usados para comparação.
- `scripts/solve_vrptw_gurobi.py` – modelo MILP completo em Python (parsing, modelagem e exportação de resultados).
- `results/` – contém `solomon_vrptw_gurobi.json` e `.csv` após cada execução.
- `docs/` – materiais de apoio (relatório em LaTeX etc.).

### Como executar
1. Instale as dependências: `pip install -r requirements.txt`.
2. Verifique se a licença do Gurobi (`gurobi.lic`) está configurada.
3. Rode o solver: `python scripts/solve_vrptw_gurobi.py`.
4. Consulte `results/solomon_vrptw_gurobi.csv` para comparar com os valores BKS.

### Dados e referências
- Instâncias: Solomon (1987) – coordenadas euclidianas, demandas, tempos de serviço e janelas `[a_i, b_i]` por cliente.
- BKS: Minocha & Tripathi (2013) – mesma referência usada no relatório.
- Solver: Gurobi Optimizer (branch-and-cut para MILP) via `gurobipy`.

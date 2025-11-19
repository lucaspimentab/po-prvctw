ï»¿## Trabalho de Pesquisa Operacional - PRVC-TW

RepositÃ³rio contendo o pipeline baseado em Gurobi para resolver o Problema de Roteamento de VeÃ­culos com Capacidade e Janelas de Tempo (PRVC-TW) nas instÃ¢ncias clÃ¡ssicas de Solomon (C101, R101 e RC101).

### Estrutura
- `instances/` - arquivos `.txt` originais das instÃ¢ncias de Solomon (100 clientes + depÃ³sito).
- `data/solomon_bks.csv` - tabela com os melhores valores conhecidos (BKS) usados para comparar distÃ¢ncia/frota.
- `scripts/solve_vrptw_gurobi.py` - modelo MILP completo em Python + gurobipy
- `results/` - pasta onde o script salva `solomon_vrptw_gurobi.json` e `.csv`.
- `docs/` - relatÃ³rio

### Como executar
1. Crie o ambiente e instale as dependÃªncias:
   ```bash
   pip install -r requirements.txt
   ```
2. Certifique-se de ter uma licenÃ§a vÃ¡lida do Gurobi configurada (`gurobi.lic`).
3. Execute o solver:
   ```bash
   python scripts/solve_vrptw_gurobi.py
   ```
4. Ao final consulte `results/solomon_vrptw_gurobi.csv` para comparar os resultados obtidos com o BKS citado na planilha.

### Dados e referÃªncias
- InstÃ¢ncias: Solomon (1987), com coordenadas euclidianas, demandas, tempos de serviÃ§o e janelas `[a_i, b_i]` para cada cliente.
- BKS: Minocha & Tripathi (2013) â referÃªncia usada para montar `data/solomon_bks.csv`.
- Solver: Gurobi Optimizer (branch-and-cut para MILP), integraÃ§Ã£o via `gurobipy`.
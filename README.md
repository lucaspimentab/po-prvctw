## VRPTW MILP Pipeline (Gurobi)

Este repositório reúne os scripts, dados e resultados utilizados no estudo do Problema de Roteamento de Veículos com Janelas de Tempo (VRPTW) nas instâncias clássicas de Solomon (C101, R101 e RC101). Todo o pipeline foi implementado em Python com o solver Gurobi, produzindo tabelas, gráficos e um relatório técnico em LaTeX.

### Estrutura
- `instances/` – arquivos `.txt` originais de Solomon (100 clientes + depósito).
- `data/solomon_bks.csv` – valores Best Known Solutions (BKS) usados para comparação.
- `scripts/solve_vrptw_gurobi.py` – parsing das instâncias, construção do MILP e exportação do resultado em JSON/CSV.
- `scripts/plot_vrptw_routes.py` – gera mapas de distribuição e rotas a partir do JSON consolidado.
- `results/` – `solomon_vrptw_gurobi.json` (rotas detalhadas) e `solomon_vrptw_gurobi.csv` (resumo por instância).
- `figures/`
- `docs/` – relatório e slides.

### Como executar
1. Instale as dependências: `pip install -r requirements.txt`.
2. Certifique-se de que a licença do Gurobi (`gurobi.lic`) está configurada.
3. Resolva as instâncias:
   ```bash
   python scripts/solve_vrptw_gurobi.py
   ```
   O script grava JSON/CSV em `results/` com distância total, veículos usados, tempo de execução e `Gap_MIP`.
4. Gere os gráficos:
   ```bash
   python scripts/plot_vrptw_routes.py
   ```
   As imagens são salvas em `figures/` (`*_grafo.png` e `*_rotas_publicacao.png`).

### Dados e referências
- **Instâncias:** Solomon (1987), com coordenadas euclidianas, demandas e janelas individuais por cliente.
- **BKS:** Minocha & Tripathi (2013), mesma base utilizada no relatório.
- **Solver:** Gurobi Optimizer (`gurobipy`), branch-and-cut com limite padrão de 900 s por instância.

Resultados completos, análise de convergência e discussão podem ser conferidos em `docs/Relatório - PRVC-TW.pdf`.

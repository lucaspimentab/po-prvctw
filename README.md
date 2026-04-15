## VRPTW MILP Pipeline (Gurobi)

Este repositório reúne os scripts, dados e resultados utilizados no estudo do Problema de Roteamento de Veículos com Janelas de Tempo (VRPTW) nas instâncias de Solomon disponíveis na pasta `In/` (56 arquivos `.txt`). Todo o pipeline foi implementado em Python com o solver Gurobi, produzindo tabelas, gráficos e um relatório técnico em LaTeX.

### Estrutura
- `In/` – conjunto de instâncias Solomon usado pelo pipeline (56 arquivos `.txt`).
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
   O script varre dinamicamente todos os `.txt` de `In/` e grava JSON/CSV em `results/` com distância total, veículos usados e tempo de execução.
4. Gere os gráficos:
   ```bash
   python scripts/plot_vrptw_routes.py
   ```
   As imagens são salvas em `figures/` (`*_grafo.png` e `*_rotas_publicacao.png`).

### Dados e referências
- **Instâncias:** Solomon (1987), com coordenadas euclidianas, demandas e janelas individuais por cliente.
- **BKS:** `data/solomon_bks.csv` (atualmente preenchida para C101, R101 e RC101).
- **Solver:** Gurobi Optimizer (`gurobipy`), branch-and-cut com limite padrão de 900 s por instância.

Resultados completos, análise de convergência e discussão podem ser conferidos em `docs/Relatório - PRVC-TW.pdf`.

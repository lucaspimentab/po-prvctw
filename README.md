## Trabalho de Pesquisa Operacional – PRVC-TW

Repositório com o script usado para resolver o Problema de Roteamento de Veículos com Capacidade e Janelas de Tempo (PRVC-TW) nas instâncias clássicas de Solomon (C101, R101 e RC101).

### Estrutura
- `instances/` – arquivos `.txt` originais baixados do repositório público do Solomon.
- `data/solomon_bks.csv` – tabela com o número de veículos e distâncias ótimas publicadas (Minocha & Tripathi, 2013).
- `scripts/solve_vrptw.py` – script principal, escrito em Python usando o solver de roteamento da Google OR-Tools.
- `results/` – pasta onde o script grava `solomon_vrptw_results.json` e `.csv`.
- `docs/trabalho_po_prvctw.tex` – artigo completo (modelo em LaTeX pronto para Overleaf), seguindo a estrutura exigida pelo professor.

### Executando
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Rode o solver:
   ```bash
   python scripts/solve_vrptw.py
   ```
3. Consulte `results/solomon_vrptw_results.csv` para comparar veículos e distâncias com os valores de referência.
4. Abra `docs/trabalho_po_prvctw.tex` no Overleaf para gerar o relatório em PDF.

### Observações
- O script já vem comentado indicando como a função objetivo, a dimensão de tempo e a restrição de capacidade são modeladas no OR-Tools.
- Caso precise alterar as instâncias, basta editar a lista `INSTANCE_SPECS` no início do arquivo `solve_vrptw.py`.

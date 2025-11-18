## Trabalho de Pesquisa Operacional - PRVC-TW

Repositorio com os scripts usados para resolver o Problema de Roteamento de Veiculos com Capacidade e Janelas de Tempo (PRVC-TW) nas instancias classicas de Solomon (C101, R101 e RC101).

### Estrutura
- `instances/` - arquivos `.txt` originais do benchmark de Solomon (100 clientes + deposito).
- `data/solomon_bks.csv` - tabela com numero de veiculos e distancias otimas publicadas (Minocha & Tripathi, 2013).
- `scripts/solve_vrptw.py` - solver heuristico usando Google OR-Tools (dimensoes de tempo e capacidade).
- `scripts/solve_vrptw_gurobi.py` - modelo MILP completo montado no Gurobi para documentar funcao objetivo e restricoes.
- `results/` - sao gerados `solomon_vrptw_results.*` (OR-Tools) e `solomon_vrptw_gurobi.*` (Gurobi) em CSV/JSON.
- `docs/trabalho_po_prvctw.tex` - artigo-base para Overleaf seguindo o roteiro pedido em aula.

### Execucao rapida
1. Instale as dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Rode o solver heuristico (OR-Tools):
   ```bash
   python scripts/solve_vrptw.py
   ```
3. Rode o modelo MILP no Gurobi (necessita licenca ativa):
   ```bash
   python scripts/solve_vrptw_gurobi.py
   ```
   Os logs do Gurobi mostram a convergencia do MIP; ao final os resultados sao gravados na pasta `results/`.

### Dados e referencias
- As instancias foram obtidas do repositorio publico de Solomon (arquivo texto tradicional), logo cada cliente possui coordenadas euclidianas, demanda, janela [inicio, fim] e tempo de servico fixo.
- A planilha `data/solomon_bks.csv` traz os melhores valores conhecidos (BKS) usados para comparar frota e distancia.
- Nao ha dados sinteticos novos aqui; tudo foi retirado diretamente da literatura classica do PRVC-TW.

### Observacoes
- O script do Gurobi usa um limite de 15 minutos por instancia (`DEFAULT_TIME_LIMIT`). Ajuste se tiver uma licenca mais forte.
- Se precisar trocar/filtrar instancias, edite a lista `INSTANCE_SPECS` em `scripts/solve_vrptw.py` (o script do Gurobi importa a mesma definicao).
- Ambos os scripts produzem CSV + JSON para facilitar a geracao de tabelas no relatorio e nos slides.

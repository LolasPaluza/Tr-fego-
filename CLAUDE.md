# CLAUDE.md — convenções e decisões do TrafficRL

## Comandos frequentes

```bash
source .venv/bin/activate
python scripts/validate_setup.py           # sanidade do ambiente
pytest -q                                  # suíte completa (~15 s)
pytest -q --cov=src/traffic_rl             # com cobertura (meta > 70%)
ruff check src/ tests/                     # lint
traffic-rl compare --smoke                 # pipeline completo em minutos
```

## Convenções

- Idioma: código/identificadores em inglês onde é jargão técnico, nomes de
  domínio do experimento em português (`fixo_igual`, `r_espera`,
  `pico_assimetrico`) — são os nomes usados no relatório e nos gráficos.
- Docstrings e comentários em português; type hints em todo código público.
- Ordem canônica das aproximações: `(E, W, N, S)` — avenida primeiro.
  Definida em `config.APPROACHES`; NUNCA reordene localmente.
- Estágios: 0=avenida frente/dir, 1=avenida esq, 2=local frente/dir,
  3=local esq (`config.STAGE_NAMES`).
- Config só via pydantic (`config.py`); nada de dicts soltos ou caminhos
  hardcoded (use `paths.py`).
- Commits pequenos e descritivos (o histórico é a revisão do autor).

## Arquitetura e porquês (resumo; ADRs completos em docs/DECISOES.md)

- **Env TraCI próprio, não sumo-rl**: o modelo de fases do sumo-rl não
  representa vermelho total (all-red) nem verde máximo forçado; o wrapper
  próprio impõe TODA a segurança (verde mín 10 s, amarelo 3 s, all-red 2 s)
  e replica a recompensa `diff-waiting-time` do sumo-rl como `r_espera`.
- **`Observation` estruturada + vetorizadores**: o env sempre computa a
  estrutura completa; baselines leem campos nomeados, o DQN vê a projeção
  (`obs_minimal`/`obs_rica`). O contrato `Controller.act(obs)` fica único.
- **libsumo = 1 simulação por processo**: treino e avaliação periódica rodam
  em subprocessos (`SubprocVecEnv(start_method="fork")`). Nunca crie dois
  envs ativos no mesmo processo; o runner de avaliação fecha um episódio
  antes de abrir o outro.
- **Seeds**: treino usa `seed × 1e6 + episódio`; avaliação periódica
  `9e8 + seed`; avaliação final `10000 + episódio`. Disjuntas por construção.
- **tripinfo é a fonte de métricas** (inclui veículos não concluídos para o
  starvation check); só é escrito completo após fechar o SUMO.
- **Double DQN** é subclasse mínima do DQN do sb3 (só o alvo muda).

## Armadilhas conhecidas

- `--waiting-time-memory 100000` é obrigatório no sumo: o default (100 s)
  zeraria a espera acumulada e quebraria `r_espera` e o starvation check.
- Teleport desligado (`--time-to-teleport -1`): política ruim gera fila real,
  não veículos teleportados que mascaram o problema.
- `probability` dos flows = veíc/h ÷ 3600; a variação entre episódios vem de
  `sumo --seed`, o .rou.xml é determinístico.
- Rodar pytest de dentro do venv do projeto; os testes geram redes em
  data/generated/ (git-ignored).

# TrafficRL — Controle Semafórico Adaptativo via Deep Reinforcement Learning

**Fase 1: um cruzamento, rigor experimental completo.**

## Motivação

Em São Paulo, a maioria dos semáforos opera com tempos fixos: o verde da
avenida dura o mesmo às 8h de segunda e às 3h de domingo, e uma via local
vazia recebe o mesmo verde de uma avenida congestionada. Isso desperdiça
capacidade viária — a cidade tem asfalto parado esperando um plano semafórico
melhor. Este projeto investiga se um agente de *reinforcement learning* pode
aprender políticas de controle que se adaptam à demanda em tempo real,
começando pelo caso mínimo que já contém o problema essencial: um cruzamento
entre uma **avenida** (3 faixas/sentido, 60 km/h) e uma **via local**
(1 faixa/sentido, 40 km/h), com demanda assimétrica.

A Fase 1 não busca o melhor agente possível — busca a **fundação experimental
certa**: baselines fortes, múltiplas seeds, testes estatísticos, ablações e
uma métrica de justiça que impede a resposta trivial ("dê verde infinito à
avenida").

## Fundamentos

**Por que RL para semáforos?** O controle semafórico é um problema sequencial
de decisão sob incerteza: o estado (filas, chegadas) evolui estocasticamente
e cada decisão afeta o futuro. RL aprende a política diretamente da interação
com um simulador, sem exigir um modelo analítico do tráfego.

**O que é max-pressure?** O controlador max-pressure (Varaiya, 2013) ativa,
a cada oportunidade, o estágio com maior "pressão" — a soma, sobre os
movimentos servidos, de (veículos a montante − veículos a jusante). Varaiya
provou que essa política simples estabiliza qualquer demanda estabilizável,
sem conhecer as taxas de chegada. É a teoria por trás dos melhores
controladores clássicos e, junto com o controle atuado, o baseline que o
agente **precisa** bater para ter mérito real.

> P. Varaiya, "Max pressure control of a network of signalized
> intersections", *Transportation Research Part C*, 36:177–195, 2013.

**Por que múltiplas seeds?** O treino de DQN é estocástico (inicialização,
replay, exploração): uma seed sortuda pode parecer um avanço. Reportamos
5 seeds independentes (42, 123, 7, 2024, 777) com média e intervalo de
confiança — o padrão mínimo para afirmar que o método (e não o acaso)
produziu o resultado.

## Estrutura

```
configs/            # YAML validado por pydantic (default.yaml + smoke.yaml)
src/traffic_rl/
  envs/             # rede (netconvert), demanda (.rou.xml), fases, ambiente Gymnasium
  controllers/      # interface Controller (ABC) + 4 baselines + adaptador DQN
  training/         # DoubleDQN (sb3), protocolo multi-seed retomável
  evaluation/       # protocolo método × cenário × episódios; ablações
  analysis/         # bootstrap, Mann-Whitney+Bonferroni, gráficos, REPORT.md
scripts/            # validate_setup.py + run_full_*.sh (máquina local)
tests/              # pytest (cobertura > 70% de src/)
docs/               # DECISOES.md (ADRs), PENDENTE_LOCAL.md
results/            # runs/, REPORT.md, figuras (gerados; fora do git)
```

A interface `Controller.act(Observation) -> estágio` é o contrato central:
baselines e agente RL são intercambiáveis no protocolo de avaliação, e na
Fase 2 cada cruzamento de um grid receberá sua instância.

## Instalação

Requisitos: Python ≥ 3.11. O SUMO vem por pip (`eclipse-sumo`), sem depender
de apt/brew — o pacote inclui os binários (`sumo`, `netconvert`) e o projeto
resolve `SUMO_HOME` automaticamente.

```bash
git clone <este-repo> && cd <este-repo>
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# (opcional, recomendado) torch só-CPU, ~10x menor que o pacote com CUDA:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
# e depois: pip install -e ".[dev]"

python scripts/validate_setup.py   # deve terminar com "Ambiente pronto"
```

Se você já tem SUMO instalado pelo sistema (apt: `sudo apt install sumo
sumo-tools`; macOS: `brew install sumo`), defina `SUMO_HOME` (ex.:
`export SUMO_HOME=/usr/share/sumo`) e ele terá precedência — exige-se
SUMO ≥ 1.19.

## Uso

```bash
traffic-rl train                      # 5 seeds × 300k passos (longo!)
traffic-rl train --smoke              # validação: 1 seed × 5k passos
traffic-rl evaluate --methods fixo_igual,max_pressure
traffic-rl compare                    # protocolo completo + results/REPORT.md
traffic-rl compare --smoke            # pipeline inteiro em minutos
traffic-rl ablation --type obs        # ablação de observação
traffic-rl ablation --type reward     # ablação de recompensa
```

`--smoke` aplica `configs/smoke.yaml` por cima da config default: episódios
de 600 s, 1 seed, 5k passos, 2 episódios de avaliação. Serve para validar o
pipeline de ponta a ponta, não para obter uma política boa.

## O ambiente

**Ação** (a cada `delta_time` = 5 s): escolher o próximo estágio verde entre
4 — (0) avenida frente/direita, (1) avenida esquerda protegida (faixa
dedicada), (2) local frente/direita, (3) local esquerda. Repetir o estágio
atual = estender o verde.

**Segurança imposta pelo ambiente, nunca pelo agente:** verde mínimo de 10 s
(pedidos precoces de troca são ignorados), verde máximo de 120 s (troca
forçada), e toda transição passa por 3 s de amarelo + 2 s de vermelho total.
Nenhuma política — aprendida ou não — consegue violar esses tempos.

**Observações** — duas representações (ablação obrigatória):

| Feature | obs_minimal | obs_rica | Normalização (máximo documentado) |
|---|---|---|---|
| Estágio atual (one-hot, 4) | ✓ | ✓ | — |
| Tempo no estágio | ✓ | ✓ | / verde máximo (120 s) |
| Fila por aproximação (4) | ✓ | ✓ | / capacidade = faixas × (300 m / 7,5 m) |
| Densidade por faixa (8) | | ✓ | / capacidade da faixa (300 m / 7,5 m) |
| Velocidade média por aproximação (4) | | ✓ | / limite da via (60 ou 40 km/h) |
| Espera acumulada do 1º veículo da fila (4) | | ✓ | / 300 s (clip em 1,0) |

Total: 9 (minimal) vs 25 (rica) dimensões. **Por que normalizar?** O DQN usa
um MLP com uma taxa de aprendizado única para todos os pesos. Se uma entrada
varia em [0, 40] (fila em veículos) e outra em [0, 1] (one-hot), os
gradientes da primeira dominam e o treino fica instável ou lento; com tudo
em [0, 1], nenhuma feature captura o passo de gradiente sozinha, e os mesmos
hiperparâmetros servem às duas representações da ablação.

**Hipótese registrada (antes de rodar):** obs_rica deve **vencer** no
`pico_invertido` — o agente precisa perceber que a demanda mudou de via, e
densidade/velocidade/espera dão essa informação — e **empatar** no
`fora_pico`, onde há folga de capacidade e qualquer política razoável basta.

**Recompensas** — três funções (ablação obrigatória), todas ÷100 para ficarem
na mesma ordem de grandeza:

- `r_espera`: −Δ(espera total acumulada) — padrão do sumo-rl; positiva quando
  a espera diminui;
- `r_fila`: −Σ(filas) no passo de decisão;
- `r_pressao`: −Σ|pressão por movimento| — a grandeza que o max-pressure
  minimiza, como sinal denso (estilo PressLight).

**Anti-starvation (configurável):** penalidade por aproximação com veículo
esperando > 120 s. Sem ela, otimizar a espera **média** permite uma política
degenerada: sacrificar a via local para sempre (pouca demanda ⇒ pouco peso
na média). É a versão RL do clássico "otimizar a média e ignorar a cauda" —
e a razão de reportarmos p95 e espera máxima, não só a média.

**Cenários** (veíc/h por sentido; chegadas estocásticas binomiais do SUMO;
10% caminhões nos picos):

| Cenário | Avenida | Local | O que testa |
|---|---|---|---|
| pico_assimetrico | 1200 | 300 | o problema-alvo (treino) |
| fora_pico | 400 | 150 | folga de capacidade |
| balanceado | 500 | 500 | controle: proporcional não deve ter vantagem |
| pico_invertido | 500 | 700 | generalização: o agente decorou "priorize a avenida"? |

## Baselines — a credibilidade do projeto

1. `fixo_igual` — verdes iguais, ciclo de 120 s: o "semáforo burro".
2. `fixo_proporcional` — verde proporcional à demanda média do cenário.
   Este baseline é **deliberadamente injusto a favor dele**: recebe de graça
   o conhecimento perfeito da demanda média, que na prática exige medição em
   campo e recalibração periódica. O agente precisa inferir a demanda das
   observações. Se mesmo assim o RL vencer, o mérito é real.
3. `atuado_gap` — estende o verde enquanto detecta veículos (até 60 s), passa
   ao próximo estágio com demanda quando abre gap: é o que controladores
   comerciais "inteligentes" fazem.
4. `max_pressure` — o controlador de Varaiya, sem aprendizado.

**O agente só tem mérito real se bater 3 e 4.** Bater apenas o `fixo_igual`
é o mínimo aceitável (critério de aceitação), não um resultado publicável —
a análise do REPORT.md deixa isso explícito, vença ou perca.

## Protocolo experimental

- Treino: 5 seeds × 300k passos no `pico_assimetrico`; avaliação periódica
  a cada 10k passos (3 episódios greedy) — a curva salva é a de avaliação,
  não a de treino; melhor checkpoint por seed + config + git hash gravados.
- Avaliação final: cada método × 4 cenários × 20 episódios com seeds de
  tráfego (10000+i) disjuntas das de treino.
- Métricas por episódio: espera média, **p95 da espera** (justiça), espera
  máxima de um único veículo (starvation check), fila máxima, throughput,
  tempo médio de viagem, espera por classe viária (avenida × local).
- Estatística: média ± IC 95% (bootstrap, 10k reamostragens); Mann-Whitney U
  entre DQN e cada baseline por cenário, com correção de Bonferroni;
  p-values na tabela do REPORT.md.

## Rodando na máquina local

Pré-requisitos: Python ≥ 3.11, ~5 GB de disco, CPU (referência: 8 núcleos,
sem GPU — o DQN é pequeno e roda bem em CPU).

```bash
# 1. instalar e validar (~5 min)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/validate_setup.py

# 2. smoke test do pipeline completo (~10 min) — precisa terminar sem erro
traffic-rl train --smoke && traffic-rl compare --smoke

# 3. treino completo (retomável; ~1h/seed; ~1,5–2h em paralelo com 8 núcleos,
#    ~5h sequencial). PARALLEL=0 força sequencial.
bash scripts/run_full_training.sh

# 4. avaliação completa + REPORT.md final (~2h)
bash scripts/run_full_evaluation.sh

# 5. ablações completas (5 variantes × 5 seeds × 300k ≈ 24–30h sequencial —
#    rode durante a noite/fim de semana; retomável se interrompido)
bash scripts/run_ablations.sh
```

**Monitorar progresso:**

```bash
tail -f results/logs/train_seed_42.log     # log por seed
tensorboard --logdir results/runs          # curvas de treino/avaliação
ls results/runs/*/seed_*/checkpoints/      # passos já salvos (retomada)
```

**Estimativa honesta de computação total** (8 núcleos, sem GPU): treino
~2 h (paralelo) + avaliação ~2 h + ablações ~24–30 h ≈ **~30 h de máquina**,
dos quais só as ablações exigem paciência — se necessário, reduza as seeds
das ablações (ex.: `train.seeds: [42, 123, 7]` numa cópia da config) e
registre a redução; o custo cai para ~60%.

## Interpretação dos resultados

O REPORT.md gerado traz, por cenário, a tabela média ± IC95 de todas as
métricas, os p-values corrigidos e um parágrafo de interpretação gerado por
template (regras determinísticas sobre os números — não por LLM). Ao ler:

- **espera média** diz eficiência; **p95 e espera máxima** dizem justiça —
  um método só é bom se as duas famílias estiverem saudáveis;
- no `balanceado`, `fixo_proporcional` ≈ `fixo_igual` por construção — se o
  DQN vence aí, aprendeu algo além de "copiar a proporção";
- no `pico_invertido`, um DQN que só decorou "priorize a avenida" degrada
  visivelmente — é o cenário de generalização;
- o gráfico "espera na via local vs na avenida" mostra o trade-off de
  justiça: procure métodos no canto inferior esquerdo.

## Limitações honestas

- **Sim-to-real gap**: SUMO com car-following calibrado de fábrica, sem
  pedestres, ônibus, faixas exclusivas, acidentes ou comportamento paulistano
  real. Resultados são evidência de conceito, não de implantação.
- **Um único cruzamento**: coordenação em rede (onda verde, spillback entre
  cruzamentos) só na Fase 2.
- **Demanda sintética**: taxas estacionárias por episódio; sem dados reais
  da CET (fica para a Fase 4).
- **Observação perfeita**: o agente vê filas exatas; detectores reais têm
  ruído e cobertura parcial.
- **Um algoritmo**: DQN apenas; PPO e afins ficam como ablação futura.

## Fase 2 — SEU grid: declare as ruas movimentadas, o agente aprende os semáforos

A Fase 2 generaliza tudo acima para uma malha NxM. Você descreve o grid em
`configs/grid.yaml` — cada rua com **nome, classe e movimento**:

```yaml
grid:
  rows:                                   # ruas leste-oeste, norte -> sul
    - {name: "Rua Harmonia", class: local,   flow_vph: 250}
    - {name: "Av. Paulista", class: avenida, flow_vph: 1100}   # <- a movimentada
    - {name: "Rua Wisard",   class: local,   flow_vph: 200}
  cols:                                   # ruas norte-sul, oeste -> leste
    - {name: "Rua Girassol",   class: local,   flow_vph: 220}
    - {name: "Av. Rebouças",   class: avenida, flow_vph: 900}
    - {name: "Rua Aspicuelta", class: local,   flow_vph: 260}
```

`class: avenida` = 3 faixas/sentido a 60 km/h; `local` = 1 faixa a 40 km/h.
`flow_vph` é quantos veículos/hora entram por cada ponta da rua — as ruas
mais movimentadas são simplesmente as de maior fluxo. Os veículos viram nos
cruzamentos pelas frações de `turn_shares` (70/15/15 por padrão), então o
tráfego se espalha pela malha como numa cidade real.

**Importante:** o agente NÃO recebe a lista de ruas movimentadas. Cada
cruzamento observa apenas suas filas locais (obs_minimal) e uma única rede
DQN — compartilhada por todos os cruzamentos (*parameter sharing*) — precisa
aprender sozinha a dar prioridade a quem tem demanda. Os fluxos declarados
só são usados pela simulação (para gerar o tráfego) e pelo baseline
`fixo_proporcional` (que continua "injusto a favor" por conhecê-los).

```bash
traffic-rl grid-train --smoke        # valida o treino do grid em ~1 min
traffic-rl grid-train                # 5 seeds × 600k transições
traffic-rl grid-compare              # baselines + DQN -> results/grid/REPORT.md
traffic-rl grid-train --grid meu_bairro.yaml   # o SEU grid
```

Segurança idêntica à Fase 1, POR cruzamento (verde mín/máx, amarelo 3 s,
all-red 2 s); baselines instanciados um por cruzamento; métricas e REPORT.md
iguais, com a espera separada por avenidas × locais conforme as classes que
você declarou. Detalhes de projeto: ADR-013 a ADR-016 em docs/DECISOES.md.

**Grid com dados reais:** `configs/grid_paulista_real.yaml` traz a região da
Paulista com demanda ancorada em números públicos (volume diário medido da
Paulista + composição de frota da CET/MSVP 2019), estimativas marcadas e
escala documentada — ver ADR-018. Use com
`traffic-rl grid-train --grid configs/grid_paulista_real.yaml`.

Tempo de referência (8 núcleos): treino do grid 3×3 ≈ 1,5–2 h/seed
(600k transições ÷ 9 cruzamentos ≈ 67k decisões ≈ 93 episódios por
cruzamento — orçamento dobrado vs Fase 1 porque o problema multi-cruzamento
é mais difícil); 5 seeds ≈ 8–10 h sequencial (rode durante a noite;
retomável por checkpoint). Avaliação completa ≈ 1 h.

## Roadmap

- **Fase 2** (entregue nesta versão): grid NxM multi-agente com hierarquia
  declarada por YAML e política DQN compartilhada.
- **Fase 3**: hierarquia viária explícita na observação/recompensa e
  coordenação entre cruzamentos (onda verde emergente).
- **Fase 4**: malha real de bairro de SP via OpenStreetMap; calibração com
  contagens reais.

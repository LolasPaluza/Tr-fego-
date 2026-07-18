# Registro de decisões técnicas (ADR leve)

Formato: contexto → decisão → consequências. Uma entrada por escolha
não-óbvia feita durante a implementação da Fase 1.

## ADR-001 — SUMO via pip (`eclipse-sumo`), não via apt

**Contexto:** a spec exige SUMO ≥ 1.19; o apt do Ubuntu do sandbox oferece
1.18. **Decisão:** instalar `eclipse-sumo` (1.27.1) via pip, que embute os
binários; `sumo_home.py` resolve `SUMO_HOME` automaticamente (env var do
usuário tem precedência). **Consequências:** instalação idêntica em qualquer
máquina (`pip install -e .` basta), versão acima do mínimo; binários pip são
os oficiais do projeto Eclipse SUMO. `validate_setup.py` verifica a versão.

## ADR-002 — Ambiente TraCI próprio em vez de construir sobre sumo-rl

**Contexto:** a spec lista sumo-rl na stack, mas exige vermelho total de 2 s
em toda transição e verde máximo imposto pelo wrapper. O modelo de fases do
sumo-rl deriva os estágios do tlLogic e insere apenas amarelos — all-red e
verde máximo forçado não são representáveis sem contorcer a API.
**Decisão:** ambiente Gymnasium próprio (~300 linhas) controlando fases por
`setRedYellowGreenState` via TraCI/libsumo, com a tabela de fases derivada
dinamicamente do .net.xml (sumolib). A recompensa `r_espera` replica a
definição `diff-waiting-time` do sumo-rl (inclusive a escala ÷100), mantendo
comparabilidade com a literatura que usa sumo-rl. **Consequências:** controle
total dos tempos de segurança (testados em unidade); uma dependência a menos;
custo de manter nosso próprio loop TraCI (mitigado por testes de contrato).

## ADR-003 — Ação = "escolher o próximo estágio" (não "manter/trocar")

**Contexto:** duas formulações comuns de ação para semáforos em RL.
**Decisão:** `Discrete(4)`: o agente escolhe o próximo estágio verde;
repetir o atual = estender. **Consequências:** o agente pode pular estágios
sem demanda (impossível no manter/trocar cíclico); max-pressure e atuado
expressam-se naturalmente na mesma interface; espaço de ação segue pequeno.

## ADR-004 — Verde máximo com troca cíclica forçada

**Contexto:** sem teto de verde, uma política pode congelar um estágio para
sempre (starvation estrutural). **Decisão:** ao atingir `max_green_s`
(120 s), o wrapper força a troca para o estágio seguinte na ordem cíclica.
**Consequências:** nenhuma política congela o cruzamento; a escolha do
"próximo" é neutra (cíclica) para não embutir heurística de prioridade no
wrapper.

## ADR-005 — libsumo + subprocessos (SubprocVecEnv fork)

**Contexto:** libsumo é ~10× mais rápido que TraCI/TCP, mas suporta UMA
simulação por processo; o treino precisa de env de treino + env de avaliação
periódica simultâneos. **Decisão:** cada env roda num subprocesso
(`SubprocVecEnv`, `start_method="fork"`); a avaliação final usa envs
sequenciais no mesmo processo (um fecha antes do outro abrir).
**Consequências:** velocidade de libsumo em todo lugar; a restrição vira
regra de arquitetura documentada (CLAUDE.md) em vez de bug latente.

## ADR-006 — Detecção do atuado_gap via medidas de faixa do TraCI

**Contexto:** a spec pede controle atuado "via lógica sobre os detectores do
SUMO". **Decisão:** usar a ocupação por faixa (`getLastStepVehicleNumber`),
que é exatamente a medida de um detector de área E2 cobrindo a faixa, exposta
na `Observation` — em vez de declarar elementos `<laneAreaDetector>` no XML.
**Consequências:** mesma informação física, arquivo de rede mais simples e o
atuado usa o MESMO contrato de observação dos demais controladores; a
equivalência está documentada aqui e no docstring do controlador.

## ADR-007 — Chegadas via `probability` (binomial), .rou.xml determinístico

**Contexto:** a spec pede chegadas estocásticas binomiais/Poisson e episódios
de avaliação com seeds distintas. **Decisão:** flows com
`probability = veíc/h ÷ 3600` (Bernoulli por segundo ⇒ binomial); o MESMO
.rou.xml serve a todos os episódios e a estocasticidade vem de `sumo --seed`.
**Consequências:** um arquivo por cenário (regenerado sob demanda), episódios
reprodutíveis por seed, sem gerar N arquivos de rota.

## ADR-008 — Teleport do SUMO desligado (`--time-to-teleport -1`)

**Contexto:** por padrão o SUMO teleporta veículos presos > 300 s, o que
esconderia exatamente o fenômeno (starvation/gridlock) que queremos medir.
**Decisão:** desligar teleport em todos os experimentos. **Consequências:**
políticas ruins produzem filas reais e métricas honestas; risco de gridlock
irrecuperável é baixo num cruzamento único com demanda abaixo da capacidade.

## ADR-009 — `--waiting-time-memory 100000`

**Contexto:** o default do SUMO (100 s) faz `getAccumulatedWaitingTime`
esquecer espera antiga — quebraria `r_espera` e o starvation check (120 s).
**Decisão:** memória de espera maior que qualquer episódio (100000 s).

## ADR-010 — DQN avaliado como best-per-seed com episódios agrupados

**Contexto:** o protocolo pede "DQN best-per-seed" na avaliação final; é
preciso decidir como agregar as 5 seeds na estatística. **Decisão:** avaliar
o melhor checkpoint de CADA seed nos 20 episódios e agrupar os 100 episódios
sob o método "dqn" (coluna `dqn_seed` preserva a origem). **Consequências:**
a variabilidade entre seeds entra na comparação com os baselines (mais
honesto do que escolher a melhor seed); o Mann-Whitney compara distribuições
com n=100 vs n=20, o que o teste suporta.

## ADR-011 — Escala ÷100 nas três recompensas

**Contexto:** ablação de recompensa exige comparar funções com os MESMOS
hiperparâmetros de DQN. **Decisão:** todas as recompensas divididas por 100,
ficando em ordem de grandeza ~[-5, 5]. **Consequências:** taxa de
aprendizado/clipping servem às três variantes; nenhuma vence por mera escala.

## ADR-012 — Torch com CUDA no sandbox (custo só de disco)

**Contexto:** o índice de wheels só-CPU do PyTorch ficou inacessível atrás do
proxy do sandbox; o pacote padrão (com libs CUDA) instala e funciona em CPU.
**Decisão:** manter o pacote padrão no sandbox; o README recomenda o wheel
só-CPU na máquina local (opcional, ~10× menor). **Consequências:** nenhum
impacto em resultados (device="cpu" em todo lugar); apenas ~4 GB de disco a
mais no ambiente cloud.

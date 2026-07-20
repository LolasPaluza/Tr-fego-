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

## ADR-013 — Fase 2: independent learners com parameter sharing (um DQN, K cruzamentos)

**Contexto:** controlar K semáforos com RL admite (a) um agente centralizado
(ação conjunta 4^K — explode), (b) K redes independentes (K× o custo de
treino, nada compartilhado), (c) UMA rede compartilhada aplicada a cada
cruzamento sobre sua observação local. **Decisão:** (c), implementado como um
`VecEnv` do sb3 com `num_envs=K` sobre a MESMA simulação — cada decisão
coleta K transições para o replay buffer comum. **Consequências:** o custo de
treino independe de K por transição; a política generaliza entre cruzamentos
(local×local e avenida×avenida compartilham estrutura); limitação honesta:
sem comunicação entre agentes, coordenação (onda verde) só emerge
implicitamente — tema da Fase 3. Exige observação de tamanho FIXO ⇒ grid usa
obs_minimal (9d), cujas features são normalizadas pelas capacidades locais.

## ADR-014 — Treino do grid: libsumo (treino) + TraCI (avaliação periódica)

**Contexto:** o truque da Fase 1 (SubprocVecEnv, 1 env por subprocesso) não
se aplica — o GridVecEnv já É um VecEnv e não pode ser aninhado; libsumo
segue limitado a 1 simulação por processo. **Decisão:** env de treino em
libsumo (rápido, em processo) e env de avaliação periódica em TraCI
(processo sumo externo) — mecanismos independentes que coexistem.
**Consequências:** avaliação ~5× mais lenta que libsumo, mas rara (a cada
eval_freq); zero IPC custom.

## ADR-015 — Demanda do grid: roteador próprio por caminhada aleatória

**Contexto:** num grid, cada veículo precisa de rota completa; flows com
`probability` (Fase 1) não escolhem conversões. Alternativas: jtrrouter do
SUMO (mais uma etapa externa com semântica própria) ou roteador em Python.
**Decisão:** roteador próprio — chegadas Bernoulli por segundo em cada
entrada declarada, conversões sorteadas por `turn_shares` em cada cruzamento
até sair da malha; arquivo `.rou.xml` por episódio, determinístico na seed.
**Consequências:** controle e testabilidade totais (testes garantem que toda
caminhada termina e que o volume bate com os fluxos declarados); ids de
veículo carregam classe+rua de origem para a análise de justiça.

## ADR-016 — Fases do semáforo derivadas por geometria

**Contexto:** a Fase 1 mapeava movimentos por nomes de aresta (`in_E` etc.);
o grid tem centenas de arestas com ids próprios. **Decisão:** generalizar a
tabela de fases para derivar (aproximação, movimento) da GEOMETRIA — rumo da
aresta de entrada dá a aproximação; produto vetorial entre rumos de entrada
e saída dá o tipo de conversão. **Consequências:** o mesmo código serve à
Fase 1 (testes seguem verdes) e a qualquer cruzamento em cruz do grid;
pré-requisito direto da Fase 4 (malhas OSM têm geometria, não nomes).

## ADR-017 — Calibração da demanda default do grid (medida, não chutada)

**Contexto:** os primeiros fluxos default do grid.yaml (avenidas 1100/900,
esquerda 15%) produziam gridlock estrutural em 1 h simulada: as conversões
das avenidas despejam veículos nas locais de 1 faixa, onde a espera pela fase
de esquerda bloqueia a faixa inteira; as filas retornam (spillback) pelos
quarteirões de 300 m e travam a malha — TODOS os métodos afogavam (espera
média > 800 s, ~40% dos veículos presos), inutilizando a comparação.
**Decisão:** calibrar o default por sondagem empírica com o baseline forte
(fixo_proporcional): avenidas 550/450, locais 120–150, conversões 84/10/6 —
ponto "carregado porém viável" (espera ~105 s, 93% concluem; fixo_igual
~176 s, deixando espaço para os métodos se diferenciarem). O aviso de
capacidade está comentado no próprio grid.yaml. **Consequências:** o
fenômeno de saturação continua acessível (basta subir os fluxos), mas o
default conta uma história comparável; a super-saturação vira experimento
consciente, não armadilha.

## ADR-018 — Grid ancorado em dados reais (grid_paulista_real.yaml)

**Contexto:** o usuário quer aproximar a simulação do trânsito real de SP
antes da Fase 4. Dados de demanda por via são escassos publicamente: o
volume diário da Av. Paulista (~82 mil veíc/dia, Prefeitura/Associação
Paulista Viva) e a composição da frota (CET MSVP 2019: pesados ≈ 4,4%) são
verificáveis; as contagens por transversal (Augusta, Haddock Lobo, Alamedas,
Brigadeiro) não foram acessíveis (PDFs da CET bloqueados pela rede do
sandbox). Waze fornece ESTADO (velocidades/lentidão), não DEMANDA
(veíc/h) — não substitui contagem. **Decisão:** criar
`configs/grid_paulista_real.yaml` com (a) valores medidos citados, (b)
estimativas por hierarquia viária claramente marcadas como estimativas, e
(c) escala global ~0,17 documentada (a malha 3×3 abstrata tem capacidade
menor que o corredor real; sem a escala, todos os métodos afogam — ADR-017).
As PROPORÇÕES entre vias são o dado real preservado. **Consequências:**
realismo honesto e auditável hoje; as estimativas viram contagens reais na
Fase 4 (MSVP/CET completo + OSM), trocando números num único arquivo.

## ADR-012 — Torch com CUDA no sandbox (custo só de disco)

**Contexto:** o índice de wheels só-CPU do PyTorch ficou inacessível atrás do
proxy do sandbox; o pacote padrão (com libs CUDA) instala e funciona em CPU.
**Decisão:** manter o pacote padrão no sandbox; o README recomenda o wheel
só-CPU na máquina local (opcional, ~10× menor). **Consequências:** nenhum
impacto em resultados (device="cpu" em todo lugar); apenas ~4 GB de disco a
mais no ambiente cloud.

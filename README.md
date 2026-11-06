# qwm-qem — World model latente (JEPA) para mitigação de erros quânticos sob drift

Fase 1 do pré-projeto: tudo em simulador com drift controlado, onde o estado físico verdadeiro
do dispositivo é conhecido. O world model aprende um latente `z_t` a partir de telemetria de
calibração e sondas baratas, prevê `z_{t+1}` condicionado à ação (executar ou recalibrar) e usa
`z_t` para corrigir valores esperados.

## Estrutura

```
qwm/
  config.py      configurações (dispositivo, dados, modelo)
  device.py      processo de drift do estado oculto θ_t (T1/T2 + TLS, erro coerente, leitura, p2)
  noise.py       θ_t -> NoiseModel do Qiskit Aer
  circuits.py    pool de circuitos-alvo (Trotter TFIM, camadas rasas), sondas, folding, correção de leitura
  dataset.py     gera a trajetória (estado, telemetria, sondas, ações, execuções em 3 escalas de ZNE, gêmeo)
  features.py    tabelas de mitigação e observações por passo
  model.py       encoder, preditor, cabeça de mitigação, SIGReg
  train.py       treino do world model (predição latente + SIGReg + mitigação)
  baselines.py   bruto, leitura, ZNE linear/Richardson, gêmeo digital, ML-QEM estático e online
  evaluate.py    métricas de H1 (MAE), H2 (sonda linear), H4 (AUROC de TLS)
  stats.py       Wilcoxon pareado, Holm, IC bootstrap
scripts/
  run_pipeline.py  uma semente, ponta a ponta
  run_seeds.py     várias sementes em paralelo + teste estatístico
```

## Como rodar

```bash
pip install -r requirements.txt
# teste rápido (~2 min de geração em CPU)
python scripts/run_pipeline.py --seed 0 --steps 240 --epochs 8 --out runs_smoke
# uma semente completa: 30 dias, 1440 passos (~10 min de geração em CPU)
python scripts/run_pipeline.py --seed 0 --out runs
# protocolo completo: 15 sementes, Wilcoxon + Holm
python scripts/run_seeds.py --seeds 15 --jobs 4 --out runs
# estresse: drift 2x mais rápido
python scripts/run_seeds.py --seeds 15 --jobs 4 --out runs_drift2 --extra "--drift-scale 2"
```

A geração de dados é CPU (Qiskit Aer, matriz densidade, 5 qubits) e é cacheada em
`data_seed*.npz`; o treino usa GPU se houver. No Colab, o gargalo é a geração: use `--jobs`
igual ao número de núcleos.

## Decisões de modelagem (para revisar)

- **Dinâmica exógena.** Executar circuitos não muda o drift; só `recalibrar` muda o estado, e ela
  zera erro coerente e deriva de leitura, mas não restaura T1/T2.
- **Telemetria é cega ao erro coerente pós-calibração.** A calibração mede o resíduo logo após
  calibrar; o que acumula depois só aparece nas sondas (`overrot`, `cx_echo`). Isso é o que dá
  espaço para o world model superar o ML-QEM com telemetria.
- **Mitigação como resíduo** sobre o valor com correção de leitura: `ŷ = y_ro + M(z, x, ỹ)`.
- **Valor ideal** disponível porque o pool é classicamente simulável (como em Liao et al. 2024).
- **Gêmeo digital** = NoiseModel montado da última telemetria; mitigação aditiva
  `ŷ = ỹ − (y_gêmeo − y_ideal)`.
- **ZNE** usa os valores com correção de leitura (folding não amplifica erro de leitura).
- **ML-QEM estático** treina em treino+validação; o world model usa a validação para escolher a época.
- **SIGReg** é implementação própria a partir do artigo do LeJEPA; o pacote oficial é `lejepa`.

## Resultado preliminar (semente 0, 30 dias, configuração padrão)

| Método | MAE teste | Shots |
| --- | --- | --- |
| ML-QEM RF (telemetria) | 0.0049 | ×1 |
| ML-QEM online | 0.0093 | ×1 |
| Gêmeo digital | 0.0110 | ×1 |
| World model completo | 0.0165 | ×1 |
| ZNE linear (com correção de leitura) | 0.0174 | ×2 |
| World model sem preditor | 0.0245 | ×1 |
| Bruto | 0.0519 | ×1 |

Leitura: o preditor latente melhora o world model (0.0245 → 0.0165) e o latente supera as
observações brutas na sonda linear (nRMSE 1.11 vs 1.64), mas o RF com telemetria vence.
Acrescentar sondas ou o próprio z ao RF não melhorou (0.0054 e 0.0056). Diagnóstico: o
benchmark ainda não estressa o drift. O pool de 40 circuitos é o mesmo no treino e no teste,
então uma correção por circuito resolve quase tudo. Próximos ajustes do desenho experimental:
circuitos inéditos no teste, varredura de `--drift-scale` e calibração menos frequente.

## Pendências conhecidas

- [ ] split com circuitos inéditos no teste (generalização de circuito, não só de tempo)
- [ ] varredura da taxa de drift (`--drift-scale` 1, 2, 4) e do intervalo de calibração
- [ ] investigar o ML-QEM MLP do sklearn (MAE 0.13, provável problema de ajuste)
- [ ] cabeça de mitigação mais regularizada ou baseada em árvores sobre z

- [ ] H3: planejamento no latente para escolher a configuração de mitigação sob orçamento de shots
- [ ] crosstalk (ZZ espúrio entre vizinhos ociosos) no simulador
- [ ] testes fora da distribuição com outras formas de drift (degraus, senoides)
- [ ] poucos eventos TLS por trajetória de teste (~3): H4 só faz sentido agregando sementes
- [ ] world model ainda sobreajusta em trajetórias curtas: regularizar a cabeça de mitigação
- [ ] validação com histórico real de calibração da IBM

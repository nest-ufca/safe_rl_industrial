# Guia de uso do `safe_rl_industrial`

## 1. Objetivo

Este guia orienta a instalação, a leitura e a execução do código usado no
estudo de proteção de serviços URLLC em um cenário de Indústria 4.0 com
*network slicing* e aprendizagem por reforço.

O fluxo principal compara:

- `marl_safe_sac`: proposta MARL com um agente inter-slice e três agentes
  intra-slice;
- `ssr`: método de referência usado como baseline;
- opcionalmente, `ray_protect`: versão de agente único implementada em Ray.

O cenário contém 30 UEs eMBB, 40 UEs URLLC e 30 UEs mMTC.

> O código é uma versão de pesquisa. Uma execução somente deve ser chamada de
> reprodução do artigo após a confirmação dos dados e de todos os parâmetros.

## 2. Arquivos principais

| Arquivo | Função |
|---|---|
| `simu_marl_industrial.py` | Treina e testa a proposta MARL com Ray/RLlib e SAC. |
| `simu_industrial.py` | Executa os agentes de referência, principalmente `ssr`. |
| `simu_ray_protect.py` | Executa a versão de agente único que protege URLLC. |
| `agents/marl_safe.py` | Define observações, ações e recompensas da proposta. |
| `channels/quadriga.py` | Carrega os canais QuaDRiGa. |
| `channels/mimic_quadriga.py` | Gera um canal sintético para testes. |
| `env_config/industrial.yml` | Define slices, requisitos, RBs e episódios. |
| `results/gen_results.py` | Gera gráficos e contabiliza violações. |
| `sixg_radio_mgmt/` | Submódulo com o ambiente de comunicação. |

## 3. Clonagem

```powershell
git clone --recurse-submodules https://github.com/nest-ufca/safe_rl_industrial.git
cd safe_rl_industrial
```

Se `sixg_radio_mgmt/` estiver vazio:

```powershell
git submodule update --init --recursive
```

## 4. Ambiente Python

O `Pipfile` e o `Pipfile.lock` foram gerados para Python 3.10. É possível ter
Python 3.10 e 3.11 instalados simultaneamente no Windows. Para este projeto,
use explicitamente o 3.10:

```powershell
py -0p
py -3.10 --version
py -3.10 -m pip install --user pipenv
py -3.10 -m pipenv --python 3.10
py -3.10 -m pipenv sync
```

Teste as importações:

```powershell
py -3.10 -m pipenv run python -c "import numpy, scipy, torch, ray, gymnasium, pettingzoo; print('Ambiente OK'); print('Ray:', ray.__version__); print('Torch:', torch.__version__)"
```

Principais versões registradas no lockfile:

| Pacote | Versão |
|---|---:|
| Python | 3.10 |
| NumPy | 1.26.4 |
| SciPy | 1.13.0 |
| Ray/RLlib | 2.10.0 |
| Stable-Baselines3 | 2.3.0 |
| PyTorch | 2.2.2 |
| Gymnasium | 0.28.1 |
| PettingZoo | 1.24.3 |

## 5. Canais QuaDRiGa

Os canais não estão no GitHub. Cada conjunto `hall_1`, `hall_2` ou `hall_3`
pode ocupar dezenas de gigabytes. Mantenha os arquivos externamente:

```text
D:\dados_quadriga\hall_2\
├── sim_1.mat
├── sim_2.mat
└── ...
```

Defina o diretório no mesmo PowerShell usado para iniciar a simulação:

```powershell
$env:QUADRIGA_CHANNEL_DIR = 'D:\dados_quadriga\hall_2'
```

Se a variável não for definida, o código procura os arquivos em
`channels/quadriga_channels/`. Esse diretório é ignorado pelo Git.

Valide um canal antes de executar:

```powershell
py -3.10 -m pipenv run python -c "from scipy.io import whosmat; print(whosmat(r'D:\dados_quadriga\hall_2\sim_1.mat'))"
```

Os canais conhecidos têm a variável `H` com dimensão
`(1, 4, 100, 100, 1001)`: 100 UEs, 100 RBs e 1001 amostras temporais.

## 6. Canal sintético e canal real

Para um teste que não usa os arquivos grandes, escolha em cada script:

```python
"channel_class": MimicQuadriga,
```

Para usar os canais reais:

```python
"channel_class": QuadrigaChannels,
```

No baseline `simu_industrial.py`, a forma equivalente é:

```python
ChannelClass=QuadrigaChannels,
```

Registre sempre qual classe de canal foi usada. Um resultado com
`MimicQuadriga` não reproduz a campanha do artigo.

## 7. CPU e GPU

`simu_marl_industrial.py` reserva uma GPU na chamada `.resources(...)`. Em uma
máquina sem GPU NVIDIA compatível, use:

```python
.resources(
    num_gpus=0,
    num_gpus_per_worker=0,
    num_gpus_per_learner_worker=0,
)
```

Para depuração inicial, mantenha:

```python
debug_mode = True
enable_restore = False
```

`enable_restore = False` evita tentar carregar checkpoints antigos ou
incompatíveis.

## 8. Teste reduzido

O primeiro objetivo é apenas validar dependências, canais e interação entre os
agentes. Não use esse resultado na comparação científica.

Em `env_config/industrial.yml`, reduza temporariamente:

```yaml
simulation:
  simu_name: industrial
  max_number_steps: 20
  max_number_episodes: 3
```

Em `simu_marl_industrial.py`, use:

```python
training_flag = True
debug_mode = True
enable_restore = False

env_config = {
    # demais opções mantidas
    "training_episodes": 2,
    "max_episode_number": 2,
    "training_epochs": 1,
    "testing_episodes": 1,
    "episode_evaluation_freq": 1,
    "number_evaluation_episodes": 1,
    "eval_initial_env_episode": 2,
}
```

Use `MimicQuadriga` para o primeiro teste ou disponibilize `sim_1.mat` a
`sim_3.mat`. Execute:

```powershell
py -3.10 -m pipenv run python simu_marl_industrial.py
```

O Ray deve criar dados em `ray_results/industrial/marl_safe_sac/` e os
históricos de teste devem aparecer em `hist/industrial/marl_safe_sac/`.

## 9. Experimento completo

Depois do teste reduzido, restaure no YAML:

```yaml
simulation:
  simu_name: industrial
  max_number_steps: 1000
  max_number_episodes: 100
```

Na proposta MARL, a divisão atual é:

```python
"training_episodes": 70,
"max_episode_number": 70,
"training_epochs": 3,
"testing_episodes": 30,
"episode_evaluation_freq": 70,
"number_evaluation_episodes": 30,
"eval_initial_env_episode": 70,
```

Execute:

```powershell
py -3.10 -m pipenv run python simu_marl_industrial.py
```

O número exato de épocas da campanha final ainda deve ser confirmado. Não
sobrescreva resultados anteriores e registre o commit de cada execução.

## 10. Baseline SSR

Em `simu_industrial.py`, confirme:

```python
agents = ["ssr"]
ChannelClass=QuadrigaChannels
```

Execute:

```powershell
py -3.10 -m pipenv run python simu_industrial.py
```

Com a divisão 70/30, os históricos de teste são esperados como episódios 70 a
99 em `hist/industrial/ssr/`.

## 11. Gráficos

Em `results/gen_results.py`, mantenha:

```python
episodes = np.arange(70, 100)
agent_names = ["marl_safe_sac", "ssr"]
```

Execute a partir da raiz do repositório:

```powershell
py -3.10 -m pipenv run python results/gen_results.py
```

Os gráficos principais são gravados em `results/industrial/`.

## 12. Resultados qualitativos esperados

- redução das violações do slice URLLC com `marl_safe_sac`;
- latência URLLC próxima ou abaixo de 1 ms durante a maior parte do teste;
- possível redução do throughput eMBB ao proteger URLLC;
- menos violações totais na proposta do que no baseline `ssr`.

Não exija igualdade numérica antes de confirmar todos os parâmetros e os canais.

## 13. Divergências a investigar

1. O artigo informa frequência de 6 GHz, mas o YAML contém `2400000`.
2. O artigo informa potência de 35 dBm, mas `channels/quadriga.py` usa `0.1 W`,
   equivalente a 20 dBm.
3. As observações no código incluem eficiência espectral e estatísticas extras.
4. A recompensa inter-slice contém uma penalização adicional para violações
   URLLC.
5. O número de épocas da campanha final precisa ser confirmado.

Não corrija essas diferenças silenciosamente. Altere uma variável por vez,
identifique a configuração e compare os resultados.

## 14. Erros comuns

| Erro | Causa provável | Ação |
|---|---|---|
| `FileNotFoundError: sim_X.mat` | Diretório incorreto ou episódio ausente. | Confira `QUADRIGA_CHANNEL_DIR`. |
| `KeyError: 'H'` | `.mat` incompatível. | Use `whosmat` e confirme a variável `H`. |
| Erro de dimensão | UEs, RBs ou TTIs diferentes. | Confirme `(1, 4, 100, 100, 1001)`. |
| Ray aguarda recursos | O script reservou uma GPU inexistente. | Defina os três valores de GPU como zero. |
| Erro de checkpoint | Resultado incompleto/incompatível. | Use `enable_restore = False`. |
| Falta `sixg_radio_mgmt` | Submódulo não inicializado. | Execute `git submodule update --init --recursive`. |
| Falta de memória | Canal e matrizes auxiliares são grandes. | Reduza episódios/TTIs e feche outros programas. |

## 15. Registro de cada experimento

Anote:

- data, commit e arquivos modificados;
- hall e classe de canal;
- episódios, TTIs, seeds e agente;
- CPU/GPU, memória e tempo de execução;
- potência, frequência e requisitos dos slices;
- caminho dos resultados;
- erros e soluções adotadas.

Até confirmar a correspondência exata com o artigo, use o termo **replicação
com os canais disponíveis**, não **reprodução**.

## 16. Entregáveis sugeridos ao aluno

1. Ambiente instalado e teste reduzido funcionando.
2. Leitura e descrição dos arquivos principais.
3. Baseline `ssr` executado.
4. Proposta `marl_safe_sac` executada.
5. Gráficos de throughput, latência e violações.
6. Tabela comparando parâmetros do artigo e do código.
7. Relatório das divergências e modificações.

Nunca envie ao GitHub `venv/`, `.idea/`, `ray_results/`, `hist/`, checkpoints,
arquivos `.mat` ou arquivos compactados com os canais.

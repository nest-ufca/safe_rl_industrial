# Safe RL Industrial

Código de pesquisa usado no estudo de proteção de serviços URLLC em um cenário
de Indústria 4.0 com *network slicing* e aprendizagem por reforço.

Este repositório está associado ao artigo do SBrT 2024 e foi publicado para
permitir o estudo, a replicação dos experimentos e a evolução do trabalho.

## Conteúdo principal

- `simu_marl_industrial.py`: proposta MARL baseada em SAC;
- `simu_industrial.py`: agentes de referência, incluindo o baseline `ssr`;
- `simu_ray_protect.py`: agente de proteção implementado com Ray/RLlib;
- `env_config/industrial.yml`: parâmetros do cenário industrial;
- `channels/quadriga.py`: leitura dos canais gerados no QuaDRiGa;
- `channels/mimic_quadriga.py`: canal sintético para testes iniciais;
- `results/gen_results.py`: geração dos gráficos e análise de violações;
- `GUIA_DE_USO.md`: roteiro detalhado de instalação e execução.

## Clonagem

O projeto utiliza `sixg_radio_mgmt` como submódulo:

```powershell
git clone --recurse-submodules https://github.com/nest-ufca/safe_rl_industrial.git
cd safe_rl_industrial
```

Se o repositório já foi clonado sem os submódulos:

```powershell
git submodule update --init --recursive
```

## Ambiente recomendado

O ambiente original foi registrado para Python 3.10. Recomenda-se manter essa
versão para evitar incompatibilidades com as versões antigas de Ray/RLlib e
PyTorch.

```powershell
py -3.10 -m pip install --user pipenv
pipenv --python 3.10
pipenv sync
```

## Canais QuaDRiGa

Os arquivos `.mat` são grandes e não fazem parte do Git. Defina o diretório que
contém `sim_1.mat`, `sim_2.mat`, etc. antes de executar com
`QuadrigaChannels`:

```powershell
$env:QUADRIGA_CHANNEL_DIR = 'D:\dados_quadriga\hall_2'
```

Sem essa variável, o código procura os arquivos em
`channels/quadriga_channels/`, que também é ignorado pelo Git.

## Primeira execução

Comece com o canal sintético (`MimicQuadriga`) e uma configuração reduzida.
Somente depois valide a leitura dos canais reais e execute o experimento
completo. Consulte [GUIA_DE_USO.md](GUIA_DE_USO.md) para o passo a passo, os
parâmetros conhecidos e as divergências que ainda precisam ser investigadas.

## Dados e resultados não versionados

O repositório não armazena canais QuaDRiGa, ambientes virtuais, checkpoints do
Ray, históricos de episódios ou arquivos compactados. Esses artefatos devem ser
mantidos em armazenamento externo e identificados no registro de cada
experimento.

## Licença

Consulte o arquivo [LICENSE](LICENSE).

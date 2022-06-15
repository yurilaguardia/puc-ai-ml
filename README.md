# puc-ai-ml

Repositório do projeto final para o curso "Inteligência Artificial e Aprendizado de Máquina" da PUC-MG

## Pré-requisitos
- Python 3.6+ já instalado na sua máquina
- git (para fazer git clone)

## Instalação

### Clonar este repositório utilizando:

```bash
git clone https://github.com/yurilaguardia/puc-ai-ml.git
```

### OPCIONAL: criar ambiente virtual para isolar o projeto
```bash
python -m venv venv
source venv/bin/activate
```

### Instalar as dependências:

```bash
cd puc-ai-ml
pip install -r requirements.txt
```

## Execução

### Rodar as células do caderno jupyter `extract_songs.ipynb`

Neste caderno haverá celulas destinadas a fazer download do MillionSongSubset, para podermos simular em qualquer computador como se deu a extração de músicas da base original (o que fizemos na infraestrutura da AWS, utilizando instância EC2 e snapshot com HD contendo os 280GB de músicas - 1 milhão de tracks).

Após a célula que baixa o arquivo já ter sido rodada uma vez e o subset já estando em seu computador, não é mais necessário rodar as células de download e descompressão.

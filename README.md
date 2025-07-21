# 📊 FIAP Data Analytics D7 – Datathon (Recrutamento Inteligente)

Projeto de Recomendação de Candidatos para Vagas, usando Machine Learning e NLP, com deploy via [Streamlit](https://streamlit.io/) e integração com GitHub.

---

## 🔥 Sobre o Projeto

Este projeto desenvolve uma solução completa para auxiliar empresas e consultorias de RH na **seleção automatizada e inteligente de candidatos para vagas de TI**.  
Inclui análise exploratória, criação de modelos preditivos, engenharia de features avançada (incluindo NLP e matching técnico), e uma interface prática para simulação via Streamlit, funcionando **tanto localmente quanto online** (modo demo cloud).

---

## 🚦 Funcionalidades

- Cadastro manual ou por arquivo de vagas e candidatos
- Ranking de candidatos por probabilidade de sucesso/recomendação
- Visualização amigável dos detalhes do candidato (incluindo currículo expandido)
- Explicabilidade do modelo (principais variáveis)
- Deploy fácil no Streamlit Cloud ou execução local

---

## 🖥️ Como rodar o projeto

### 1. Clone o repositório

```bash
git clone https://github.com/shibbys/FiapDataAnaliticsD7-Datathon.git
cd FiapDataAnaliticsD7-Datathon
```
### 2. Escolha o modo de execução

No arquivo principal do app (`streamlit-Dashboard.py` ou `Datathon-Grupo-13-DTAT7-Analise-treinamento-Decision.ipynb`), defina no topo:

```python
MODO_LOCAL = 1  # Para rodar localmente com a base/modelo completo
MODO_LOCAL = 0  # Para rodar online (Streamlit Cloud) com amostras/modelo leve

```
### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Rode o app localmente
```bash
streamlit run dashboard-Streamlit.py
```
## ⚡️ Sobre os dados e modelos

- O projeto utiliza arquivos de base completos e amostras reduzidas (*_sample.json) para ambientes cloud.

- Para rodar localmente com o pipeline completo, hospedados aqui utilizando LFS.

- Modelos leves (modelo_rf_light.pkl) e arquivos de amostra garantem performance e deploy no Streamlit Cloud.

## 📁 Estrutura dos arquivos
```
├── json
│ ├── applicants.json
│ ├── applicants_sample.json
│ ├── prospects.json
│ ├── vagas.json
│ ├── vagas_sample.json
├── models
│ ├── features.pkl
│ ├── features_light.pkl
│ ├── importances.pkl
│ ├── importances_light.pkl
│ ├── modelo_rf.pkl
│ ├── modelo_rf_light.pkl
│ ├── threshold.txt
├── Datathon-Grupo-13-DTAT7-Analise-treinamento-Decision.ipynb
├── README.md
├── dashboard-Streamlit.py
├── requirements.txt
└── ...
```
## 📝 Observações

- O app exibirá uma mensagem informando se está em modo local ou cloud.

- No modo cloud, carregue apenas arquivos pequenos/sample.

- Para datasets completos, use localmente e baixe os arquivos conforme instruções.

## 👨‍💻 Autores e Contato
- Marlon J Fernandez (RM353490)
- Roberto C Muller (RM353491)
Dúvidas ou sugestões?

Abra um Issue ou envie um e-mail para [marlon.shibby@gmail.com]

## 🏆 Créditos

Projeto desenvolvido para o Datathon FIAP Data Analytics – 2024/2025.

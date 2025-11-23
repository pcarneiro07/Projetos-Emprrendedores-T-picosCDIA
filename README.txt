🌊 Monitor Artemis — Detecção de Anomalias em Dados Hídricos

Projeto executado localmente no VSCode usando Streamlit + Python

📌 Descrição Geral

O Monitor Artemis é um sistema interativo desenvolvido para análise e detecção de anomalias em dados hídricos.
Ele integra:

Pipeline de ingestão direto do Supabase

Normalização e cálculo de Z-scores

Ensemble de modelos de detecção de outliers

Clusterização automática dos níveis de anomalia

Dashboard interativo em Streamlit

Resumo automático diário gerado por IA (OpenAI)

O resultado é um índice diário de anomalia (0–100) classificado como:

Normal

Atenção

Crítica

E uma análise detalhada, incluindo abas de variáveis de qualidade da água (Quali) e nível/vazão (Quanti).

🧩 Arquivos do Projeto

A pasta ZIP contém:

📁 monitor-artemis/
│
├── app.py                 → código principal do Streamlit
├── requirements.txt       → lista de dependências

🛠 Como Rodar o Projeto no VSCode (LOCAL)
1. Instalar Python

Use Python 3.10 ou superior.

Verificar versão:

python --version


3. Instalar dependências
pip install -r requirements.txt


(requirements.txt carregado pelo projeto:)


requirements



5. Rodar a aplicação

No terminal do VSCode:

streamlit run app.py


O dashboard abrirá automaticamente no navegador:

http://localhost:8501


Arquivo principal:


app

📊 Funcionalidades Principais
✔ Card de índice diário

Mostra o nível da anomalia do dia selecionado.

✔ Gráfico de histórico

Evolução do índice ao longo do tempo com faixas de Normal/Atenção/Crítica.

✔ Aba Quali (Qualidade da água)

pH

Oxigênio dissolvido

Condutividade elétrica

Turbidez

Temperatura

Com coloração baseada em Z-score.

✔ Aba Quanti (Nível e vazão)

Vazão

Leitura de régua

Cota referenciada

✔ Resumo diário via IA

Texto natural gerado automaticamente com base:

situação do dia

variáveis ordenadas por anomalia

interpretação qualitativa (normal / leve desvio / forte anomalia)

Sem números, sem listas, sem alucinações.

🧠 Modelagem de Anomalias

O projeto utiliza um ensemble robusto envolvendo:

Isolation Forest

Local Outlier Factor (LOF)

One-Class SVM

Elliptic Envelope

Estatística robusta (robust Z-score MAD)

A combinação de modelos é feita via otimização de pesos Dirichlet, maximizando Silhouette Score com penalização de spreads exagerados.

Os limiares dos níveis (Normal, Atenção, Crítica) são aprendidos dinamicamente via K-Means.

🚀 Próximos Passos

Expandir o conjunto de dados (atualmente: maio/24 → outubro/24)

Suporte à sazonalidade

Ingestão automática contínua

Sistema de alertas automáticos (email/WhatsApp)

Deploy opcional na nuvem (Azure Container Apps)

👥 Autores

Projeto desenvolvido no curso de Ciência de Dados e Inteligência Artificial, integrando:

Pipeline

Modelagem

Dashboard

LLM

Equipe:
Pedro Carneiro
Raphael von Zuben
Pedro Lucas Amâncio
Leonardo Marchi
Gabriel Joaquim
Gabriel Joaquim

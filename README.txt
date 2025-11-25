🌊 Monitor Artemis — Detecção de Anomalias em Dados Hídricos

Sistema desenvolvido em Python + Streamlit, executado localmente no VSCode, com foco em análise inteligente e detecção automática de anomalias em dados hídricos provenientes do Supabase.

📌 Descrição Geral

O Monitor Artemis é um dashboard interativo que integra:

Pipeline de ingestão direto do Supabase

Normalização e cálculo de Z-scores robustos (MAD)

Ensemble de modelos de detecção de outliers

Clusterização automática para níveis de anomalia

Dashboard em Streamlit

Resumo diário gerado automaticamente por IA (OpenAI)

O sistema produz um índice diário (0–100) classificado como:

Normal

Atenção

Crítica

E entrega uma análise técnica detalhada distribuída nas abas:

Quali — variáveis de qualidade da água

Quanti — nível e vazão

Resumo IA — interpretação textual automática sem alucinações

🧩 Estrutura do Projeto

A pasta ZIP contém:

monitor-artemis/
│
├── app.py               # Código principal do Streamlit
├── requirements.txt     # Dependências do projeto

🛠 Como Rodar o Projeto no VSCode (LOCAL)
1. Verifique o Python

Use Python 3.10+.

python --version

2. Instale as dependências
pip install -r requirements.txt

3. Execute a aplicação
streamlit run app.py


O dashboard abrirá automaticamente em:

👉 http://localhost:8501

📊 Funcionalidades Principais
✔ Índice diário

Card exibindo o nível de anomalia do dia selecionado.

✔ Gráfico histórico

Visualização temporal do índice com faixas:

Normal

Atenção

Crítica

✔ Aba Quali (Qualidade da Água)

Inclui variáveis com coloração baseada no Z-score:

pH

Oxigênio dissolvido

Condutividade elétrica

Turbidez

Temperatura

✔ Aba Quanti (Vazão/Nível)

Vazão

Leitura de régua

Cota referenciada

✔ Resumo diário via IA

Geração automática de texto interpretativo considerando:

Situação do dia

Variáveis ordenadas por anomalia

Análise qualitativa (normal / leve desvio / forte anomalia)

Sem números, sem listas, sem alucinações.

🧠 Modelagem de Anomalias

O pipeline utiliza um ensemble robusto composto por:

Isolation Forest

Local Outlier Factor (LOF)

One-Class SVM

Elliptic Envelope

Estatística robusta (Z-score via MAD)

A fusão dos modelos é feita com:

Otimização de pesos Dirichlet

Maximização do Silhouette Score

Penalização de spreads exagerados

Os limiares dos níveis Normal/Atenção/Crítica são aprendidos dinamicamente via:

K-Means

🚀 Próximos Passos

Expandir o dataset (maio/24 → outubro/24 atualmente)

Suporte à sazonalidade

Ingestão contínua automatizada

Sistema de alertas (email/WhatsApp)

Deploy opcional na nuvem (Azure Container Apps)

👥 Autores

Projeto desenvolvido no curso de Ciência de Dados e Inteligência Artificial, integrando:

Pipeline

Modelagem

Dashboard

IA

Equipe

Pedro Carneiro

Raphael von Zuben

Pedro Lucas Amâncio

Leonardo Marchi

Gabriel Joaquim

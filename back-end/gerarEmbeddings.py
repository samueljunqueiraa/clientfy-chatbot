import google.generativeai as generativeai
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import pickle

load_dotenv()
chave_secreta = os.getenv('API_KEY')
generativeai.configure(api_key=chave_secreta)

csv_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRFOHQWNEE_T7RqFY9uAz9wSagCN-khrEV13iqCyQYljXqxSiTAqVYElxVLh2_Ghp3iD7IMurjyvGTJ/pub?gid=621212545&single=true&output=csv'
df = pd.read_csv(csv_url)


model = 'models/gemini-embedding-exp-03-07'

import time

def delayEmbeddings(row):
    print(f"Processando: {row['Nome do Cliente']}")
    time.sleep(10)  # Delay de 10 segundos entre chamadas
    return gerarEmbeddings(row)


def gerarEmbeddings(row):
    texto = f"Cliente: {row['Nome do Cliente']}. Faturamento: {row['Faturamento do Cliente']}. Ramo: {row['Ramo de Atividade']}. Endereço: {row['Endereço do Cliente']}."
    result = generativeai.embed_content(
        model=model,
        content=texto,
        task_type="retrieval_document",
        title=row['Nome do Cliente']
    )
    return result['embedding']

df["Embeddings"] = df.apply(delayEmbeddings, axis=1)

pickle.dump(df, open('datasetEmbedding2025.pkl', 'wb'))

print("Embeddings gerados e salvos com sucesso.")

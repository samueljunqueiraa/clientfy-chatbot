from flask import Flask, jsonify, request
import numpy as np
import google.generativeai as generativeai
from google import genai
from google.genai import types
import pickle
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)
CORS(app)  # Libera o CORS para o front

model = 'models/gemini-embedding-exp-03-07'
modeloEmbeddings = pickle.load(open('datasetEmbedding2025.pkl', 'rb'))
chave_secreta = os.getenv('API_KEY')
generativeai.configure(api_key=chave_secreta)
print("🔑 Chave carregada:", chave_secreta)


def gerarBuscarConsulta(consulta, dataset):
    if "maior faturamento" in consulta.lower():
        cliente = dataset.loc[dataset["Faturamento do Cliente"].idxmax()]

        return (
            f"O {cliente['Nome do Cliente']} ({cliente['Endereço do Cliente']}), "
            f"do ramo {cliente['Ramo de Atividade'].lower()}, é o cliente com maior faturamento mensal, "
            f"totalizando R$ {cliente['Faturamento do Cliente']:,.2f}."
        )

    # consulta normal via embeddings
    embedding_consulta = generativeai.embed_content(
        model=model,
        content=consulta,
        task_type="retrieval_query",
    )

    produtos_escalares = np.dot(
        np.stack(dataset["Embeddings"]),
        embedding_consulta['embedding']
    )

    indice = int(np.argmax(produtos_escalares))
    cliente = dataset.iloc[indice]

    return (
        f"Cliente: {cliente.get('Nome do Cliente', 'N/A')}\n"
        f"Faturamento: {cliente.get('Faturamento do Cliente', 'N/A')}\n"
        f"Ramo de Atividade: {cliente.get('Ramo de Atividade', 'N/A')}\n"
        f"Endereço: {cliente.get('Endereço do Cliente', 'N/A')}"
    )


def melhorarResposta(inputText):
    client = genai.Client(api_key=chave_secreta)
    model = "gemini-1.5-flash"

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=inputText),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_k=32,
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(
                text="""Considere a consulta e resposta, reescreva as sentenças de resposta de uma forma alternativa, não apresente opções de reescrita"""
            ),
        ],
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    return response.text


@app.route("/api", methods=["POST"])
def results():
    auth_key = request.headers.get("Authorization")
    if auth_key != chave_secreta:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json(force=True)
        consulta = data["consulta"]
        print("🟨 Consulta recebida:", consulta)

        resultado = gerarBuscarConsulta(consulta, modeloEmbeddings)
        print("🟩 Resultado bruto:", resultado)

        prompt = f"Consulta: {consulta} Resposta: {resultado}"
        response = melhorarResposta(prompt)

        print("✅ Resposta final:", response)
        return jsonify({"mensagem": response})

    except Exception as e:
        print("❌ Erro ao processar:", str(e))
        return jsonify({"mensagem": "Erro interno no servidor."}), 500


if __name__ == "__main__":
    app.run(debug=True)




import json
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load structured IPC JSON file
json_path = r"C:\Users\raji\Downloads\FIR IPC Section Drafting\structured_ipc (1).json"  # Update with your actual path
with open(json_path, "r", encoding="utf-8") as f:
    ipc_data = json.load(f)

# Convert JSON to DataFrame
df = pd.DataFrame(ipc_data)

# Handle missing or non-text values in 'Description'
df['Description'] = df['Description'].astype(str).fillna("")

# Load Sentence-BERT model (Upgrade to BGE model if needed)
model = SentenceTransformer("BAAI/bge-m3")

# Convert descriptions into embeddings
embeddings = model.encode(df['Description'].tolist(), convert_to_numpy=True)

# Create FAISS index with HNSW for efficient search
dim = embeddings.shape[1]
index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors in HNSW graph
index.add(embeddings)

# Load LegalBERT for question-answering
legal_bert_pipeline = pipeline("question-answering", model="nlpaueb/legal-bert-base-uncased")


def search_ipc(query, top_k=5):
    """Search IPC sections using FAISS similarity."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(df):  # Ensure index is within bounds
            results.append({
                "Section": df.iloc[idx]['Section'],
                "Root": df.iloc[idx]['Root'],
                "Offence": df.iloc[idx]['Offence'],
                "Description": df.iloc[idx]['Description'],
                "Similarity": 1 - distances[0][i]  # Convert L2 distance to similarity score
            })
    return results


def generate_legal_response(query):
    """Generate a structured legal response using LegalBERT."""
    retrieved_sections = search_ipc(query, top_k=5)

    # Format context for LegalBERT
    context = "\n\n".join([
        f"Section {r['Section']}: {r['Offence']}\n{r['Description']}" for r in retrieved_sections
    ])

    qa_input = {
        "question": f"Provide a detailed legal answer based on the following IPC sections: {query}",
        "context": context
    }

    response = legal_bert_pipeline(qa_input)['answer']

    # Format final output with retrieved sections
    final_response = f"**LegalBERT Response:**\n"
    for r in retrieved_sections:
        final_response += f"\n**Section {r['Section']}**: {r['Offence']}\n{r['Description']}\n"
    final_response += f"\n**Answer:** {response}"

    return final_response


# Example usage
query = "What is the punishment for theft?"
response = generate_legal_response(query)
print(response)

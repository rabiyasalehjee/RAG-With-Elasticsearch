import numpy as np
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import tempfile
import subprocess
import os
import warnings
import logging

logging.basicConfig(filename='RAG_LLM.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")

model = SentenceTransformer('all-MiniLM-L6-v2')

# Elasticsearch Docker connection 
es = Elasticsearch([{'scheme': 'http', 'host': 'localhost', 'port': 9200}])

# Elasticsearch index name
index_name = "documents"

# Perform a similarity search in Elasticsearch
def search_most_similar_documents(query, k=5):
    """Retrieve documents from Elasticsearch and calculate cosine similarity in Python."""
    try:
        # Convert query to embedding
        query_embedding = model.encode([query])[0]

        # Perform a match_all query to retrieve all embeddings
        response = es.search(
            index=index_name,
            body={
                "_source": ["content", "embedding"],
                "query": {"match_all": {}},
                "size": 1000  # Retrieve enough documents for similarity comparison
            }
        )

        # Extract embeddings and content
        documents = []
        embeddings = []
        for hit in response['hits']['hits']:
            documents.append(hit['_source']['content'])
            embeddings.append(hit['_source']['embedding'])

        # Convert embeddings to NumPy arrays for cosine similarity calculation
        embeddings = np.array(embeddings)
        query_embedding = np.array(query_embedding)

        # Compute cosine similarity 
        dot_products = np.dot(embeddings, query_embedding)
        query_norm = np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(embeddings, axis=1)
        cosine_similarities = dot_products / (query_norm * doc_norms)

        # Combine documents and their cosine similarities
        results = list(zip(documents, cosine_similarities))

        # Sort by similarity in descending order and return top k results
        results = sorted(results, key=lambda x: x[1], reverse=True)[:k]
        return results

    except Exception as e:
        logging.error(f"Error during Elasticsearch search: {e}")
        return []

def call_llama_model(prompt):
    """Call the Llama model with a given prompt. Modify the command to match your Llama model's setup."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt', encoding='utf-8') as temp_file:
            temp_file.write(prompt)
            temp_file_path = temp_file.name

        command = f'ollama run llama3.1:8b < "{temp_file_path}"'
        response = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if response.returncode != 0:
            error_message = response.stderr.decode('utf-8', errors='replace').strip()
            logging.error(f"Ollama returned an error: {error_message}")
            raise RuntimeError(f"Ollama returned an error: {error_message}")

        return response.stdout.decode('utf-8', errors='replace').strip()

    except Exception as e:
        logging.error(f"Error during Llama call: {e}")
        raise

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def run_rag_application(input_dataset):
    """Main function to run the RAG application."""
    try:
        with open(input_dataset, 'r', encoding='utf-8') as file:
            user_input = file.read()

        # Get the most similar documents and their Elasticsearch scores
        similar_documents = search_most_similar_documents(user_input, k=5)

        if not similar_documents:
            logging.warning("No similar documents found.")
            return
        
        logging.info(f"Search Results from Elasticsearch:")
        for idx, (doc, dist) in enumerate(similar_documents, 1):
            logging.info(f"  - Rank {idx}: {doc} (Score: {dist})")

        # Combine all relevant documents for context
        combined_documents = "\n".join([doc for doc, _ in similar_documents])

        # Define the prompt for the Llama model
        prompt = f"""
        You are an expert in aviation data standards and data cleaning. Based on the provided input dataset and retrieved knowledge, generate a set of rules and standards that should be followed by any flight dataset. The rules should ensure data quality, consistency, and compliance with aviation standards.

        Input Dataset:\n{user_input}

        Retrieved Context:\n{combined_documents}

        Please include only the generated rules in your response.
        """
        response = call_llama_model(prompt)

        logging.info(f"\nLLM Response:\n  - {response}")
        print(f"\nLLM Response:\n  - {response}")

    except Exception as e:
        logging.error(f"Error running RAG application: {e}")

if __name__ == "__main__":
    
    input_dataset = "rag_and_llm.csv"
    run_rag_application(input_dataset)

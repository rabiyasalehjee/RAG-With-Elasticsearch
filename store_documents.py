import numpy as np
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Elasticsearch Docker connection 
es = Elasticsearch([{'scheme': 'http', 'host': 'localhost', 'port': 9200}])

corpus_of_documents = [
    "Remove duplicate entries based on flight number, departure date, and departure time.",
    "Ensure no missing values for critical columns: flight number, departure time, arrival time, departure airport, arrival airport.",
    "Ensure correct data types for all columns (e.g., departure date/time in datetime format, ticket price in float).",
    "Convert all date and time fields to a consistent format (e.g., ISO 8601 YYYY-MM-DD HH:MM:SS).",
    "Identify and remove or correct outliers (e.g., flight delays greater than 24 hours, negative ticket prices).",
    "Standardize categorical values (e.g., airline names, airport codes, seat classes).",
    "Ensure airport codes and airline codes follow proper IATA/ICAO standards.",
    "Ensure flight status is logically consistent with the data (e.g., if status is delayed, a delay time should be present).",
    "Normalize all ticket prices to a single currency if multiple currencies are used.",
    "Ensure arrival time is after departure time, and that flight duration is consistent with these times.",
    "Ensure valid airport and route data (e.g., departure airport and arrival airport should not be identical).",
    "Handle flight cancellations appropriately (e.g., remove or flag records where relevant data is missing or inconsistent).",
    "Convert all time-related data to a single time zone (e.g., UTC or local time for departure airport).",
    "Validate that flight numbers are in the correct alphanumeric format (e.g., airline code + digits).",
    "Validate geospatial data (latitudes and longitudes) to ensure consistency with valid airport locations.",
    "Ensure weather-related delays or cancellations are consistent with the reported data for that date and location."
]

# Create an index in Elasticsearch if it doesn't exist
index_name = "documents"
index_body = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text"
            },
            "embedding": {
                "type": "dense_vector",
                "dims": 384  # For the 'all-MiniLM-L6-v2' model
            }
        }
    }
}

# Create the index if it doesn't already exist
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=index_body)

# Convert documents into embeddings using SentenceTransformer
def get_embeddings(documents):
    """Convert a list of documents into embeddings."""
    return model.encode(documents)

# Store documents and their embeddings into Elasticsearch
def store_documents_in_es(documents, embeddings):
    """Store documents and their embeddings in Elasticsearch."""
    actions = []
    for idx, (doc, emb) in enumerate(zip(documents, embeddings)):
        action = {
            "_op_type": "index",
            "_index": index_name,
            "_id": idx,  # Unique document ID
            "_source": {
                "content": doc,
                "embedding": emb.tolist()  # Store embeddings as a list in JSON format
            }
        }
        actions.append(action)
    
    # Bulk index the documents and embeddings
    bulk(es, actions)

# Run the document storage process
if __name__ == "__main__":
    # Generate embeddings for the corpus of documents
    document_embeddings = get_embeddings(corpus_of_documents)

    # Store the documents and their embeddings into Elasticsearch
    store_documents_in_es(corpus_of_documents, document_embeddings)

    print("Documents and embeddings stored in Elasticsearch.")

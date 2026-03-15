#!/usr/bin/env python3
"""
Simple script to get ChromaDB statistics from localhost:8001
"""

import chromadb
import json


def get_chroma_stats():
    """Fetch and display ChromaDB statistics."""
    try:
        # Connect to ChromaDB HTTP client
        client = chromadb.HttpClient(host="localhost", port=8001)
        
        # Get list of collections
        collections = client.list_collections()
        
        print("=" * 60)
        print("ChromaDB Statistics")
        print("=" * 60)
        print(f"\nHost: localhost:8001")
        print(f"Tenant: {client.tenant}")
        print(f"Database: {client.database}")
        print(f"ChromaDB Version: {client.get_version()}")
        print(f"\nTotal Collections: {len(collections)}\n")
        
        # Iterate through collections and get stats
        total_documents = 0
        for collection in collections:
            collection_name = collection.name
            col = client.get_collection(name=collection_name)
            doc_count = col.count()
            total_documents += doc_count
            
            print(f"Collection: '{collection_name}'")
            print(f"  - Documents: {doc_count}")
            print()
        
        print("=" * 60)
        print(f"Total Documents Across All Collections: {total_documents}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        print("Make sure ChromaDB is running on localhost:8001")
        return False
    
    return True


if __name__ == "__main__":
    get_chroma_stats()

import chromadb

def main():
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="my_collection")

    collection.add(
        ids=["id1", "id2"],
        documents=[
            "Python is a high-level programming language used for web development, data science, and automation",
            "Oranges are a type of citrus fruit that are round, orange in color, and rich in vitamin C",
        ],
    )

    results = collection.query(
        query_texts=["programming languages for building websites"],
        n_results=2,
    )

    print(results)


if __name__ == "__main__":
    main()

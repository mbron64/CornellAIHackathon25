from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

class VectorStoreManager:
    def __init__(self, api_key, dimension=1024):
        self.pc = Pinecone(api_key=api_key)
        self.dimension = dimension

    def create_index(self, index_name):
        if index_name in self.pc.list_indexes().names():
            self.pc.delete_index(index_name)

        return self.pc.create_index(
            name=index_name,
            dimension=self.dimension,
            metric='cosine',
            spec={
                "serverless": {
                    "cloud": "aws",
                    "region": "us-east-1"
                }
            }
        )

    def create_vector_store(self, documents, embeddings, index_name, namespace):
        return PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=index_name,
            namespace=namespace
        ) 
import math

class Chunk:
    """Represents a piece of text with a simulated embedding proxy."""
    def __init__(self, id: int, text: str, embedding_proxy: float):
        self.id = id
        self.text = text
        self.embedding_proxy = embedding_proxy # A simplified 1D representation of an embedding

    def __repr__(self):
        return f"Chunk(ID={self.id}, Proxy={self.embedding_proxy:.2f}, Text='{self.text[:50]}...')"

class TreeNode:
    """Represents a node in the binary chunk tree."""
    def __init__(self, chunk: Chunk = None, pivot_value: float = None):
        self.chunk = chunk  # Stores a chunk if this is a leaf node
        self.pivot_value = pivot_value # Value used to split data in internal nodes
        self.left = None
        self.right = None

    def is_leaf(self) -> bool:
        return self.chunk is not None

class BinaryChunkTree:
    """
    A simplified binary tree structure for organizing text chunks
    to enable faster retrieval based on a numerical proxy.
    """
    def __init__(self):
        self.root = None

    def build_tree(self, chunks: list[Chunk]):
        """
        Initiates the recursive building of the binary chunk tree.
        """
        self.root = self._build_recursive(chunks)

    def _build_recursive(self, chunks: list[Chunk]):
        """Helper for recursive tree building."""
        if not chunks:
            return None
        
        # If only one chunk, it's a leaf node containing the actual data.
        if len(chunks) == 1:
            return TreeNode(chunk=chunks[0])

        # Sort chunks by their embedding_proxy to find a suitable split point.
        # This ensures the tree is ordered and enables efficient binary search-like traversal.
        sorted_chunks = sorted(chunks, key=lambda c: c.embedding_proxy)
        
        mid_idx = len(sorted_chunks) // 2
        
        # The embedding_proxy of the chunk at the median index becomes the pivot for this node.
        # This node will be an internal node, guiding the search by comparing query values.
        node = TreeNode(pivot_value=sorted_chunks[mid_idx].embedding_proxy)
        
        # Recursively build left and right subtrees.
        # Chunks with proxy values less than the pivot go to the left subtree.
        # Chunks with proxy values greater than or equal to the pivot go to the right subtree.
        node.left = self._build_recursive(sorted_chunks[:mid_idx])
        node.right = self._build_recursive(sorted_chunks[mid_idx:])
        
        return node

    def retrieve(self, query_embedding_proxy: float) -> Chunk | None:
        """
        Traverses the tree to find the most relevant chunk based on the query's
        embedding proxy. This simulates faster retrieval by narrowing the search space
        logarithmically, reducing RAG latency.
        """
        current_node = self.root
        print(f"  Starting retrieval for query proxy {query_embedding_proxy:.2f}")
        
        while current_node:
            if current_node.is_leaf():
                print(f"  Reached leaf node. Retrieved chunk ID: {current_node.chunk.id}")
                return current_node.chunk
            
            # This decision-making process at each internal node is the core of
            # how binary chunk trees accelerate retrieval in RAG systems.
            # Instead of scanning all chunks, we quickly narrow down the search path.
            if query_embedding_proxy < current_node.pivot_value:
                print(f"  Query {query_embedding_proxy:.2f} < Pivot {current_node.pivot_value:.2f}. Going LEFT.")
                current_node = current_node.left
            else:
                print(f"  Query {query_embedding_proxy:.2f} >= Pivot {current_node.pivot_value:.2f}. Going RIGHT.")
                current_node = current_node.right
        
        return None # Should not happen if tree is properly built and query is within range

# --- Main execution --- 
if __name__ == "__main__":
    print("--- Building Binary Chunk Tree for RAG Retrieval ---")

    # Sample text chunks with simulated embedding proxy values.
    # In a real RAG system, these proxy values would be derived from actual embeddings
    # of the text chunks and represent their semantic content.
    chunks_data = [
        {"id": 1, "text": "Büyük Dil Modelleri (LLM) yapay zekanın temelidir.", "embedding_proxy": 0.1},
        {"id": 2, "text": "RAG, LLM'lere harici bilgi erişimi sağlar.", "embedding_proxy": 0.3},
        {"id": 3, "text": "Gecikme, RAG sistemlerinde kullanıcı deneyimini etkiler.", "embedding_proxy": 0.5},
        {"id": 4, "text": "Binary chunk ağaçları, veri erişimini hızlandırır.", "embedding_proxy": 0.7},
        {"id": 5, "text": "Ağaç yapısı, arama süresini logaritmik olarak azaltır.", "embedding_proxy": 0.9},
        {"id": 6, "text": "Chunk'lar, metin parçacıklarıdır.", "embedding_proxy": 0.2},
        {"id": 7, "text": "LLM'ler, geniş veri setleri üzerinde eğitilir.", "embedding_proxy": 0.05},
        {"id": 8, "text": "RAG, güncel bilgileri LLM'lere getirir.", "embedding_proxy": 0.4},
        {"id": 9, "text": "Veri erişimi, RAG performansının anahtarıdır.", "embedding_proxy": 0.6},
        {"id": 10, "text": "Ağaç tabanlı indeksleme, verimli arama sağlar.", "embedding_proxy": 0.8},
    ]

    chunks = [Chunk(data["id"], data["text"], data["embedding_proxy"]) for data in chunks_data]

    tree = BinaryChunkTree()
    tree.build_tree(chunks) 

    print("\n--- Simulating RAG Retrieval ---")

    # Simulate user queries with specific semantic intents, represented by proxy values.
    # These query values would typically come from embedding the user's question.
    query_proxy_1 = 0.75 # Query related to "Binary chunk trees" or "efficient search"
    print(f"\nQuery 1: Searching for chunk with proxy value around {query_proxy_1:.2f}")
    retrieved_chunk_1 = tree.retrieve(query_proxy_1)
    if retrieved_chunk_1:
        print(f"  Retrieved Chunk 1 (ID: {retrieved_chunk_1.id}): '{retrieved_chunk_1.text}'")
    else:
        print("  No chunk retrieved for Query 1.")

    query_proxy_2 = 0.32 # Query related to "RAG providing external info"
    print(f"\nQuery 2: Searching for chunk with proxy value around {query_proxy_2:.2f}")
    retrieved_chunk_2 = tree.retrieve(query_proxy_2)
    if retrieved_chunk_2:
        print(f"  Retrieved Chunk 2 (ID: {retrieved_chunk_2.id}): '{retrieved_chunk_2.text}'")
    else:
        print("  No chunk retrieved for Query 2.")

    query_proxy_3 = 0.08 # Query related to "LLM fundamentals"
    print(f"\nQuery 3: Searching for chunk with proxy value around {query_proxy_3:.2f}")
    retrieved_chunk_3 = tree.retrieve(query_proxy_3)
    if retrieved_chunk_3:
        print(f"  Retrieved Chunk 3 (ID: {retrieved_chunk_3.id}): '{retrieved_chunk_3.text}'")
    else:
        print("  No chunk retrieved for Query 3.")

    print("\n--- End of Demonstration ---")

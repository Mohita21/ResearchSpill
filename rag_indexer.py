#!/usr/bin/env python3
"""
RAG Indexer for Research Papers
Implements dual indexing: BGE embeddings in Chroma + BM25 for keyword search
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import torch first to avoid numpy compatibility issues
try:
    import torch
    logger.info(f"PyTorch version: {torch.__version__}")
except ImportError:
    logger.error("torch not installed. Run: pip install torch")
    raise

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
    raise

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    logger.error("chromadb not installed. Run: pip install chromadb")
    raise

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    logger.error("rank-bm25 not installed. Run: pip install rank-bm25")
    raise

from rag_chunker import Chunk, PaperChunker


class RAGIndexer:
    """Dual indexer: Vector embeddings (Chroma) + BM25 keyword search"""
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        persist_directory: str = "./rag_index",
        collection_name: str = "research_papers"
    ):
        """
        Initialize the indexer
        
        Args:
            embedding_model: Sentence transformer model name
            persist_directory: Directory to persist indexes
            collection_name: Name for the Chroma collection
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        self.collection_name = collection_name
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Initialize Chroma client with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_directory / "chroma")
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # BM25 index storage
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_corpus: List[List[str]] = []
        self.bm25_chunk_ids: List[str] = []
        self.chunk_metadata: Dict[str, Dict] = {}
        
        # Load existing BM25 index if available
        self._load_bm25_index()
        
        # Sync BM25 from Chroma if Chroma has data but BM25 doesn't
        self._sync_bm25_from_chroma()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Lowercase and split on non-alphanumeric characters
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _save_bm25_index(self):
        """Save BM25 index to disk"""
        bm25_path = self.persist_directory / "bm25_index.pkl"
        data = {
            'corpus': self.bm25_corpus,
            'chunk_ids': self.bm25_chunk_ids,
            'chunk_metadata': self.chunk_metadata
        }
        with open(bm25_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"BM25 index saved to {bm25_path}")
    
    def _load_bm25_index(self):
        """Load BM25 index from disk if available"""
        bm25_path = self.persist_directory / "bm25_index.pkl"
        if bm25_path.exists():
            with open(bm25_path, 'rb') as f:
                data = pickle.load(f)
            self.bm25_corpus = data['corpus']
            self.bm25_chunk_ids = data['chunk_ids']
            self.chunk_metadata = data.get('chunk_metadata', {})
            if self.bm25_corpus:
                self.bm25_index = BM25Okapi(self.bm25_corpus)
            logger.info(f"Loaded BM25 index with {len(self.bm25_corpus)} documents")
    
    def _sync_bm25_from_chroma(self):
        """
        Sync BM25 index from Chroma data if BM25 is missing/incomplete.
        This handles recovery from interrupted indexing.
        """
        try:
            chroma_count = self.collection.count()
            bm25_count = len(self.bm25_chunk_ids)
            
            if chroma_count > 0 and bm25_count < chroma_count:
                logger.info(f"BM25 index incomplete ({bm25_count}) vs Chroma ({chroma_count}). Syncing...")
                
                # Get all documents from Chroma
                result = self.collection.get(include=['documents', 'metadatas'])
                
                if result and result['ids']:
                    existing_bm25_ids = set(self.bm25_chunk_ids)
                    new_count = 0
                    
                    for i, chunk_id in enumerate(result['ids']):
                        if chunk_id not in existing_bm25_ids:
                            text = result['documents'][i] if result['documents'] else ''
                            metadata = result['metadatas'][i] if result['metadatas'] else {}
                            
                            # Add to BM25
                            tokens = self._tokenize(text)
                            self.bm25_corpus.append(tokens)
                            self.bm25_chunk_ids.append(chunk_id)
                            
                            # Store metadata
                            self.chunk_metadata[chunk_id] = {
                                'text': text,
                                **metadata
                            }
                            new_count += 1
                    
                    if new_count > 0:
                        # Rebuild BM25 index
                        self.bm25_index = BM25Okapi(self.bm25_corpus)
                        self._save_bm25_index()
                        logger.info(f"Synced {new_count} chunks from Chroma to BM25. Total: {len(self.bm25_chunk_ids)}")
                    else:
                        logger.info("BM25 already in sync with Chroma")
        except Exception as e:
            logger.warning(f"Error syncing BM25 from Chroma: {e}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for embedding
            
        Returns:
            List of embedding vectors
        """
        # Use convert_to_tensor=True and handle conversion ourselves
        # This avoids the numpy compatibility issues
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > batch_size,  # Only show for large batches
            convert_to_tensor=True,  # Get PyTorch tensor
            normalize_embeddings=True  # For cosine similarity
        )
        
        # Convert tensor to list (avoids numpy entirely)
        if isinstance(embeddings, torch.Tensor):
            return embeddings.cpu().tolist()
        elif hasattr(embeddings, 'tolist'):
            return embeddings.tolist()
        else:
            return [list(emb) for emb in embeddings]
    
    def get_existing_chunk_ids(self) -> set:
        """Get set of chunk IDs already in the index (from both Chroma and BM25)"""
        existing_ids = set(self.bm25_chunk_ids)
        bm25_count = len(self.bm25_chunk_ids)
        
        # Also check Chroma for any IDs not in BM25 (handles interrupted indexing)
        try:
            count = self.collection.count()
            logger.info(f"Chroma collection count: {count}, BM25 count: {bm25_count}")
            
            if count > 0:
                # Fetch all IDs from Chroma
                # Chroma's get() without filters returns all documents
                result = self.collection.get(include=[])  # Only get IDs, not embeddings/documents
                if result and result['ids']:
                    chroma_ids = set(result['ids'])
                    existing_ids = existing_ids.union(chroma_ids)
                    logger.info(f"Found {len(chroma_ids)} existing chunks in Chroma")
        except Exception as e:
            logger.warning(f"Error checking Chroma for existing IDs: {e}")
        
        logger.info(f"Total existing chunk IDs: {len(existing_ids)}")
        return existing_ids
    
    def index_chunks(self, chunks: List[Chunk], batch_size: int = 100, skip_existing: bool = True):
        """
        Index chunks in both Chroma (vector) and BM25 (keyword)
        
        Args:
            chunks: List of Chunk objects to index
            batch_size: Batch size for indexing
            skip_existing: If True, skip chunks that are already indexed (avoid re-embedding)
        """
        if not chunks:
            logger.warning("No chunks to index")
            return
        
        # Filter out already indexed chunks if skip_existing is True
        if skip_existing:
            existing_ids = self.get_existing_chunk_ids()
            if existing_ids:
                original_count = len(chunks)
                chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
                skipped_count = original_count - len(chunks)
                if skipped_count > 0:
                    logger.info(f"Skipping {skipped_count} chunks that are already indexed")
                
                if not chunks:
                    logger.info("All chunks are already indexed. No new embeddings needed.")
                    return
        
        logger.info(f"Indexing {len(chunks)} new chunks...")
        
        # Process in batches - embed AND save each batch immediately
        # This ensures we don't lose progress if there's a crash
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for batch_num, i in enumerate(range(0, len(chunks), batch_size)):
            batch_end = min(i + batch_size, len(chunks))
            batch_chunks = chunks[i:batch_end]
            
            # Prepare batch data
            batch_texts = [chunk.text for chunk in batch_chunks]
            batch_ids = [chunk.chunk_id for chunk in batch_chunks]
            
            # Prepare metadata for this batch
            batch_metadatas = []
            for chunk in batch_chunks:
                meta = {
                    'paper_id': chunk.paper_id,
                    'title': chunk.title[:500] if chunk.title else '',
                    'authors': ', '.join(chunk.authors[:5]) if chunk.authors else '',
                    'published': chunk.published or '',
                    'source': chunk.source or '',
                    'categories': ', '.join(chunk.categories[:5]) if chunk.categories else '',
                    'section': chunk.section or '',
                    'chunk_index': chunk.chunk_index
                }
                batch_metadatas.append(meta)
                self.chunk_metadata[chunk.chunk_id] = chunk.to_dict()
            
            # Generate embeddings for this batch
            logger.info(f"Batch {batch_num + 1}/{total_batches}: Embedding {len(batch_chunks)} chunks...")
            batch_embeddings = self.embed_texts(batch_texts)
            
            # Save to Chroma immediately
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_texts
            )
            
            # Add to BM25 index
            for chunk in batch_chunks:
                tokens = self._tokenize(chunk.text)
                self.bm25_corpus.append(tokens)
                self.bm25_chunk_ids.append(chunk.chunk_id)
            
            logger.info(f"Batch {batch_num + 1}/{total_batches}: Saved to Chroma and BM25")
        
        # Rebuild BM25 index with all documents
        self.bm25_index = BM25Okapi(self.bm25_corpus)
        
        # Save BM25 index
        self._save_bm25_index()
        
        logger.info(f"Indexing complete. Total indexed: {self.collection.count()}")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the indexed collection"""
        stats = {
            'total_chunks': self.collection.count(),
            'bm25_documents': len(self.bm25_corpus),
            'embedding_model': str(self.embedding_model),
            'embedding_dim': self.embedding_dim,
            'persist_directory': str(self.persist_directory)
        }
        return stats
    
    def clear_index(self):
        """Clear all indexed data"""
        logger.warning("Clearing all indexed data...")
        
        # Clear Chroma
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Clear BM25
        self.bm25_index = None
        self.bm25_corpus = []
        self.bm25_chunk_ids = []
        self.chunk_metadata = {}
        
        # Remove saved BM25 index
        bm25_path = self.persist_directory / "bm25_index.pkl"
        if bm25_path.exists():
            bm25_path.unlink()
        
        logger.info("Index cleared")


def main():
    """Main function to index papers"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Index research papers for RAG')
    parser.add_argument('--papers-dir', type=str, default='papers', help='Directory containing paper files')
    parser.add_argument('--index-dir', type=str, default='./rag_index', help='Directory to save index')
    parser.add_argument('--embedding-model', type=str, default='BAAI/bge-large-en-v1.5', help='Embedding model')
    parser.add_argument('--chunk-size', type=int, default=512, help='Chunk size in characters')
    parser.add_argument('--chunk-overlap', type=int, default=128, help='Chunk overlap in characters')
    parser.add_argument('--clear', action='store_true', help='Clear existing index before indexing')
    
    args = parser.parse_args()
    
    # Initialize chunker
    chunker = PaperChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Chunk papers
    papers_dir = Path(args.papers_dir)
    chunks = chunker.chunk_papers_directory(papers_dir)
    
    if not chunks:
        logger.error("No chunks created. Check papers directory.")
        return
    
    # Initialize indexer
    indexer = RAGIndexer(
        embedding_model=args.embedding_model,
        persist_directory=args.index_dir
    )
    
    # Clear if requested
    if args.clear:
        indexer.clear_index()
    
    # Index chunks
    indexer.index_chunks(chunks)
    
    # Print stats
    stats = indexer.get_collection_stats()
    print(f"\n{'='*80}")
    print("Indexing Statistics:")
    print(f"{'='*80}")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()

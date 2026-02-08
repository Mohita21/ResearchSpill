#!/usr/bin/env python3
"""
RAG Retriever with Hybrid Search
Combines vector search, BM25 keyword search, and metadata filtering
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
    raise

try:
    import chromadb
except ImportError:
    logger.error("chromadb not installed. Run: pip install chromadb")
    raise

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    logger.error("rank-bm25 not installed. Run: pip install rank-bm25")
    raise

from rag_indexer import RAGIndexer


@dataclass
class SearchResult:
    """Represents a search result"""
    chunk_id: str
    text: str
    score: float
    metadata: Dict
    source: str  # 'vector', 'bm25', 'hybrid'
    
    def to_dict(self) -> Dict:
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'score': self.score,
            'metadata': self.metadata,
            'source': self.source
        }


class HybridRetriever:
    """
    Hybrid retriever combining:
    1. Vector search (semantic similarity)
    2. BM25 search (keyword matching)
    3. Metadata filtering
    4. Score fusion (RRF or weighted)
    5. Optional reranking
    """
    
    def __init__(
        self,
        indexer: RAGIndexer,
        reranker_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        vector_weight: float = 0.5,
        bm25_weight: float = 0.3,
        metadata_weight: float = 0.2,
        rrf_k: int = 60
    ):
        """
        Initialize the hybrid retriever
        
        Args:
            indexer: RAGIndexer instance with indexed data
            reranker_model: Cross-encoder model for reranking (None to disable)
            vector_weight: Weight for vector search scores
            bm25_weight: Weight for BM25 scores
            metadata_weight: Weight for metadata match boost
            rrf_k: Constant for Reciprocal Rank Fusion
        """
        self.indexer = indexer
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.metadata_weight = metadata_weight
        self.rrf_k = rrf_k
        
        # Initialize reranker if specified
        self.reranker = None
        if reranker_model:
            logger.info(f"Loading reranker model: {reranker_model}")
            self.reranker = CrossEncoder(reranker_model)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def vector_search(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Perform vector similarity search
        
        Args:
            query: Search query
            top_k: Number of results to return
            metadata_filter: Optional metadata filter (Chroma where clause)
            
        Returns:
            List of SearchResult objects
        """
        # Generate query embedding (embed_texts already returns a list)
        query_embedding = self.indexer.embed_texts([query])[0]
        
        # Build where clause from metadata filter
        where_clause = None
        if metadata_filter:
            where_clause = self._build_chroma_where(metadata_filter)
        
        # Query Chroma
        results = self.indexer.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Convert to SearchResult objects
        search_results = []
        if results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                # Chroma returns distances, convert to similarity scores
                # For cosine distance: similarity = 1 - distance
                distance = results['distances'][0][i] if results['distances'] else 0
                score = 1 - distance
                
                result = SearchResult(
                    chunk_id=chunk_id,
                    text=results['documents'][0][i] if results['documents'] else '',
                    score=score,
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {},
                    source='vector'
                )
                search_results.append(result)
        
        return search_results
    
    def bm25_search(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Perform BM25 keyword search
        
        Args:
            query: Search query
            top_k: Number of results to return
            metadata_filter: Optional metadata filter
            
        Returns:
            List of SearchResult objects
        """
        if not self.indexer.bm25_index:
            logger.warning("BM25 index not available")
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.indexer.bm25_index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1]
        
        search_results = []
        for idx in top_indices:
            if len(search_results) >= top_k:
                break
            
            chunk_id = self.indexer.bm25_chunk_ids[idx]
            score = scores[idx]
            
            # Skip zero scores
            if score <= 0:
                continue
            
            # Get metadata
            metadata = self.indexer.chunk_metadata.get(chunk_id, {})
            
            # Apply metadata filter if provided
            if metadata_filter and not self._matches_filter(metadata, metadata_filter):
                continue
            
            result = SearchResult(
                chunk_id=chunk_id,
                text=metadata.get('text', ''),
                score=score,
                metadata=metadata,
                source='bm25'
            )
            search_results.append(result)
        
        return search_results
    
    def _build_chroma_where(self, metadata_filter: Dict) -> Dict:
        """Convert metadata filter to Chroma where clause"""
        where_clauses = []
        
        for key, value in metadata_filter.items():
            if value is None:
                continue
            
            if isinstance(value, str):
                where_clauses.append({key: {"$eq": value}})
            elif isinstance(value, list):
                # OR condition for list values
                or_clauses = [{key: {"$eq": v}} for v in value]
                if or_clauses:
                    where_clauses.append({"$or": or_clauses})
            elif isinstance(value, dict):
                # Direct operator (e.g., {"$gte": 2024})
                where_clauses.append({key: value})
        
        if not where_clauses:
            return None
        elif len(where_clauses) == 1:
            return where_clauses[0]
        else:
            return {"$and": where_clauses}
    
    def _matches_filter(self, metadata: Dict, metadata_filter: Dict) -> bool:
        """Check if metadata matches filter"""
        for key, value in metadata_filter.items():
            if key not in metadata:
                return False
            
            meta_value = metadata[key]
            
            if isinstance(value, str):
                if value.lower() not in str(meta_value).lower():
                    return False
            elif isinstance(value, list):
                if not any(v.lower() in str(meta_value).lower() for v in value):
                    return False
        
        return True
    
    def reciprocal_rank_fusion(
        self,
        result_lists: List[List[SearchResult]],
        k: int = 60
    ) -> List[SearchResult]:
        """
        Combine multiple result lists using Reciprocal Rank Fusion
        
        Args:
            result_lists: List of result lists from different sources
            k: RRF constant (default 60)
            
        Returns:
            Fused and sorted list of SearchResult objects
        """
        rrf_scores = {}
        result_map = {}
        
        for results in result_lists:
            for rank, result in enumerate(results):
                chunk_id = result.chunk_id
                rrf_score = 1.0 / (k + rank + 1)
                
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0
                    result_map[chunk_id] = result
                
                rrf_scores[chunk_id] += rrf_score
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Create fused results
        fused_results = []
        for chunk_id in sorted_ids:
            result = result_map[chunk_id]
            fused_result = SearchResult(
                chunk_id=chunk_id,
                text=result.text,
                score=rrf_scores[chunk_id],
                metadata=result.metadata,
                source='hybrid'
            )
            fused_results.append(fused_result)
        
        return fused_results
    
    def weighted_fusion(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Combine results using weighted score combination
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            
        Returns:
            Fused and sorted list of SearchResult objects
        """
        combined_scores = {}
        result_map = {}
        
        # Normalize and weight vector scores
        if vector_results:
            max_vector = max(r.score for r in vector_results) or 1
            for result in vector_results:
                chunk_id = result.chunk_id
                normalized_score = result.score / max_vector
                combined_scores[chunk_id] = self.vector_weight * normalized_score
                result_map[chunk_id] = result
        
        # Normalize and weight BM25 scores
        if bm25_results:
            max_bm25 = max(r.score for r in bm25_results) or 1
            for result in bm25_results:
                chunk_id = result.chunk_id
                normalized_score = result.score / max_bm25
                
                if chunk_id in combined_scores:
                    combined_scores[chunk_id] += self.bm25_weight * normalized_score
                else:
                    combined_scores[chunk_id] = self.bm25_weight * normalized_score
                    result_map[chunk_id] = result
        
        # Sort by combined score
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        
        # Create fused results
        fused_results = []
        for chunk_id in sorted_ids:
            result = result_map[chunk_id]
            fused_result = SearchResult(
                chunk_id=chunk_id,
                text=result.text,
                score=combined_scores[chunk_id],
                metadata=result.metadata,
                source='hybrid'
            )
            fused_results.append(fused_result)
        
        return fused_results
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Rerank results using cross-encoder
        
        Args:
            query: Original query
            results: Results to rerank
            top_k: Number of results to return after reranking
            
        Returns:
            Reranked list of SearchResult objects
        """
        if not self.reranker or not results:
            return results[:top_k]
        
        # Prepare pairs for cross-encoder
        pairs = [(query, result.text) for result in results]
        
        # Get reranking scores
        scores = self.reranker.predict(pairs)
        
        # Sort by reranking scores
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Create reranked results
        reranked = []
        for result, score in scored_results[:top_k]:
            reranked_result = SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=float(score),
                metadata=result.metadata,
                source='reranked'
            )
            reranked.append(reranked_result)
        
        return reranked
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        initial_k: int = 20,
        metadata_filter: Optional[Dict] = None,
        use_reranking: bool = True,
        fusion_method: str = 'rrf'  # 'rrf' or 'weighted'
    ) -> List[SearchResult]:
        """
        Perform hybrid search with optional reranking
        
        Args:
            query: Search query
            top_k: Number of final results
            initial_k: Number of results for initial retrieval
            metadata_filter: Optional metadata filter
            use_reranking: Whether to use cross-encoder reranking
            fusion_method: Score fusion method ('rrf' or 'weighted')
            
        Returns:
            List of SearchResult objects
        """
        logger.info(f"Searching for: {query}")
        
        # Perform vector search
        vector_results = self.vector_search(query, top_k=initial_k, metadata_filter=metadata_filter)
        logger.info(f"Vector search returned {len(vector_results)} results")
        
        # Perform BM25 search
        bm25_results = self.bm25_search(query, top_k=initial_k, metadata_filter=metadata_filter)
        logger.info(f"BM25 search returned {len(bm25_results)} results")
        
        # Fuse results
        if fusion_method == 'rrf':
            fused_results = self.reciprocal_rank_fusion([vector_results, bm25_results], k=self.rrf_k)
        else:
            fused_results = self.weighted_fusion(vector_results, bm25_results)
        
        logger.info(f"Fusion returned {len(fused_results)} unique results")
        
        # Rerank if enabled
        if use_reranking and self.reranker:
            results = self.rerank(query, fused_results[:initial_k], top_k=top_k)
            logger.info(f"Reranking returned {len(results)} results")
        else:
            results = fused_results[:top_k]
        
        return results


def main():
    """Test the hybrid retriever"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test hybrid search')
    parser.add_argument('--index-dir', type=str, default='./rag_index', help='Index directory')
    parser.add_argument('--query', type=str, required=True, help='Search query')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    parser.add_argument('--no-rerank', action='store_true', help='Disable reranking')
    
    args = parser.parse_args()
    
    # Initialize indexer (loads existing index)
    indexer = RAGIndexer(persist_directory=args.index_dir)
    
    # Initialize retriever
    retriever = HybridRetriever(indexer)
    
    # Perform search
    results = retriever.search(
        query=args.query,
        top_k=args.top_k,
        use_reranking=not args.no_rerank
    )
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Search Results for: {args.query}")
    print(f"{'='*80}")
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result.score:.4f} | Source: {result.source}")
        print(f"    Title: {result.metadata.get('title', 'N/A')[:60]}...")
        print(f"    Section: {result.metadata.get('section', 'N/A')}")
        print(f"    Text: {result.text[:200]}...")
        print("-" * 40)


if __name__ == '__main__':
    main()

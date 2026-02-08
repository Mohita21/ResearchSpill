#!/usr/bin/env python3
"""
RAG Pipeline for Research Papers
Complete pipeline: indexing, hybrid search, and LLM generation with OpenAI
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Generator
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    logger.error("openai not installed. Run: pip install openai")
    raise

from rag_chunker import PaperChunker
from rag_indexer import RAGIndexer
from rag_retriever import HybridRetriever, SearchResult


class RAGPipeline:
    """
    Complete RAG pipeline for research papers
    - Document loading and chunking
    - Dual indexing (vector + BM25)
    - Hybrid search with reranking
    - LLM generation with OpenAI
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful research assistant specializing in AI and machine learning papers. 
Your task is to answer questions based on the provided research paper excerpts.

Guidelines:
- Base your answers on the provided context
- If the context doesn't contain enough information, say so
- Cite specific papers when making claims
- Use technical language appropriate for the topic
- Be concise but comprehensive"""
    
    def __init__(
        self,
        papers_dir: str = "papers",
        index_dir: str = "./rag_index",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        openai_model: str = "gpt-4o-mini",
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        config_path: Optional[str] = None
    ):
        """
        Initialize the RAG pipeline
        
        Args:
            papers_dir: Directory containing paper .txt files
            index_dir: Directory to store/load index
            embedding_model: Sentence transformer model for embeddings
            reranker_model: Cross-encoder model for reranking
            openai_model: OpenAI model for generation
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            config_path: Optional path to config.yaml
        """
        # Load config if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            papers_dir = config.get('papers_dir', papers_dir)
            index_dir = config.get('index_dir', index_dir)
            embedding_model = config.get('embedding_model', embedding_model)
            reranker_model = config.get('reranker_model', reranker_model)
            openai_model = config.get('openai_model', openai_model)
            chunk_size = config.get('chunk_size', chunk_size)
            chunk_overlap = config.get('chunk_overlap', chunk_overlap)
        
        self.papers_dir = Path(papers_dir)
        self.index_dir = Path(index_dir)
        self.openai_model = openai_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.openai_client = OpenAI(api_key=api_key)
        
        # Initialize chunker
        self.chunker = PaperChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize indexer
        self.indexer = RAGIndexer(
            embedding_model=embedding_model,
            persist_directory=str(self.index_dir)
        )
        
        # Initialize retriever
        self.retriever = HybridRetriever(
            indexer=self.indexer,
            reranker_model=reranker_model
        )
        
        logger.info("RAG Pipeline initialized")
    
    def index_papers(self, clear_existing: bool = False, force_reindex: bool = False):
        """
        Index all papers in the papers directory.
        
        By default, this method performs incremental indexing:
        - Only new papers/chunks are embedded and indexed
        - Existing embeddings are reused (not recalculated)
        
        Args:
            clear_existing: Whether to clear existing index before indexing
            force_reindex: If True, re-embed all chunks even if they exist (slower)
        """
        if not self.papers_dir.exists():
            raise ValueError(f"Papers directory not found: {self.papers_dir}")
        
        if clear_existing:
            self.indexer.clear_index()
        
        # Get current stats
        stats = self.indexer.get_collection_stats()
        existing_count = stats['total_chunks']
        
        if existing_count > 0:
            logger.info(f"Index already contains {existing_count} chunks.")
            if not force_reindex:
                logger.info("Checking for new papers to index (existing embeddings will be reused)...")
        
        # Chunk papers
        logger.info("Chunking papers...")
        chunks = self.chunker.chunk_papers_directory(self.papers_dir)
        
        if not chunks:
            raise ValueError("No chunks created. Check papers directory.")
        
        # Index chunks (will skip existing if not force_reindex)
        logger.info("Indexing chunks (skipping already indexed)...")
        self.indexer.index_chunks(chunks, skip_existing=not force_reindex)
        
        # Get final stats
        final_stats = self.indexer.get_collection_stats()
        new_chunks = final_stats['total_chunks'] - existing_count
        
        if new_chunks > 0:
            logger.info(f"Added {new_chunks} new chunks. Total: {final_stats['total_chunks']}")
        else:
            logger.info(f"No new chunks added. Total indexed: {final_stats['total_chunks']}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict] = None,
        use_reranking: bool = True
    ) -> List[SearchResult]:
        """
        Search for relevant chunks
        
        Args:
            query: Search query
            top_k: Number of results to return
            metadata_filter: Optional metadata filter
            use_reranking: Whether to use reranking
            
        Returns:
            List of SearchResult objects
        """
        return self.retriever.search(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
            use_reranking=use_reranking
        )
    
    def format_context(self, results: List[SearchResult]) -> str:
        """Format search results as context for LLM"""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            title = result.metadata.get('title', 'Unknown Title')
            section = result.metadata.get('section', 'Unknown Section')
            authors = result.metadata.get('authors', 'Unknown Authors')
            
            context_parts.append(f"""
[Source {i}]
Paper: {title}
Authors: {authors}
Section: {section}
---
{result.text}
---
""")
        
        return '\n'.join(context_parts)
    
    def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate response using OpenAI
        
        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional custom system prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated response
        """
        system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
        user_message = f"""Based on the following research paper excerpts, answer the question.

Context from papers:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""
        
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def generate_stream(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Generator[str, None, None]:
        """
        Generate response using OpenAI with streaming
        
        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional custom system prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
            
        Yields:
            Response chunks
        """
        system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
        user_message = f"""Based on the following research paper excerpts, answer the question.

Context from papers:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""
        
        stream = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict] = None,
        use_reranking: bool = True,
        stream: bool = False,
        system_prompt: Optional[str] = None
    ) -> Dict:
        """
        Complete RAG query: search + generate
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            metadata_filter: Optional metadata filter
            use_reranking: Whether to use reranking
            stream: Whether to stream the response
            system_prompt: Optional custom system prompt
            
        Returns:
            Dict with 'answer', 'sources', and optionally 'stream'
        """
        # Search for relevant chunks
        results = self.search(
            query=question,
            top_k=top_k,
            metadata_filter=metadata_filter,
            use_reranking=use_reranking
        )
        
        if not results:
            return {
                'answer': "I couldn't find any relevant information in the papers to answer your question.",
                'sources': []
            }
        
        # Format context
        context = self.format_context(results)
        
        # Generate response
        if stream:
            return {
                'stream': self.generate_stream(question, context, system_prompt),
                'sources': [r.to_dict() for r in results]
            }
        else:
            answer = self.generate(question, context, system_prompt)
            return {
                'answer': answer,
                'sources': [r.to_dict() for r in results]
            }
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            'papers_dir': str(self.papers_dir),
            'index_stats': self.indexer.get_collection_stats(),
            'openai_model': self.openai_model,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }


def main():
    """Interactive CLI for RAG pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Research Paper RAG Pipeline')
    parser.add_argument('--papers-dir', type=str, default='papers', help='Papers directory')
    parser.add_argument('--index-dir', type=str, default='./rag_index', help='Index directory')
    parser.add_argument('--config', type=str, help='Path to config.yaml')
    parser.add_argument('--reindex', action='store_true', help='Clear and rebuild index from scratch')
    parser.add_argument('--force-reembed', action='store_true', help='Force re-embedding all chunks (even existing ones)')
    parser.add_argument('--query', type=str, help='Single query (non-interactive mode)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of chunks to retrieve')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        papers_dir=args.papers_dir,
        index_dir=args.index_dir,
        config_path=args.config
    )
    
    # Index papers if needed (incremental by default - skips existing embeddings)
    pipeline.index_papers(clear_existing=args.reindex, force_reindex=args.force_reembed)
    
    # Print stats
    stats = pipeline.get_stats()
    print(f"\n{'='*80}")
    print("RAG Pipeline Statistics:")
    print(f"{'='*80}")
    print(f"Papers directory: {stats['papers_dir']}")
    print(f"Total indexed chunks: {stats['index_stats']['total_chunks']}")
    print(f"OpenAI model: {stats['openai_model']}")
    print(f"{'='*80}\n")
    
    if args.query:
        # Non-interactive mode
        print(f"Query: {args.query}\n")
        result = pipeline.query(args.query, top_k=args.top_k)
        print(f"Answer:\n{result['answer']}\n")
        print(f"\nSources ({len(result['sources'])}):")
        for i, source in enumerate(result['sources'], 1):
            print(f"  [{i}] {source['metadata'].get('title', 'N/A')[:50]}...")
    else:
        # Interactive mode
        print("Interactive RAG Mode (type 'quit' to exit)\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\nSearching and generating response...\n")
                
                # Stream response
                result = pipeline.query(question, top_k=args.top_k, stream=True)
                
                print("Assistant: ", end="", flush=True)
                for chunk in result['stream']:
                    print(chunk, end="", flush=True)
                print("\n")
                
                # Print sources
                print(f"Sources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'], 1):
                    title = source['metadata'].get('title', 'N/A')[:50]
                    section = source['metadata'].get('section', 'N/A')
                    print(f"  [{i}] {title}... ({section})")
                print()
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


if __name__ == '__main__':
    main()

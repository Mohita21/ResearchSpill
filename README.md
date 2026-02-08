# ResearchSpill

**A production-ready RAG (Retrieval-Augmented Generation) system for AI research papers**

ResearchSpill is a comprehensive end-to-end pipeline designed to help researchers, practitioners, and AI enthusiasts stay up-to-date with the latest developments in Generative AI and Large Language Models. The system automatically discovers, scrapes, indexes, and enables intelligent querying of research papers from prestigious conferences like ICML, NeurIPS, ICLR, and more.

## What is ResearchSpill?

ResearchSpill solves a critical problem in AI research: **information overload**. With hundreds of papers published daily on arXiv, it's nearly impossible to keep track of the latest advances in your field. ResearchSpill provides:

- **Automated Paper Discovery**: Continuously monitors arXiv for relevant papers using curated search queries
- **Intelligent Indexing**: Creates a searchable knowledge base using state-of-the-art embedding models
- **Semantic Search**: Find papers by meaning, not just keywords
- **AI-Powered Answers**: Get comprehensive answers to your questions, backed by citations from actual papers

Whether you're a researcher looking for related work, a practitioner seeking implementation details, or a student learning about new techniques, ResearchSpill makes it easy to find and understand the information you need.

---

## Key Features

### Intelligent Paper Scraping

The paper scraper (`paper_scraper.py`) automates the tedious process of finding and downloading relevant research papers:

- **Automated Discovery**: Uses carefully curated keyword lists to search arXiv for papers on Generative AI, LLMs, transformers, diffusion models, and related topics
- **Conference Filtering**: Prioritizes papers from top-tier venues (ICML, NeurIPS, ICLR, AAAI, ACL, EMNLP, etc.) to ensure quality
- **Full-Text Extraction**: Downloads PDFs and extracts complete text content while preserving metadata (title, authors, abstract, publication date)
- **Robust Error Handling**: Gracefully handles network issues, corrupted PDFs, and rate limits to ensure reliable operation

### Advanced RAG System

The RAG system implements state-of-the-art retrieval and generation techniques:

- **Hybrid Chunking**: Combines structural chunking (by academic sections like Introduction, Methods, Results) with semantic chunking using configurable overlap to maintain context
- **Dual Indexing**: Maintains both dense vector embeddings (using BGE-large-en-v1.5) and sparse BM25 keyword index for comprehensive search coverage
- **Hybrid Search**: Intelligently combines semantic similarity search, keyword matching, and metadata filtering to find the most relevant passages
- **Score Fusion**: Supports both Reciprocal Rank Fusion (RRF) and weighted combination strategies to merge results from multiple retrieval methods
- **Cross-Encoder Reranking**: Uses a cross-encoder model to rerank top candidates for maximum precision
- **LLM Generation**: Integrates with OpenAI GPT models with streaming support to generate comprehensive, citation-backed answers

### Why Hybrid Search?

Traditional keyword search (BM25) is great at finding exact matches but misses semantic relationships. Vector search excels at semantic similarity but can miss important keyword matches. ResearchSpill's hybrid approach combines the best of both worlds, ensuring you find relevant papers whether you search by exact terminology or conceptual meaning.

---

## How It Works

ResearchSpill operates in three main phases:

### Phase 1: Data Collection
The paper scraper monitors arXiv for relevant papers, downloads PDFs, and extracts text content. This creates a local repository of research papers in a structured format.

### Phase 2: Indexing
The chunker breaks papers into semantically meaningful pieces, and the indexer creates both vector embeddings and BM25 indexes. This dual-index approach enables comprehensive search.

### Phase 3: Query & Generation
When you ask a question, the retriever finds relevant chunks using hybrid search, and the pipeline generates a comprehensive answer using an LLM, complete with source citations.

## Use Cases

ResearchSpill is designed for various research and learning scenarios:

- **Literature Review**: Quickly find papers related to your research topic
- **Staying Current**: Keep up with the latest developments in AI/ML
- **Learning New Concepts**: Get explanations of complex topics with paper citations
- **Implementation Research**: Find details about how specific techniques are implemented
- **Comparative Analysis**: Compare different approaches across multiple papers
- **Citation Discovery**: Find papers that discuss specific methods or concepts

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ResearchSpill Architecture                         │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │   arXiv API  │
                              └──────┬───────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         1. PAPER SCRAPER                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Search    │───▶│  Download   │───▶│   Extract   │───▶│    Save     │  │
│  │   Papers    │    │    PDFs     │    │    Text     │    │   .txt      │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                            ┌────────────────┐
                            │  papers/*.txt  │
                            └────────┬───────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         2. CHUNKER                                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Parse     │───▶│  Detect     │───▶│   Split     │───▶│  Preserve   │  │
│  │  Metadata   │    │  Sections   │    │   Chunks    │    │  Metadata   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                            ┌────────────────┐
                            │ Chunk Objects  │
                            │  with Metadata │
                            └────────┬───────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         3. INDEXER (Dual Index)                              │
│                                                                              │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │      Vector Index           │    │         BM25 Index                  │ │
│  │  ┌─────────────────────┐    │    │  ┌─────────────────────────────┐    │ │
│  │  │  BGE-large-en-v1.5  │    │    │  │    Tokenized Corpus         │    │ │
│  │  │  (1024-dim vectors) │    │    │  │    (Okapi BM25)             │    │ │
│  │  └──────────┬──────────┘    │    │  └──────────────┬──────────────┘    │ │
│  │             │               │    │                 │                   │ │
│  │             ▼               │    │                 ▼                   │ │
│  │  ┌─────────────────────┐    │    │  ┌─────────────────────────────┐    │ │
│  │  │   ChromaDB (HNSW)   │    │    │  │   Pickle Serialization      │    │ │
│  │  │   Cosine Similarity │    │    │  │   + Chunk Metadata Store    │    │ │
│  │  └─────────────────────┘    │    │  └─────────────────────────────┘    │ │
│  └─────────────────────────────┘    └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                            ┌────────────────┐
                            │   rag_index/   │
                            │  (Persistent)  │
                            └────────┬───────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         4. RETRIEVER (Hybrid Search)                         │
│                                                                              │
│  ┌───────────────┐                              ┌───────────────────────┐   │
│  │  User Query   │──────────────────────────────│   Metadata Filter     │   │
│  └───────┬───────┘                              │   (Optional)          │   │
│          │                                      └───────────┬───────────┘   │
│          ▼                                                  │               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Parallel Search                                  │  │
│  │  ┌─────────────────────┐         ┌─────────────────────┐             │  │
│  │  │   Vector Search     │         │    BM25 Search      │             │  │
│  │  │   (Semantic)        │         │    (Keyword)        │             │  │
│  │  │   top-k=20          │         │    top-k=20         │             │  │
│  │  └──────────┬──────────┘         └──────────┬──────────┘             │  │
│  └─────────────┼────────────────────────────────┼────────────────────────┘  │
│                │                                │                           │
│                ▼                                ▼                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Score Fusion                                      │   │
│  │  ┌────────────────────────┐    ┌────────────────────────┐           │   │
│  │  │  Reciprocal Rank       │ OR │  Weighted Combination  │           │   │
│  │  │  Fusion (RRF)          │    │  (Normalized Scores)   │           │   │
│  │  │  score = Σ 1/(k+rank)  │    │  score = w₁·v + w₂·b   │           │   │
│  │  └────────────────────────┘    └────────────────────────┘           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Cross-Encoder Reranking                           │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │  ms-marco-MiniLM-L-6-v2                                     │    │   │
│  │  │  Scores (query, passage) pairs for precision                │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                           ┌─────────────────┐
                           │  Top-K Results  │
                           │  with Scores    │
                           └────────┬────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         5. PIPELINE (Generation)                             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Context Formatting                                │    │
│  │  [Source 1] Paper: {title}, Section: {section}                      │    │
│  │  ---                                                                 │    │
│  │  {chunk_text}                                                        │    │
│  │  ---                                                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    OpenAI GPT Generation                             │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  System: Research assistant prompt                          │    │    │
│  │  │  User: Context + Question                                   │    │    │
│  │  │  Model: gpt-4o-mini (configurable)                          │    │    │
│  │  │  Streaming: Supported                                       │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Response                                          │    │
│  │  { answer: "...", sources: [{chunk_id, text, score, metadata}] }    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
ResearchSpill/
├── paper_scraper.py      # arXiv paper discovery and PDF extraction
├── rag_chunker.py        # Structural + semantic document chunking
├── rag_indexer.py        # Dual indexing (Vector + BM25)
├── rag_retriever.py      # Hybrid search with reranking
├── rag_pipeline.py       # End-to-end RAG orchestration
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
├── papers/               # Scraped paper text files (100 papers)
│   └── *.txt
└── rag_index/            # Persistent index storage
    ├── bm25_index.pkl    # Serialized BM25 index
    └── chroma/           # ChromaDB vector database
        └── ...
```

---

## Component Deep Dive

### 1. Paper Scraper (`paper_scraper.py`)

The paper scraper is your automated research assistant that continuously monitors arXiv for relevant papers.

**How it works:**

1. **Query Construction**: Builds multiple search queries combining keywords like "large language model", "generative ai", "transformer", etc.
2. **arXiv Search**: Uses the official arXiv API to search for papers matching the queries
3. **Quality Filtering**: Prioritizes papers that mention prestigious conferences (ICML, NeurIPS, ICLR, etc.)
4. **PDF Download**: Downloads PDFs with proper rate limiting and error handling
5. **Text Extraction**: Uses `pdfplumber` to extract text from PDFs, handling various formats
6. **Metadata Preservation**: Saves structured metadata (title, authors, date, categories, abstract) alongside the full text

| Feature | Description |
|---------|-------------|
| **Search Strategy** | Multiple query patterns targeting GenAI/LLM terminology |
| **Conference Detection** | Filters for papers mentioning ICML, NeurIPS, ICLR, etc. |
| **PDF Processing** | Uses `pdfplumber` for robust text extraction |
| **Metadata Preservation** | Saves title, authors, date, categories, abstract |
| **Rate Limiting** | Respectful API usage with configurable delays |

**Keyword Coverage:**
- Generative AI, LLMs, GPT, Claude, Gemini
- Transformers, Attention Mechanisms
- Foundation Models, Multimodal AI
- Diffusion Models, RAG, Prompt Engineering
- Fine-tuning, RLHF, Instruction Tuning

**Output Format**: Each paper is saved as a `.txt` file with metadata header followed by the full paper content.

### 2. Chunker (`rag_chunker.py`)

The chunker breaks papers into semantically meaningful pieces while preserving context.

**Why Chunking Matters:**

Research papers are too long to process as a single unit (often 10,000+ words). Chunking breaks them into digestible pieces that can be:
- Embedded efficiently (embedding models have token limits)
- Retrieved precisely (return only relevant sections, not entire papers)
- Understood contextually (each chunk maintains enough context to be meaningful)

**Academic-Aware Chunking:**

Unlike generic text splitters, ResearchSpill understands academic paper structure:

```python
# Section detection patterns
SECTION_PATTERNS = [
    'Abstract', 'Introduction', 'Related Work',
    'Methods', 'Experiments', 'Results',
    'Discussion', 'Conclusion', 'References'
]
```

| Setting | Default | Description |
|---------|---------|-------------|
| `chunk_size` | 512 chars | Target chunk size |
| `chunk_overlap` | 128 chars | Overlap for context continuity |
| `min_chunk_size` | 100 chars | Minimum chunk threshold |
| `respect_sentence_boundaries` | True | Avoid mid-sentence splits |

**Chunking Strategy:**
1. **Parse Metadata**: Extracts title, authors, abstract, etc. from the paper header
2. **Detect Sections**: Identifies academic sections (Introduction, Methods, Results, etc.)
3. **Smart Splitting**: Splits sections into overlapping chunks, respecting sentence boundaries
4. **Metadata Preservation**: Each chunk carries metadata (paper title, authors, section name)
5. **Abstract Handling**: Creates a dedicated chunk for the abstract (high-value content)

**Why Overlap?** The 128-character overlap ensures that information at chunk boundaries isn't lost. For example, if a key concept is explained across a boundary, both chunks will contain enough context to understand it.

### 3. Indexer (`rag_indexer.py`)

The indexer creates a dual-index architecture that enables both semantic and keyword-based search.

**Why Dual Indexing?**

Different search methods excel at different tasks:
- **Vector Search**: Great for finding semantically similar content ("attention mechanism" matches "self-attention", "multi-head attention")
- **BM25 Search**: Excellent for exact terminology ("BERT" won't match "transformer" unless explicitly mentioned)

By maintaining both indexes, ResearchSpill ensures comprehensive search coverage.

#### Vector Index (Semantic Search)

- **Model**: `BAAI/bge-large-en-v1.5` - A state-of-the-art embedding model that converts text to 1024-dimensional vectors
- **Storage**: ChromaDB with HNSW (Hierarchical Navigable Small World) index for fast approximate nearest neighbor search
- **Similarity**: Cosine distance (measures angle between vectors, not magnitude)
- **Features**: 
  - Incremental indexing (only embeds new chunks)
  - Automatic crash recovery
  - Persistent storage (survives restarts)

#### BM25 Index (Keyword Search)

- **Algorithm**: Okapi BM25 - A probabilistic ranking function based on term frequency and document length
- **Storage**: Pickle serialization for fast loading
- **Features**: 
  - Automatic sync with ChromaDB on startup
  - Handles tokenization and normalization
  - Efficient scoring for keyword queries

**Incremental Indexing:**

```python
# Only new chunks are embedded - saves time and compute
indexer.index_chunks(chunks, skip_existing=True)
```

ResearchSpill is smart about indexing. If you add new papers, it only embeds the new content, reusing existing embeddings. This makes updates fast and efficient.

**Crash Recovery:**

```python
# Automatic recovery from interrupted indexing
def _sync_bm25_from_chroma(self):
    """Sync BM25 index from Chroma if incomplete"""
    ...
```

If indexing is interrupted (power loss, crash, etc.), ResearchSpill automatically recovers by syncing the BM25 index from ChromaDB on next startup.

### 4. Retriever (`rag_retriever.py`)

The retriever implements a sophisticated hybrid search pipeline that combines multiple retrieval strategies.

#### Search Pipeline

```
Query → [Vector Search] ──┐
                          ├──→ [Score Fusion] → [Reranking] → Results
Query → [BM25 Search]  ───┘
```

**How it works:**

1. **Parallel Search**: The query is sent to both vector and BM25 search simultaneously
2. **Score Fusion**: Results from both methods are combined using either RRF or weighted combination
3. **Reranking**: A cross-encoder model reranks the top candidates for maximum precision
4. **Final Results**: Returns the top-k most relevant chunks with scores and metadata

#### Score Fusion Methods

**Reciprocal Rank Fusion (RRF):**

```
score(d) = Σ 1/(k + rank(d))
```

- **How it works**: Combines rankings rather than raw scores
- **Advantage**: Robust to score scale differences between methods
- **When to use**: Default choice for most use cases
- **Parameter k**: Default 60 (higher values reduce emphasis on rank position)

**Weighted Combination:**

```
score(d) = w_vector × norm(vector_score) + w_bm25 × norm(bm25_score)
```

- **How it works**: Normalizes scores from each method and combines with weights
- **Advantage**: More control over individual search methods
- **When to use**: When you want to emphasize one method over another
- **Default weights**: vector=0.5, bm25=0.3, metadata=0.2

#### Reranking: The Secret Weapon

**Why Reranking?**

Initial retrieval methods (vector + BM25) are fast but not perfect. They rank based on:
- Vector search: Similarity in embedding space
- BM25: Term frequency and document length

Reranking uses a more sophisticated model that:
- Reads both the query AND the passage together
- Understands their interaction and relevance
- Provides much more accurate relevance scores

**Trade-off**: Reranking is slower (processes each query-passage pair), so we only rerank the top candidates.

- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Purpose**: Precision-focused scoring of (query, passage) pairs
- **When**: Applied to top 20 candidates after fusion, returns top 5-10
- **Impact**: Typically improves relevance by 10-20% compared to fusion alone

### 5. Pipeline (`rag_pipeline.py`)

The pipeline orchestrates the complete RAG workflow, tying all components together.

**End-to-End Query Flow:**

```python
# Complete query flow
result = pipeline.query(
    question="What are the latest advances in LLM fine-tuning?",
    top_k=5,
    use_reranking=True,
    stream=True
)
```

**What happens under the hood:**

1. **Query Processing**: Your question is prepared for search
2. **Hybrid Retrieval**: Both vector and BM25 search run in parallel
3. **Score Fusion**: Results are combined using RRF or weighted combination
4. **Reranking**: Cross-encoder reranks top candidates (if enabled)
5. **Context Formatting**: Retrieved chunks are formatted with citations
6. **LLM Generation**: OpenAI GPT generates a comprehensive answer
7. **Response**: Returns answer with source citations

#### Generation Features

- **Model**: OpenAI GPT-4o-mini (configurable - can use GPT-4, GPT-3.5-turbo, etc.)
- **Context Formatting**: Each source is clearly labeled with paper title, authors, and section
- **Streaming**: Real-time token-by-token output for better UX
- **System Prompt**: Configured as a research assistant that cites sources and admits uncertainty
- **Citation Tracking**: Every answer includes references to the source chunks

**Context Window Management:**

The pipeline intelligently manages the context window:
- Retrieves top-k most relevant chunks (default: 5)
- Formats each with metadata for proper citation
- Ensures total context fits within model limits
- Prioritizes most relevant information

**Example Output:**

```
Answer: Recent advances in LLM fine-tuning include parameter-efficient methods 
like LoRA and QLoRA, which reduce memory requirements by 90%...

Sources:
[1] "Efficient Fine-Tuning of Large Language Models" (Methods section)
[2] "QLoRA: Quantized Low-Rank Adaptation" (Introduction)
[3] "Parameter-Efficient Transfer Learning" (Results)
```

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for answer generation)
- 4GB+ RAM recommended (for embedding models)
- Internet connection (for downloading papers and models)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ResearchSpill.git
cd ResearchSpill

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY=your-api-key-here
```

### Step 1: Scrape Papers

First, download research papers from arXiv:

```bash
# Default: 50 papers from last 30 days
python paper_scraper.py

# Custom options
python paper_scraper.py --max-papers 100 --days-back 60 --output-dir my_papers
```

This will create a `papers/` directory with extracted text files. Each file contains the paper's metadata and full content.

### Step 2: Index Papers

Next, create a searchable index of the papers:

```bash
# Index with incremental updates (reuses existing embeddings - FAST)
python rag_pipeline.py

# Force complete reindex (clears everything and starts fresh)
python rag_pipeline.py --reindex

# Force re-embedding all chunks (recalculates embeddings - SLOW)
python rag_pipeline.py --force-reembed
```

**Note**: The first indexing run will take several minutes as it generates embeddings for all chunks. Subsequent runs are much faster as they skip already-indexed content.

### Step 3: Query Papers

Now you can ask questions about the papers:

```bash
# Interactive mode (recommended for exploration)
python rag_pipeline.py

# Single query (useful for scripting)
python rag_pipeline.py --query "How does RLHF work?" --top-k 10

# With custom config
python rag_pipeline.py --config config.yaml --query "Explain transformers"
```

### Example Usage

```bash
$ python rag_pipeline.py

Interactive RAG Mode (type 'quit' to exit)

You: What are the latest techniques for fine-tuning large language models?

Searching and generating response...

---

## Configuration

ResearchSpill is highly configurable. Edit `config.yaml` to customize the pipeline behavior:

```yaml
# Directories
papers_dir: "papers"           # Where scraped papers are stored
index_dir: "./rag_index"       # Where the search index is persisted

# Models
embedding_model: "BAAI/bge-large-en-v1.5"              # Dense embedding model
reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2" # Reranking model
openai_model: "gpt-4o-mini"                            # LLM for answer generation

# Chunking
chunk_size: 512          # Target size for each chunk (characters)
chunk_overlap: 128       # Overlap between chunks to maintain context
min_chunk_size: 100      # Minimum chunk size to keep

# Retrieval
initial_retrieval_k: 20  # Number of candidates from each search method
final_top_k: 5           # Number of results after reranking

# Hybrid search weights (for weighted fusion)
vector_weight: 0.5       # Weight for semantic similarity scores
bm25_weight: 0.3         # Weight for keyword matching scores
metadata_weight: 0.2     # Weight for metadata match boost

# Score fusion
fusion_method: "rrf"     # "rrf" (Reciprocal Rank Fusion) or "weighted"
rrf_k: 60                # Constant for RRF (higher = less emphasis on rank)

# Generation
temperature: 0.7         # Higher = more creative, lower = more focused
max_tokens: 1024         # Maximum length of generated answers
```

### Configuration Tips

- **chunk_size**: Smaller chunks (256-512) work better for precise questions. Larger chunks (512-1024) are better for broader context.
- **fusion_method**: Use "rrf" for balanced results, "weighted" for more control over individual search methods.
- **initial_retrieval_k**: Increase for better recall, decrease for faster queries.
- **temperature**: Use 0.3-0.5 for factual answers, 0.7-0.9 for more creative responses.

---

## Performance Characteristics

Understanding the system's performance helps set appropriate expectations:

| Metric | Value | Notes |
|--------|-------|-------|
| Papers Indexed | 100 (configurable) | Can scale to thousands |
| Chunks per Paper | ~20-50 | Varies by paper length and chunk_size |
| Embedding Dimension | 1024 | BGE-large-en-v1.5 model |
| Index Size | ~50-100 MB | For 100 papers (~3000 chunks) |
| Query Latency | ~2-5 seconds | With reranking; <1s without |
| Embedding Batch Size | 32 | Configurable based on GPU/RAM |
| Indexing Speed | ~2-5 min | First run for 100 papers |
| Incremental Update | <30 seconds | For 10 new papers |

### Scalability

ResearchSpill is designed to scale:
- **Small collections** (10-100 papers): Works great on laptops
- **Medium collections** (100-1000 papers): Recommended for most use cases
- **Large collections** (1000+ papers): May require more RAM and disk space

### Hardware Requirements

- **Minimum**: 4GB RAM, 2GB disk space
- **Recommended**: 8GB RAM, 5GB disk space, GPU (optional, speeds up embedding)
- **Optimal**: 16GB RAM, 10GB disk space, GPU with 4GB+ VRAM

---

## Advanced Usage

### Standalone Component Usage

```python
# Use chunker independently
from rag_chunker import PaperChunker
chunker = PaperChunker(chunk_size=512, chunk_overlap=128)
chunks = chunker.chunk_papers_directory(Path("papers"))

# Use indexer independently
from rag_indexer import RAGIndexer
indexer = RAGIndexer(persist_directory="./my_index")
indexer.index_chunks(chunks)

# Use retriever independently
from rag_retriever import HybridRetriever
retriever = HybridRetriever(indexer)
results = retriever.search("attention mechanisms", top_k=10)
```

### Metadata Filtering

```python
# Filter by paper or section
results = pipeline.search(
    query="attention mechanisms",
    metadata_filter={
        "section": "Methods",
        "categories": ["cs.CL", "cs.LG"]
    }
)
```

### Custom System Prompts

```python
custom_prompt = """You are a technical reviewer analyzing ML papers.
Focus on methodology strengths and weaknesses."""

result = pipeline.query(
    question="Analyze the attention mechanism approaches",
    system_prompt=custom_prompt
)
```

---

## Troubleshooting

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `OPENAI_API_KEY not set` | Environment variable not configured | Run `export OPENAI_API_KEY=your-key` in terminal |
| PDF extraction fails | Some PDFs are scanned images or corrupted | This is expected for some papers; the scraper will skip them |
| Slow first indexing | Generating embeddings for all chunks | Normal behavior; subsequent runs are much faster (incremental) |
| BM25 out of sync | Interrupted indexing or corruption | Delete `bm25_index.pkl` and rerun - it will auto-sync from ChromaDB |
| ChromaDB errors | Database corruption or version mismatch | Delete `rag_index/chroma/` directory and reindex |
| Out of memory | Embedding model too large for system | Use a smaller model like `all-MiniLM-L6-v2` in config.yaml |
| Slow queries | Reranking enabled with large initial_k | Reduce `initial_retrieval_k` or disable reranking |
| Poor search results | Index outdated or incomplete | Run with `--reindex` flag to rebuild from scratch |

### Performance Tips

- **First-time indexing**: Expect 2-5 minutes for 100 papers (depends on hardware)
- **Incremental updates**: Only new papers are embedded, taking seconds
- **Query latency**: 2-5 seconds with reranking, <1 second without
- **Memory usage**: ~2-4GB for embedding model, ~500MB for index
- **Disk space**: ~50-100MB for index, ~1MB per paper

### Getting Help

If you encounter issues not covered here:
1. Check the logs for detailed error messages
2. Ensure all dependencies are installed (`pip install -r requirements.txt`)
3. Verify your Python version is 3.8 or higher
4. Try with a fresh virtual environment
5. Open an issue on GitHub with error details

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `arxiv` | arXiv API client |
| `pdfplumber` | PDF text extraction |
| `sentence-transformers` | Embedding & reranking models |
| `chromadb` | Vector database |
| `rank-bm25` | BM25 implementation |
| `openai` | GPT generation |
| `torch` | PyTorch backend |
| `PyYAML` | Configuration parsing |

---

## License

MIT License - See LICENSE file for details.

---

## Frequently Asked Questions

### Q: Do I need a GPU?
**A:** No, but it helps. The embedding model runs faster on GPU, but CPU-only operation works fine for small to medium collections.

### Q: How much does it cost to use?
**A:** The only cost is OpenAI API usage for answer generation. Typical queries cost $0.001-0.01 depending on context length and model. All other components (embedding, indexing, search) are free and run locally.

### Q: Can I use a different LLM instead of OpenAI?
**A:** Yes! The pipeline is designed to be modular. You can modify `rag_pipeline.py` to use any LLM (Anthropic Claude, local models via Ollama, etc.).

### Q: How often should I update the index?
**A:** Depends on your needs. Run the scraper weekly or monthly to stay current. The incremental indexing makes updates fast.

### Q: Can I search papers from specific conferences only?
**A:** Yes! Use metadata filtering in queries. The scraper already tags papers with conference information when available.

### Q: What if I want to index my own papers (not from arXiv)?
**A:** Place your papers as `.txt` files in the `papers/` directory with the same metadata format, and run the indexer.

### Q: Is my data private?
**A:** Yes! All indexing and search happens locally. Only your queries and retrieved context are sent to OpenAI for answer generation.

### Q: Can I use this for non-AI papers?
**A:** Absolutely! Just modify the search keywords in `paper_scraper.py` to target your domain of interest.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<div align="center">
  <b>Built for the AI research community</b>
</div>

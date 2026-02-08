#!/usr/bin/env python3
"""
RAG Chunker for Research Papers
Implements hybrid chunking (structural + semantic) with metadata preservation
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a chunk of text with metadata"""
    text: str
    chunk_id: str
    paper_id: str
    title: str
    authors: List[str]
    published: str
    source: str
    categories: List[str]
    pdf_url: str
    section: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'chunk_id': self.chunk_id,
            'paper_id': self.paper_id,
            'title': self.title,
            'authors': self.authors,
            'published': self.published,
            'source': self.source,
            'categories': self.categories,
            'pdf_url': self.pdf_url,
            'section': self.section,
            'chunk_index': self.chunk_index,
            'start_char': self.start_char,
            'end_char': self.end_char,
            **self.metadata
        }


class PaperChunker:
    """Hybrid chunker for academic papers: structural + semantic chunking"""
    
    # Section patterns for academic papers
    SECTION_PATTERNS = [
        (r'^(?:Abstract|ABSTRACT)\s*[:\n]?', 'Abstract'),
        (r'^(?:\d+\.?\s*)?(?:Introduction|INTRODUCTION)\s*[:\n]?', 'Introduction'),
        (r'^(?:\d+\.?\s*)?(?:Related\s+Work|RELATED\s+WORK|Background|BACKGROUND|Literature\s+Review)\s*[:\n]?', 'Related Work'),
        (r'^(?:\d+\.?\s*)?(?:Method(?:s|ology)?|METHOD(?:S|OLOGY)?|Approach|APPROACH)\s*[:\n]?', 'Methods'),
        (r'^(?:\d+\.?\s*)?(?:Experiment(?:s)?|EXPERIMENT(?:S)?|Evaluation|EVALUATION)\s*[:\n]?', 'Experiments'),
        (r'^(?:\d+\.?\s*)?(?:Result(?:s)?|RESULT(?:S)?|Findings|FINDINGS)\s*[:\n]?', 'Results'),
        (r'^(?:\d+\.?\s*)?(?:Discussion|DISCUSSION)\s*[:\n]?', 'Discussion'),
        (r'^(?:\d+\.?\s*)?(?:Conclusion(?:s)?|CONCLUSION(?:S)?|Summary|SUMMARY)\s*[:\n]?', 'Conclusion'),
        (r'^(?:Reference(?:s)?|REFERENCE(?:S)?|Bibliography|BIBLIOGRAPHY)\s*[:\n]?', 'References'),
        (r'^(?:Appendix|APPENDIX|Appendices|APPENDICES)\s*[:\n]?', 'Appendix'),
        (r'^(?:Acknowledgement(?:s)?|ACKNOWLEDGEMENT(?:S)?)\s*[:\n]?', 'Acknowledgements'),
    ]
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_size: int = 100,
        respect_sentence_boundaries: bool = True
    ):
        """
        Initialize the chunker
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size to keep
            respect_sentence_boundaries: Whether to split at sentence boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.respect_sentence_boundaries = respect_sentence_boundaries
    
    def parse_paper_metadata(self, content: str) -> Dict:
        """Extract metadata from paper header"""
        metadata = {
            'title': '',
            'authors': [],
            'published': '',
            'source': '',
            'categories': [],
            'pdf_url': '',
            'abstract': ''
        }
        
        lines = content.split('\n')
        
        for line in lines[:20]:  # Check first 20 lines for metadata
            if line.startswith('Title:'):
                metadata['title'] = line.replace('Title:', '').strip()
            elif line.startswith('Authors:'):
                authors_str = line.replace('Authors:', '').strip()
                metadata['authors'] = [a.strip() for a in authors_str.split(',')]
            elif line.startswith('Published:'):
                metadata['published'] = line.replace('Published:', '').strip()
            elif line.startswith('Source:'):
                metadata['source'] = line.replace('Source:', '').strip()
            elif line.startswith('Categories:'):
                cats_str = line.replace('Categories:', '').strip()
                metadata['categories'] = [c.strip() for c in cats_str.split(',')]
            elif line.startswith('PDF URL:'):
                metadata['pdf_url'] = line.replace('PDF URL:', '').strip()
        
        # Extract abstract
        abstract_match = re.search(
            r'Abstract:\s*(.*?)(?=\n={10,}|$)',
            content,
            re.DOTALL | re.IGNORECASE
        )
        if abstract_match:
            metadata['abstract'] = abstract_match.group(1).strip()
        
        return metadata
    
    def extract_paper_content(self, content: str) -> str:
        """Extract main paper content (after metadata header)"""
        # Find the separator line and extract content after it
        separator_match = re.search(r'={10,}\nFull Paper Content:\n={10,}\n', content)
        if separator_match:
            return content[separator_match.end():].strip()
        return content
    
    def detect_sections(self, content: str) -> List[Tuple[str, int, int]]:
        """Detect section boundaries in the paper"""
        sections = []
        lines = content.split('\n')
        current_pos = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                current_pos += len(line) + 1
                continue
            
            for pattern, section_name in self.SECTION_PATTERNS:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    sections.append((section_name, current_pos, i))
                    break
            
            current_pos += len(line) + 1
        
        return sections
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - handles common cases
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(
        self,
        text: str,
        section: str = "Unknown"
    ) -> List[Tuple[str, int, int]]:
        """
        Chunk text with overlap, respecting sentence boundaries
        
        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        if not text or len(text) < self.min_chunk_size:
            if text.strip():
                return [(text.strip(), 0, len(text))]
            return []
        
        chunks = []
        
        if self.respect_sentence_boundaries:
            sentences = self.split_into_sentences(text)
            current_chunk = []
            current_length = 0
            chunk_start = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                if current_length + sentence_length > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunk_end = chunk_start + len(chunk_text)
                    chunks.append((chunk_text, chunk_start, chunk_end))
                    
                    # Calculate overlap
                    overlap_text = ''
                    overlap_length = 0
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) <= self.chunk_overlap:
                            overlap_text = s + ' ' + overlap_text
                            overlap_length += len(s) + 1
                        else:
                            break
                    
                    # Start new chunk with overlap
                    current_chunk = [overlap_text.strip()] if overlap_text.strip() else []
                    current_length = overlap_length
                    chunk_start = chunk_end - overlap_length
                
                current_chunk.append(sentence)
                current_length += sentence_length + 1
            
            # Add remaining chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append((chunk_text, chunk_start, chunk_start + len(chunk_text)))
        
        else:
            # Simple character-based chunking with overlap
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append((chunk_text, start, end))
                start = end - self.chunk_overlap
        
        return chunks
    
    def chunk_paper(self, filepath: Path) -> List[Chunk]:
        """
        Chunk a single paper file
        
        Args:
            filepath: Path to the paper .txt file
            
        Returns:
            List of Chunk objects
        """
        logger.info(f"Chunking paper: {filepath.name}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse metadata
        metadata = self.parse_paper_metadata(content)
        paper_id = filepath.stem
        
        # Extract main content
        paper_content = self.extract_paper_content(content)
        
        # Detect sections
        sections = self.detect_sections(paper_content)
        
        chunks = []
        chunk_index = 0
        
        if sections:
            # Chunk by sections
            for i, (section_name, section_start, line_num) in enumerate(sections):
                # Determine section end
                if i + 1 < len(sections):
                    section_end = sections[i + 1][1]
                else:
                    section_end = len(paper_content)
                
                section_text = paper_content[section_start:section_end].strip()
                
                # Skip references section (usually not useful for RAG)
                if section_name == 'References':
                    continue
                
                # Chunk the section
                section_chunks = self.chunk_text(section_text, section_name)
                
                for chunk_text, start_char, end_char in section_chunks:
                    if len(chunk_text) >= self.min_chunk_size:
                        chunk = Chunk(
                            text=chunk_text,
                            chunk_id=f"{paper_id}_chunk_{chunk_index}",
                            paper_id=paper_id,
                            title=metadata['title'],
                            authors=metadata['authors'],
                            published=metadata['published'],
                            source=metadata['source'],
                            categories=metadata['categories'],
                            pdf_url=metadata['pdf_url'],
                            section=section_name,
                            chunk_index=chunk_index,
                            start_char=section_start + start_char,
                            end_char=section_start + end_char
                        )
                        chunks.append(chunk)
                        chunk_index += 1
        else:
            # No sections detected, chunk the entire content
            content_chunks = self.chunk_text(paper_content, "Full Paper")
            
            for chunk_text, start_char, end_char in content_chunks:
                if len(chunk_text) >= self.min_chunk_size:
                    chunk = Chunk(
                        text=chunk_text,
                        chunk_id=f"{paper_id}_chunk_{chunk_index}",
                        paper_id=paper_id,
                        title=metadata['title'],
                        authors=metadata['authors'],
                        published=metadata['published'],
                        source=metadata['source'],
                        categories=metadata['categories'],
                        pdf_url=metadata['pdf_url'],
                        section="Full Paper",
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=end_char
                    )
                    chunks.append(chunk)
                    chunk_index += 1
        
        # Add abstract as a separate chunk if available
        if metadata['abstract'] and len(metadata['abstract']) >= self.min_chunk_size:
            abstract_chunk = Chunk(
                text=metadata['abstract'],
                chunk_id=f"{paper_id}_abstract",
                paper_id=paper_id,
                title=metadata['title'],
                authors=metadata['authors'],
                published=metadata['published'],
                source=metadata['source'],
                categories=metadata['categories'],
                pdf_url=metadata['pdf_url'],
                section="Abstract",
                chunk_index=-1,  # Special index for abstract
                start_char=0,
                end_char=len(metadata['abstract'])
            )
            chunks.insert(0, abstract_chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {filepath.name}")
        return chunks
    
    def chunk_papers_directory(self, papers_dir: Path) -> List[Chunk]:
        """
        Chunk all papers in a directory
        
        Args:
            papers_dir: Path to directory containing paper .txt files
            
        Returns:
            List of all Chunk objects
        """
        all_chunks = []
        paper_files = list(papers_dir.glob("*.txt"))
        
        logger.info(f"Found {len(paper_files)} paper files to chunk")
        
        for filepath in paper_files:
            try:
                chunks = self.chunk_paper(filepath)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error chunking {filepath.name}: {e}")
                continue
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks


if __name__ == '__main__':
    # Test the chunker
    import argparse
    
    parser = argparse.ArgumentParser(description='Chunk research papers for RAG')
    parser.add_argument('--papers-dir', type=str, default='papers', help='Directory containing paper files')
    parser.add_argument('--chunk-size', type=int, default=512, help='Target chunk size in characters')
    parser.add_argument('--overlap', type=int, default=128, help='Chunk overlap in characters')
    
    args = parser.parse_args()
    
    chunker = PaperChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap
    )
    
    papers_dir = Path(args.papers_dir)
    chunks = chunker.chunk_papers_directory(papers_dir)
    
    # Print sample chunks
    print(f"\n{'='*80}")
    print(f"Sample chunks (first 3):")
    print(f"{'='*80}")
    
    for chunk in chunks[:3]:
        print(f"\nChunk ID: {chunk.chunk_id}")
        print(f"Section: {chunk.section}")
        print(f"Title: {chunk.title[:50]}...")
        print(f"Text preview: {chunk.text[:200]}...")
        print("-" * 40)

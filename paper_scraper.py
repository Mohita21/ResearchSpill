#!/usr/bin/env python3
"""
Research Paper Scraper for GenAI and LLM Papers
Finds and scrapes papers from prestigious conferences (ICML, NeurIPS, ICLR, etc.)
"""

import os
import re
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import arxiv
import pdfplumber
from urllib.parse import urlparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaperScraper:
    """Scraper for research papers on GenAI and LLMs"""
    
    # Keywords for GenAI and LLM research
    GENAI_KEYWORDS = [
        'generative ai', 'generative artificial intelligence',
        'large language model', 'llm', 'llms',
        'gpt', 'chatgpt', 'claude', 'gemini',
        'transformer', 'attention mechanism',
        'foundation model', 'foundation models',
        'multimodal', 'text-to-image', 'text-to-video',
        'diffusion model', 'stable diffusion',
        'retrieval augmented generation', 'rag',
        'prompt engineering', 'in-context learning',
        'fine-tuning', 'parameter efficient',
        'instruction tuning', 'reinforcement learning from human feedback', 'rlhf'
    ]
    
    # Conference keywords
    CONFERENCE_KEYWORDS = [
        'icml', 'neurips', 'nips', 'iclr', 'aaai',
        'acl', 'emnlp', 'naacl', 'cvpr', 'eccv', 'iccv'
    ]
    
    def __init__(self, output_dir: str = "papers", max_papers: int = 50):
        """
        Initialize the paper scraper
        
        Args:
            output_dir: Directory to save scraped papers
            max_papers: Maximum number of papers to scrape
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_papers = max_papers
        self.scraped_papers = []
        
    def search_arxiv_papers(self, days_back: int = 30, max_results: int = 100) -> List[Dict]:
        """
        Search arXiv for recent GenAI/LLM papers
        
        Args:
            days_back: Number of days to look back
            max_results: Maximum results to fetch from arXiv
            
        Returns:
            List of paper dictionaries
        """
        logger.info(f"Searching arXiv for papers from the last {days_back} days...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        papers = []
        
        # Search queries combining GenAI keywords with conference keywords
        queries = [
            '("large language model" OR "llm" OR "llms" OR "generative ai" OR "foundation model")',
            '(transformer AND "language model")',
            '("generative ai" OR "genai")',
            '(gpt OR chatgpt OR claude)',
        ]
        
        seen_ids = set()
        
        for query in queries:
            if len(papers) >= max_results:
                break
                
            try:
                search = arxiv.Search(
                    query=query,
                    max_results=min(50, max_results - len(papers)),
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                for result in search.results():
                    if result.entry_id in seen_ids:
                        continue
                    seen_ids.add(result.entry_id)
                    
                    # Check if paper mentions any conference
                    title_abstract = (result.title + " " + result.summary).lower()
                    mentions_conference = any(
                        conf in title_abstract 
                        for conf in self.CONFERENCE_KEYWORDS
                    )
                    
                    # Check if paper is about GenAI/LLM
                    is_genai = any(
                        keyword in title_abstract 
                        for keyword in self.GENAI_KEYWORDS
                    )
                    
                    if is_genai or mentions_conference:
                        paper_info = {
                            'id': result.entry_id.split('/')[-1],
                            'title': result.title,
                            'authors': [author.name for author in result.authors],
                            'summary': result.summary,
                            'published': result.published,
                            'pdf_url': result.pdf_url,
                            'categories': result.categories,
                            'source': 'arxiv'
                        }
                        papers.append(paper_info)
                        
                        if len(papers) >= max_results:
                            break
                            
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error searching arXiv with query '{query}': {e}")
                continue
        
        logger.info(f"Found {len(papers)} papers from arXiv")
        return papers
    
    def download_pdf(self, url: str, output_path: Path) -> bool:
        """
        Download PDF from URL
        
        Args:
            url: URL of the PDF
            output_path: Path to save the PDF
            
        Returns:
            True if successful, False otherwise
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {e}")
            return False
    
    def pdf_to_text(self, pdf_path: Path) -> Optional[str]:
        """
        Convert PDF to text
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text or None if error
        """
        try:
            text_content = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_content.append(text)
            
            return '\n\n'.join(text_content)
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return None
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for filesystem
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
        return filename
    
    def save_paper_text(self, paper: Dict, text_content: str) -> Path:
        """
        Save paper text to file
        
        Args:
            paper: Paper dictionary
            text_content: Extracted text content
            
        Returns:
            Path to saved file
        """
        # Create filename from title
        filename = self.sanitize_filename(paper['title'])
        filepath = self.output_dir / f"{filename}.txt"
        
        # Add metadata header
        header = f"""Title: {paper['title']}
Authors: {', '.join(paper['authors'])}
Published: {paper['published']}
Source: {paper['source']}
Categories: {', '.join(paper.get('categories', []))}
PDF URL: {paper.get('pdf_url', 'N/A')}

Abstract:
{paper.get('summary', 'N/A')}

{'='*80}
Full Paper Content:
{'='*80}

"""
        
        full_content = header + text_content
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        logger.info(f"Saved paper: {filepath}")
        return filepath
    
    def scrape_paper(self, paper: Dict) -> bool:
        """
        Scrape a single paper
        
        Args:
            paper: Paper dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pdf_url = paper.get('pdf_url')
            if not pdf_url:
                logger.warning(f"No PDF URL for paper: {paper['title']}")
                return False
            
            # Download PDF to temporary location
            temp_pdf = self.output_dir / f"temp_{paper['id']}.pdf"
            
            logger.info(f"Downloading PDF for: {paper['title']}")
            if not self.download_pdf(pdf_url, temp_pdf):
                return False
            
            # Extract text from PDF
            logger.info(f"Extracting text from PDF: {paper['title']}")
            text_content = self.pdf_to_text(temp_pdf)
            
            if not text_content:
                logger.warning(f"Could not extract text from PDF: {paper['title']}")
                temp_pdf.unlink()  # Clean up
                return False
            
            # Save text file
            self.save_paper_text(paper, text_content)
            
            # Clean up temporary PDF
            temp_pdf.unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"Error scraping paper {paper.get('title', 'Unknown')}: {e}")
            return False
    
    def run(self, days_back: int = 30):
        """
        Main method to run the scraper
        
        Args:
            days_back: Number of days to look back for papers
        """
        logger.info("Starting paper scraper...")
        
        # Search for papers
        papers = self.search_arxiv_papers(days_back=days_back, max_results=self.max_papers * 2)
        
        if not papers:
            logger.warning("No papers found!")
            return
        
        # Limit to max_papers
        papers = papers[:self.max_papers]
        
        logger.info(f"Scraping {len(papers)} papers...")
        
        successful = 0
        failed = 0
        
        for i, paper in enumerate(papers, 1):
            logger.info(f"\n[{i}/{len(papers)}] Processing: {paper['title']}")
            
            if self.scrape_paper(paper):
                successful += 1
                self.scraped_papers.append(paper)
            else:
                failed += 1
            
            # Rate limiting
            time.sleep(2)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Scraping complete!")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Papers saved to: {self.output_dir.absolute()}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Scrape GenAI and LLM research papers from prestigious conferences'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='papers',
        help='Directory to save scraped papers (default: papers)'
    )
    parser.add_argument(
        '--max-papers',
        type=int,
        default=50,
        help='Maximum number of papers to scrape (default: 50)'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        default=30,
        help='Number of days to look back for papers (default: 30)'
    )
    
    args = parser.parse_args()
    
    scraper = PaperScraper(
        output_dir=args.output_dir,
        max_papers=args.max_papers
    )
    
    scraper.run(days_back=args.days_back)


if __name__ == '__main__':
    main()

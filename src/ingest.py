"""SEC 10-K filing ingestion pipeline."""

from pathlib import Path
from typing import List, Dict, Optional
import re
import time

from bs4 import BeautifulSoup

from .config import config


class SECIngestor:
    """Download and parse SEC 10-K filings."""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else config.sec_filings_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_all(self) -> Dict[str, int]:
        """Download 10-K filings for all tickers."""
        from sec_edgar_downloader import Downloader
        
        results = {}
        downloader = Downloader(
            company_name=config.sec.company_name,
            email_address=config.sec.email,
            download_folder=str(self.data_dir)
        )
        
        for ticker in config.sec.tickers:
            try:
                print(f"Downloading {config.sec.filing_type} for {ticker}...")
                downloader.get(
                    config.sec.filing_type,
                    ticker,
                    limit=config.sec.filing_limit
                )
                results[ticker] = config.sec.filing_limit
                print(f"  Downloaded {ticker} successfully")
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"  Error downloading {ticker}: {e}")
                results[ticker] = 0
        
        return results
    
    def parse_filing(self, filepath: Path) -> Dict:
        """Parse a single filing HTML file."""
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

        # SEC header block contains filing metadata
        start = raw.find("<SEC-HEADER>")
        end = raw.find("</SEC-HEADER>")
        header = raw[start:end] if 0 <= start < end else ""

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "sec-header"]):
            tag.decompose()
        
        # Extract text
        text = soup.get_text(separator="\n", strip=True)
        
        # Clean text
        text = self._clean_text(text)
        
# Extract metadata
        metadata = self._extract_metadata(filepath, text, header)

        return {
            "content": text,
            "metadata": metadata,
            "source": str(filepath)
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove page numbers
        text = re.sub(r'\n\d+\n', '\n', text)
        
        return text.strip()
    
    def _extract_metadata(self, filepath: Path, text: str, header: str = "") -> Dict:
        """Extract metadata from filing."""
        # Ticker from sec-edgar-filings/<TICKER>/10-K/<accession>/file.txt
        try:
            ticker = filepath.parts[-4].upper()
        except (IndexError, AttributeError):
            ticker = "UNKNOWN"

        # Extract fiscal period / filing date from SEC header
        date_match = re.search(r'(?:CONFORMED PERIOD OF REPORT|FILED AS OF DATE):\s*(\d{8})', header)
        filing_date = "unknown"
        if date_match:
            d = date_match.group(1)
            filing_date = f"{d[0:4]}-{d[4:6]}-{d[6:8]}"
        
        # Extract section (simplified)
        section = "general"
        text_lower = text[:5000].lower()
        if "risk factors" in text_lower:
            section = "risk_factors"
        elif "financial statements" in text_lower:
            section = "financial_statements"
        elif "business" in text_lower:
            section = "business"
        
        return {
            "ticker": ticker,
            "filing_type": config.sec.filing_type,
            "filing_date": filing_date,
            "section": section,
            "source": str(filepath)
        }
    
    def load_all_filings(self) -> List[Dict]:
        """Load and parse all downloaded filings."""
        all_filings = []
        extensions = (".htm", ".html", ".txt")
        for filepath in self.data_dir.rglob("*"):
            if filepath.is_file() and filepath.suffix.lower() in extensions:
                try:
                    filing = self.parse_filing(filepath)
                    all_filings.append(filing)
                except Exception as e:
                    print(f"Error parsing {filepath}: {e}")
        
        return all_filings

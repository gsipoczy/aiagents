from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from typing import List

class SmartPdfProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[" "]
        )

    def process_pdf(self, pdf_path:str) -> List[Document]:
        processed_chunks = []
        
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            for page_num, page in enumerate(pages):
                # Clean the text
                cleaned_text = self._clean_text(page.page_content)

                # Skip nearly empty pages
                if len(cleaned_text.strip()) < 50:
                    continue

                # Split the cleaned text into chunks
                chunks = self.text_splitter.create_documents(
                    texts = [cleaned_text],
                    metadatas = [{
                        **page.metadata,
                        "page": page_num + 1,
                        "total_pages": len(pages),
                        "chunk_method": "RecursiveCharacterTextSplitter",
                        "char_count": len(cleaned_text),
                        "source": pdf_path,

                    }]
                )
                processed_chunks.extend(chunks)
        except Exception as e: pass

        return processed_chunks
    
    def _clean_text(self, text: str) -> str:
        # Remove extra whitespaces
        text = ' '.join(text.split())

        # Replace encoded characters
        replacements = {
            '\u2013': '-',  # en dash
            '\u2014': '-',  # em dash
            '\u2018': "'",  # left single quotation mark
            '\u2019': "'",  # right single quotation mark
            '\u201c': '"',  # left double quotation mark
            '\u201d': '"',  # right double quotation mark
            '\u2026': '...',  # ellipsis
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        return text
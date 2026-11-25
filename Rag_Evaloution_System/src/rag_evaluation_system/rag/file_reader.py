from pymupdf import open
import re
from typing import List
from dotenv import load_dotenv
from pathlib import Path


def extract_paragraphs(pdf_path: str | Path)-> List[str]:
    """ This function take a pdf path and returns its list of paragraphs"""
    doc = open(pdf_path)
    paragraphs = []
    
    for page in doc:
        blocks = page.get_text("blocks")
        
        blocks = sorted(blocks,key= lambda b:(b[1],b[0]))
        
        for block in blocks:
            text = block[4]
            text = re.sub(r"\r+"," ",text).strip()
            
            if text:
                paragraphs.append(text)
                
    return paragraphs           


#print(extract_paragraphs("C:\EveryPython\Rag_Evaloution_System\pdfs\Penguins_ACL.pdf"))
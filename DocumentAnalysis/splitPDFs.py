import re
import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text_by_hierarchical_paragraphs(text):
    # Regular expression to match hierarchical paragraphs up to three levels like 1.1, 1.1.1, etc.
    paragraph_pattern = re.compile(r'^\d+\.\d+\.\d+\. |\d+\.\d+\. ')
    lines = text.split("\n")
    chunks = []
    current_chunk = ""

    for line in lines:
        if paragraph_pattern.match(line.strip()):
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
        current_chunk += line + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Get the current working directory
current_directory = os.getcwd()

# Construct the full path to your repository
pdf_path_1 = os.path.join(current_directory, "dafman91-119.pdf")

# Extract text
dafman_text = extract_text_from_pdf(pdf_path_1)

# Split text
dafman_chunks = split_text_by_hierarchical_paragraphs(dafman_text)

dafman_chunk_file = open("dafman91-119_chunks.txt", "a")
for chunk in dafman_chunks:
    dafman_chunk_file.write(f"Chunk: {chunk}\n")
dafman_chunk_file.close()
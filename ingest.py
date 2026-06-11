from pypdf import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text


def split_text(text, chunk_size=1000, overlap=200):
    chunks = []

    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    text = load_pdf(r"C:\Project\PDF_RAG_ChatBot\attention.pdf")
    chunks = split_text(text)

    print(f"Total chunks: {len(chunks)}")
    print(chunks[0])
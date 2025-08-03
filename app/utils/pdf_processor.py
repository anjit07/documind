from langchain.document_loaders import PyMuPDFLoader
class PDFProcessor:
    
    def __init__(self):
        # Initialize any necessary configurations or parameters here
        pass

    # Extract text from a PDF file using LangChain's PyMuPDFLoader
    def extract_text(self, pdf_path):

        print(f"Load file from the path: {pdf_path}")
        # Use LangChain's built-in loader
        loader = PyMuPDFLoader(pdf_path)
        # Load the PDF into LangChain's document format
        documents = loader.load()

        print(f"Successfully loaded {len(documents)} document chunks from the PDF.")
        return documents
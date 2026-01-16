from langchain_text_splitters import MarkdownHeaderTextSplitter

def chunk_markdown(markdown_text: str):
    """
    Splits markdown text into chunks based on headers.
    This ensures that 'Experience' stays with the experience details.
    """
    # Define the headers we want to split by
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = markdown_splitter.split_text(markdown_text)
    
    # Return list of strings (content) and metadata
    return docs
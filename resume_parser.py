"""
resume_parser.py
-----------------
Extracts plain text from uploaded resume/CV files (PDF, DOCX, or TXT) so it
can be fed into the personality prediction pipeline.

Note: resumes are structurally very different from the free-flowing personal
essays the model was trained on (bullet points, headers, dense noun phrases
vs. stream-of-consciousness prose). This is a known limitation - predictions
on resume text should be treated as a rough signal, not a validated
assessment. Worth being upfront about this in any writeup or demo.
"""

import os

import docx
import PyPDF2


def extract_text_from_pdf(file_path_or_buffer):
    """Extracts text from a PDF file. Accepts either a file path or an
    already-open file-like object (e.g. from Streamlit's file uploader)."""
    reader = PyPDF2.PdfReader(file_path_or_buffer)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_docx(file_path_or_buffer):
    """Extracts text from a .docx file."""
    document = docx.Document(file_path_or_buffer)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_text_from_txt(file_path_or_buffer):
    """Extracts text from a plain .txt file."""
    if hasattr(file_path_or_buffer, "read"):
        content = file_path_or_buffer.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        return content
    with open(file_path_or_buffer, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_resume_text(file_path_or_buffer, filename=None):
    """
    Auto-detects file type by extension and extracts text accordingly.

    Args:
        file_path_or_buffer: a file path (str) or file-like object
        filename: required if passing a buffer (to detect extension);
                  ignored if file_path_or_buffer is already a path string

    Returns:
        extracted plain text (str)

    Raises:
        ValueError: if the file type isn't supported
    """
    if filename is None:
        filename = file_path_or_buffer if isinstance(file_path_or_buffer, str) else ""

    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path_or_buffer)
    elif ext == ".docx":
        return extract_text_from_docx(file_path_or_buffer)
    elif ext == ".txt":
        return extract_text_from_txt(file_path_or_buffer)
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Supported formats: .pdf, .docx, .txt"
        )

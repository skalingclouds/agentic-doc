# agentic-doc

A Python library that wraps around [VisionAgent document extraction REST API](https://va.landing.ai/demo/doc-extraction) to make documents extraction easy.

## Quick Start

### Installation

```bash
pip install agentic-doc
```

### Prerequisites

Set your Vision Agent API key as an environment variable (or put it in a `.env` file):

```bash
export VISION_AGENT_API_KEY=<your-api-key>
```

NOTE: the API key can be found from [here](https://va.landing.ai/account/api-key)

### Basic Usage

#### Parse a single document

```python
from agentic_doc.parse import parse_documents

results = parse_documents(["path/to/image.png"])
parsed_doc = results[0]
print(parsed_doc.markdown)  # Get markdown representation
print(parsed_doc.chunks)  # Get structured chunks of content
```

#### Parse multiple documents and save results to a directory

```python
from agentic_doc.parse import parse_and_save_documents

file_paths = ["path/to/your/document.pdf", "path/to/another/document.pdf"]
result_save_dir = "path/to/save/results"

result_paths = parse_and_save_documents(file_paths, result_save_dir)
```

## Configuration Options

The library uses a [`Settings` object](./agentic_doc/config.py) to manage configuration. You can customize these settings either through environment variables:


```bash
export MAX_WORKERS=4 # Number of worker threads for parallel processing, defaults to 10
export MAX_RETRIES=100 # Maximum number of retry attempts for failed requests, defaults to 100
```

## API Reference

### Main Functions

#### `parse_documents(file_paths: list[str | Path]) -> list[ParsedDocument]`

Parse multiple documents and return their parsed results.

- **Parameters:**
  - `file_paths`: List of paths to documents (PDF or images)
- **Returns:**
  - List of `ParsedDocument` objects containing parsed results
- **Raises:**
  - `FileNotFoundError`: If any input file doesn't exist

#### `parse_and_save_documents(file_paths: list[str | Path], *, result_save_dir: str | Path) -> list[Path]`

Parse multiple documents and save results to specified directory.

- **Parameters:**
  - `file_paths`: List of paths to documents
  - `result_save_dir`: Directory to save parsed results
- **Returns:**
  - A list of json file paths to the saved results. The file paths are sorted by the order of the input file paths. The file name is the original file name with a timestamp appended. E.g. "document.pdf" -> "document_20250313_070305.json".
- **Raises:**
  - `FileNotFoundError`: If any input file doesn't exist

#### `parse_and_save_document(file_path: str | Path, *, result_save_dir: str | Path = None) -> Path | ParsedDocument`

Parse a single document and optionally save results.

- **Parameters:**
  - `file_path`: Path to document
  - `result_save_dir`: Optional directory to save results
- **Returns:**
  - If `result_save_dir` provided: Path to saved result file
  - If no `result_save_dir`: ParsedDocument object
- **Raises:**
  - `FileNotFoundError`: If input file doesn't exist
  - `ValueError`: If file type is not supported

### Data Classes

#### ParsedDocument

Represents a parsed document with the following attributes:

- `markdown`: str - Markdown representation of the document
- `chunks`: list[Chunk] - List of parsed content chunks, sorted by page index, then the layout of the content in the page
- `start_page_idx`: Optional[int] - Starting page index for PDFs
- `end_page_idx`: Optional[int] - Ending page index for PDFs
- `doc_type`: Literal["pdf", "image"] - Type of document

#### Chunk

Represents a parsed content chunk with the following attributes:

- `text`: str - Extracted text content
- `grounding`: list[Grounding] - List of content locations in document
- `chunk_type`: Literal["text", "error"] - Type of chunk
- `chunk_id`: Optional[str] - ID of the chunk

## Error Handling

The library implements a robust retry mechanism for handling API failures:

- Retries are performed for HTTP status codes: 408, 429, 502, 503, 504
- Exponential backoff with jitter is used for retry wait time
- Initial retry wait time is 1 second, increasing exponentially
- Maximum retry wait time is 300 seconds
- Jitter of 5 seconds is added to prevent thundering herd

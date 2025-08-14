import base64
import json
import os
import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import httpx
import numpy as np
import pymupdf
import pytest
import requests
from PIL import Image
from pydantic_core import Url
from requests.exceptions import ConnectionError as RequestsConnectionError
from tenacity import RetryCallState

from agentic_doc.common import (
    Chunk,
    ChunkGrounding,
    ChunkGroundingBox,
    ChunkType,
    Document,
    ParsedDocument,
)
from agentic_doc.config import VisualizationConfig, settings
from agentic_doc.utils import (
    _crop_groundings,
    _crop_image,
    _place_mark,
    _read_img_rgb,
    check_endpoint_and_api_key,
    download_file,
    get_file_type,
    is_valid_httpurl,
    log_retry_failure,
    page_to_image,
    save_groundings_as_images,
    split_pdf,
    viz_chunks,
    viz_parsed_document,
    get_chunk_from_reference,
)


@pytest.mark.parametrize(
    "api_key_str, mock_response_status, side_effect, expected_exception, expected_msg",
    [
        # No API key
        ("", None, None, ValueError, "API key is not set"),
        # Endpoint down
        (
            base64.b64encode(b"user:pass").decode(),
            None,
            RequestsConnectionError("mocked connection error"),
            ValueError,
            "endpoint URL",
        ),
        # 404 Not Found
        (
            base64.b64encode(b"user:pass").decode(),
            404,
            None,
            ValueError,
            "API key is not valid for this endpoint",
        ),
        # 401 Unauthorized
        (
            base64.b64encode(b"user:pass").decode(),
            401,
            None,
            ValueError,
            "API key is invalid",
        ),
    ],
)
def test_check_endpoint_and_api_key_failures(
    api_key_str, mock_response_status, side_effect, expected_exception, expected_msg
):
    if side_effect is not None:
        mock_requests_get = MagicMock(side_effect=side_effect)
    else:
        mock_resp = MagicMock()
        mock_resp.status_code = mock_response_status
        mock_requests_get = MagicMock(return_value=mock_resp)

    with patch("agentic_doc.utils.requests.head", mock_requests_get):
        with pytest.raises(expected_exception) as exc_info:
            check_endpoint_and_api_key("https://example123.com", api_key=api_key_str)

        assert expected_msg in str(exc_info.value)


def test_check_endpoint_and_api_key_success():
    valid_api_key = base64.b64encode(b"user:pass").decode()

    mock_resp = MagicMock()
    mock_resp.json.return_value = {}

    with patch("agentic_doc.utils.requests.get", return_value=mock_resp):
        check_endpoint_and_api_key("https://example.com", valid_api_key)


def test_download_file_with_url(results_dir):
    url = "https://pdfobject.com/pdf/sample.pdf"
    output_file_path = Path(results_dir) / "sample.pdf"
    download_file(url, str(output_file_path))
    assert output_file_path.exists()
    assert output_file_path.name == "sample.pdf"
    assert output_file_path.stat().st_size > 0


def test_download_file_failure(monkeypatch):
    # Mock httpx.stream to simulate a failed download
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "Not Found"

    # Create a context manager mock
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response

    # Mock stream to return our context manager
    mock_stream = MagicMock(return_value=mock_context)
    monkeypatch.setattr(httpx, "stream", mock_stream)

    with pytest.raises(Exception) as exc_info:
        download_file("https://example.com/nonexistent.pdf", "output.pdf")

    # Just check that the error message contains "Download failed"
    assert "Download failed" in str(exc_info.value)


# Convert a standard PDF page to an RGB image with actual dimensions at default DPI
def test_convert_pdf_page_to_rgb_image_with_actual_dimensions(complex_pdf):
    with pymupdf.open(complex_pdf) as pdf_doc:
        result = page_to_image(pdf_doc, 0)
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3  # RGB channels
        assert result.dtype == np.uint8


# Handle PDF with RGBA content by dropping alpha channel
def test_handle_rgba_content_by_dropping_alpha_channel(monkeypatch):
    # Create a PDF document
    with pymupdf.open() as pdf_doc:
        pdf_doc.new_page(width=100, height=100)
        # Create a mock pixmap with RGBA data (4 channels)
        rgba_data = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba_data[..., 3] = 255  # Set alpha channel to 255

        # Create a mock get_pixmap method that returns a pixmap with RGBA data
        class MockPixmap:
            def __init__(self):
                self.samples = rgba_data.tobytes()
                self.h = 100
                self.w = 100

        def mock_get_pixmap(*args, **kwargs):
            return MockPixmap()

        monkeypatch.setattr(pymupdf.Page, "get_pixmap", mock_get_pixmap)

        # Call the function under test
        result = page_to_image(pdf_doc, 0)

        # Assert the result has only 3 channels (RGB, no alpha)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100, 3)


def test_is_valid_httpurl():
    # Valid URLs
    assert is_valid_httpurl("http://example.com")
    assert is_valid_httpurl("https://example.com")
    assert is_valid_httpurl("https://example.com/path/to/file.pdf")

    # Invalid URLs
    assert not is_valid_httpurl("ftp://example.com")
    assert not is_valid_httpurl("file:///path/to/file.pdf")
    assert not is_valid_httpurl("/path/to/file.pdf")
    assert not is_valid_httpurl("example.com")
    assert not is_valid_httpurl("not a url")


def test_get_file_type_pdf(temp_dir):
    # Create a PDF file with proper header
    pdf_path = temp_dir / "test.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.7\n")

    assert get_file_type(pdf_path) == "pdf"


def test_get_file_type_image(temp_dir):
    # Create a fake image file
    img_path = temp_dir / "test.jpg"
    with open(img_path, "wb") as f:
        f.write(b"JFIF")  # Some non-PDF content

    assert get_file_type(img_path) == "image"


def test_get_file_type_fallback_to_extension(temp_dir):
    # File that can't be opened
    nonexistent_path = temp_dir / "nonexistent.pdf"
    assert get_file_type(nonexistent_path) == "pdf"

    nonexistent_image = temp_dir / "nonexistent.jpg"
    assert get_file_type(nonexistent_image) == "image"


def test_split_pdf(multi_page_pdf, temp_dir):
    # Test splitting a multi-page PDF
    output_dir = temp_dir / "split_output"
    result = split_pdf(multi_page_pdf, output_dir, split_size=2)

    # For a 5-page PDF with split_size=2, we should get 3 parts
    assert len(result) == 3

    # Check that each Document object has the correct page ranges
    assert result[0].start_page_idx == 0
    assert result[0].end_page_idx == 1

    assert result[1].start_page_idx == 2
    assert result[1].end_page_idx == 3

    assert result[2].start_page_idx == 4
    assert result[2].end_page_idx == 4

    # Check that the files were actually created
    for doc in result:
        assert Path(doc.file_path).exists()


def test_split_pdf_with_invalid_split_size(multi_page_pdf, temp_dir):
    output_dir = temp_dir / "split_output"

    # Test with invalid split_size values
    with pytest.raises(AssertionError):
        split_pdf(multi_page_pdf, output_dir, split_size=0)


def test_log_retry_failure_inline_block(monkeypatch, capsys):
    # Setup a mock retry state
    retry_state = MagicMock()
    retry_state.attempt_number = 3
    outcome = MagicMock()
    outcome.failed = True
    outcome.exception.return_value = Exception("Test error")
    retry_state.outcome = outcome

    # Set the retry logging style to inline_block
    settings.retry_logging_style = "inline_block"

    # Call the function
    log_retry_failure(retry_state)

    # Check that the progress block was printed
    captured = capsys.readouterr()
    assert "███" in captured.out


def test_log_retry_failure_none(monkeypatch, capsys, caplog):
    # Setup a mock retry state
    retry_state = MagicMock()
    retry_state.attempt_number = 3
    outcome = MagicMock()
    outcome.failed = True
    outcome.exception.return_value = Exception("Test error")
    retry_state.outcome = outcome

    # Set the retry logging style to none
    settings.retry_logging_style = "none"

    # Call the function
    log_retry_failure(retry_state)

    # Check that nothing was printed or logged
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "attempt" not in caplog.text


def test_log_retry_failure_invalid_style(monkeypatch):
    # Setup a mock retry state
    retry_state = MagicMock()
    retry_state.attempt_number = 3
    outcome = MagicMock()
    outcome.failed = True
    outcome.exception.return_value = Exception("Test error")
    retry_state.outcome = outcome

    # Set an invalid retry logging style
    settings.retry_logging_style = "invalid"
    
    # Call the function and check that it raises a ValueError
    with pytest.raises(ValueError) as exc_info:
        log_retry_failure(retry_state)

    assert "Invalid retry logging style" in str(exc_info.value)


def test_viz_parsed_document_image(temp_dir, mock_parsed_document):
    # Create a test image
    img_path = temp_dir / "test_image.png"
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    img.save(img_path)

    # Test visualization without saving
    with patch("agentic_doc.utils.get_file_type", return_value="image"):
        images = viz_parsed_document(img_path, mock_parsed_document)

        # Check that we got an image back
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)
        assert images[0].width == 200
        assert images[0].height == 200

    # Test visualization with saving
    output_dir = temp_dir / "viz_output"
    with patch("agentic_doc.utils.get_file_type", return_value="image"):
        images = viz_parsed_document(
            img_path, mock_parsed_document, output_dir=output_dir
        )

        # Check that the image was saved
        assert (output_dir / f"{img_path.stem}_viz_page_0.png").exists()


def test_viz_parsed_document_pdf(temp_dir, mock_multi_page_parsed_document):
    # Mock pymupdf.open and page_to_image to avoid needing a real PDF
    mock_page_image = np.zeros((200, 200, 3), dtype=np.uint8)

    with patch("agentic_doc.utils.pymupdf.open"), patch(
        "agentic_doc.utils.page_to_image", return_value=mock_page_image
    ), patch("agentic_doc.utils.get_file_type", return_value="pdf"):

        pdf_path = temp_dir / "test.pdf"
        output_dir = temp_dir / "viz_output"

        # Test visualization with saving
        images = viz_parsed_document(
            pdf_path, mock_multi_page_parsed_document, output_dir=output_dir
        )

        # Check that we got the right number of images back
        assert len(images) == 3  # 3 pages in mock_multi_page_parsed_document

        # Check that the images were saved
        for i in range(3):
            assert (output_dir / f"{pdf_path.stem}_viz_page_{i}.png").exists()


def test_viz_chunks():
    # Create a test image
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    # Create some test chunks
    chunks = [
        Chunk(
            text="Test Title",
            chunk_type=ChunkType.text,
            chunk_id="1",
            grounding=[
                ChunkGrounding(
                    page=0, box=ChunkGroundingBox(l=0.1, t=0.1, r=0.9, b=0.2)
                )
            ],
        ),
        Chunk(
            text="Test Text",
            chunk_type=ChunkType.text,
            chunk_id="2",
            grounding=[
                ChunkGrounding(
                    page=0, box=ChunkGroundingBox(l=0.1, t=0.3, r=0.9, b=0.4)
                )
            ],
        ),
    ]

    # Test with default visualization config
    result = viz_chunks(img, chunks)
    assert isinstance(result, np.ndarray)
    assert result.shape == (200, 200, 3)

    # Test with custom visualization config
    viz_config = VisualizationConfig(thickness=2, font_scale=0.7)
    result = viz_chunks(img, chunks, viz_config)
    assert isinstance(result, np.ndarray)
    assert result.shape == (200, 200, 3)


def test_crop_image():
    # Create a test image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Fill with different colors to verify the crop
    img[0:50, 0:50] = [255, 0, 0]  # Top-left quadrant: red
    img[0:50, 50:100] = [0, 255, 0]  # Top-right quadrant: green
    img[50:100, 0:50] = [0, 0, 255]  # Bottom-left quadrant: blue
    img[50:100, 50:100] = [255, 255, 0]  # Bottom-right quadrant: yellow

    # Test crop with normalized coordinates
    bbox = ChunkGroundingBox(l=0.25, t=0.25, r=0.75, b=0.75)
    crop = _crop_image(img, bbox)

    # The crop should be a 50x50 region from the center of the image
    assert crop.shape == (50, 50, 3)

    # Test with coordinates at the boundaries
    bbox = ChunkGroundingBox(l=0.0, t=0.0, r=1.0, b=1.0)
    crop = _crop_image(img, bbox)
    assert crop.shape == (100, 100, 3)


def test_crop_groundings(temp_dir):
    # Create a test image
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Create a directory to save the crops
    crop_save_dir = temp_dir / "crops"

    # Create test chunks
    chunks = [
        Chunk(
            text="Test Document",
            chunk_type=ChunkType.text,
            chunk_id="11111",
            grounding=[
                ChunkGrounding(
                    page=0, box=ChunkGroundingBox(l=0.1, t=0.1, r=0.9, b=0.2)
                )
            ],
        ),
        Chunk(
            text="This is a test document.",
            chunk_type=ChunkType.text,
            chunk_id="22222",
            grounding=[
                ChunkGrounding(
                    page=0, box=ChunkGroundingBox(l=0.1, t=0.3, r=0.9, b=0.4)
                )
            ],
        ),
    ]

    # Mock cv2.imencode to make it return a valid result
    mock_buffer = MagicMock()
    mock_buffer.tobytes.return_value = b"mock_png_data"

    with patch("cv2.imencode", return_value=(True, mock_buffer)), patch(
        "pathlib.Path.write_bytes"
    ) as mock_write:

        # Test without inplace modification
        result = _crop_groundings(img, chunks, crop_save_dir, inplace=False)

        # Check that the result contains the chunk_id as keys
        assert "11111" in result
        assert "22222" in result

        # Check that write_bytes was called for each chunk
        assert mock_write.call_count >= 2

        # Verify the grounding image_path is still None (since inplace=False)
        assert chunks[0].grounding[0].image_path is None

        # Reset the mock for the next test
        mock_write.reset_mock()

        # Test with inplace modification
        result = _crop_groundings(img, chunks, crop_save_dir, inplace=True)

        # Check that write_bytes was called for each chunk
        assert mock_write.call_count >= 2

        # Check that the image_path was set in the chunks when inplace=True
        assert chunks[0].grounding[0].image_path is not None
        assert chunks[1].grounding[0].image_path is not None


def test_save_groundings_as_images_image(temp_dir):
    # Create a test image file
    img_path = temp_dir / "test.jpg"
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    img.save(img_path)

    # Create a directory to save the groundings
    save_dir = temp_dir / "groundings"

    # Create custom chunks with known types
    chunks = [
        Chunk(
            text="Test Document",
            chunk_type=ChunkType.text,
            chunk_id="11111",
            grounding=[
                ChunkGrounding(
                    page=0, box=ChunkGroundingBox(l=0.1, t=0.1, r=0.9, b=0.2)
                )
            ],
        ),
        Chunk(
            text="This is a test document.",
            chunk_type=ChunkType.text,
            chunk_id="22222",
            grounding=[
                ChunkGrounding(
                    page=0, box=ChunkGroundingBox(l=0.1, t=0.3, r=0.9, b=0.4)
                )
            ],
        ),
    ]

    # Mock the required functions to avoid filesystem operations
    mock_buffer = MagicMock()
    mock_buffer.tobytes.return_value = b"mock_png_data"

    with patch("agentic_doc.utils.get_file_type", return_value="image"), patch(
        "agentic_doc.utils.cv2.imread",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ), patch("cv2.imencode", return_value=(True, mock_buffer)), patch(
        "pathlib.Path.write_bytes"
    ) as mock_write, patch(
        "pathlib.Path.mkdir", return_value=None
    ):

        result = save_groundings_as_images(img_path, chunks, save_dir)

        # Check that the result contains the chunk_id as keys
        assert "11111" in result
        assert "22222" in result

        # Check that write_bytes was called (twice, once for each chunk)
        assert mock_write.call_count == 2


def test_save_groundings_as_images_pdf(temp_dir):
    # Create a dummy PDF file
    pdf_path = temp_dir / "test.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.7\n")

    # Create a directory to save the groundings
    save_dir = temp_dir / "groundings"

    # Create custom chunks with different page indices
    chunks = [
        Chunk(
            text="Title",
            chunk_type=ChunkType.text,
            chunk_id="11111",
            grounding=[
                ChunkGrounding(
                    page=0, box=ChunkGroundingBox(l=0.1, t=0.1, r=0.9, b=0.2)
                )
            ],
        ),
        Chunk(
            text="Page content",
            chunk_type=ChunkType.text,
            chunk_id="22222",
            grounding=[
                ChunkGrounding(
                    page=0, box=ChunkGroundingBox(l=0.1, t=0.3, r=0.9, b=0.4)
                )
            ],
        ),
        Chunk(
            text="Header",
            chunk_type=ChunkType.text,
            chunk_id="33333",
            grounding=[
                ChunkGrounding(
                    page=1, box=ChunkGroundingBox(l=0.1, t=0.1, r=0.9, b=0.2)
                )
            ],
        ),
    ]

    # Mock the required functions to avoid filesystem operations
    mock_buffer = MagicMock()
    mock_buffer.tobytes.return_value = b"mock_png_data"

    with patch("agentic_doc.utils.get_file_type", return_value="pdf"), patch(
        "agentic_doc.utils.pymupdf.open"
    ) as mock_pymupdf_open, patch(
        "agentic_doc.utils.page_to_image",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ), patch(
        "cv2.imencode", return_value=(True, mock_buffer)
    ), patch(
        "pathlib.Path.write_bytes"
    ) as mock_write, patch(
        "pathlib.Path.mkdir", return_value=None
    ):

        # Mock the context manager returned by pymupdf.open
        mock_pdf_doc = MagicMock()
        mock_pymupdf_open.return_value.__enter__.return_value = mock_pdf_doc

        # Call the function
        result = save_groundings_as_images(pdf_path, chunks, save_dir)

        # Check that the result contains the chunk_ids
        assert "11111" in result
        assert "22222" in result
        assert "33333" in result

        # Check that write_bytes was called for each chunk
        assert mock_write.call_count == 3


def test_read_img_rgb():
    # Create a mock for cv2.imread and cv2.cvtColor
    with patch(
        "agentic_doc.utils.cv2.imread",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ), patch(
        "agentic_doc.utils.cv2.cvtColor",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ):

        # Test with a regular RGB image
        result = _read_img_rgb("test.jpg")
        assert result.shape == (100, 100, 3)

    # Test with a grayscale image
    with patch(
        "agentic_doc.utils.cv2.imread",
        return_value=np.zeros((100, 100, 1), dtype=np.uint8),
    ), patch("agentic_doc.utils.cv2.cvtColor") as mock_cvtColor:

        # Set return value directly instead of using side_effect
        mock_cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        result = _read_img_rgb("test.jpg")
        assert result.shape == (100, 100, 3)
        # Check that cvtColor was called at least once
        assert mock_cvtColor.call_count >= 1

    # Test with an RGBA image
    with patch(
        "agentic_doc.utils.cv2.imread",
        return_value=np.zeros((100, 100, 4), dtype=np.uint8),
    ), patch(
        "agentic_doc.utils.cv2.cvtColor",
        return_value=np.zeros((100, 100, 4), dtype=np.uint8),
    ):

        result = _read_img_rgb("test.png")
        assert result.shape == (100, 100, 3)  # Should drop the alpha channel


def test_split_pdf_edge_cases(temp_dir):
    # Test edge cases for PDF splitting
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    # Create a 1-page PDF
    single_page_pdf = temp_dir / "single_page.pdf"
    doc = SimpleDocTemplate(str(single_page_pdf), pagesize=letter)
    styles = getSampleStyleSheet()
    doc.build([Paragraph("Single page content", styles["Normal"])])

    # Test with split_size=1 on a 1-page PDF
    output_dir = temp_dir / "split_single"
    result = split_pdf(single_page_pdf, output_dir, split_size=1)

    assert len(result) == 1
    assert result[0].start_page_idx == 0
    assert result[0].end_page_idx == 0


def test_download_file_with_custom_filename(results_dir):
    # Test downloading to a specific filename
    url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    custom_filename = Path(results_dir) / "custom_name.pdf"

    download_file(url, str(custom_filename))

    assert custom_filename.exists()
    assert custom_filename.name == "custom_name.pdf"
    assert custom_filename.stat().st_size > 0


def test_get_file_type_with_various_extensions(temp_dir):
    # Test file type detection with different extensions

    # PDF files
    for ext in [".pdf", ".PDF"]:
        pdf_file = temp_dir / f"test{ext}"
        with open(pdf_file, "wb") as f:
            f.write(b"%PDF-1.7\n")
        assert get_file_type(pdf_file) == "pdf"

    # Image files
    for ext in [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".JPG", ".PNG"]:
        img_file = temp_dir / f"test{ext}"
        with open(img_file, "wb") as f:
            f.write(b"fake image data")
        assert get_file_type(img_file) == "image"


def test_is_valid_httpurl_edge_cases():
    # Test edge cases for URL validation

    # Valid URLs with different protocols
    assert is_valid_httpurl("http://example.com")
    assert is_valid_httpurl("https://example.com")
    assert is_valid_httpurl("https://sub.domain.example.com/path/file.pdf")
    assert is_valid_httpurl("http://localhost:8080/test")

    # Invalid URLs (scheme-based validation)
    assert not is_valid_httpurl("")
    assert not is_valid_httpurl("   ")
    assert not is_valid_httpurl("ftp://example.com")
    assert not is_valid_httpurl("ftps://example.com")
    assert not is_valid_httpurl("file:///local/path")
    assert not is_valid_httpurl("mailto:test@example.com")
    assert not is_valid_httpurl("just-a-string")

    # Note: The function only validates scheme, so "http://" and "https://" are considered valid
    assert is_valid_httpurl("http://")
    assert is_valid_httpurl("https://")


def test_page_to_image_different_dpi_settings(complex_pdf, monkeypatch):
    # Test page conversion with different DPI settings

    # Test with high DPI
    with pymupdf.open(complex_pdf) as pdf_doc:
        result_high_dpi = page_to_image(pdf_doc, 0, 300)
        assert isinstance(result_high_dpi, np.ndarray)
        assert result_high_dpi.shape[2] == 3

    # Test with low DPI
    with pymupdf.open(complex_pdf) as pdf_doc:
        result_low_dpi = page_to_image(pdf_doc, 0, 72)
        assert isinstance(result_low_dpi, np.ndarray)
        assert result_low_dpi.shape[2] == 3

    # High DPI should produce larger images
    assert (result_high_dpi.shape[0] * result_high_dpi.shape[1]) > (
        result_low_dpi.shape[0] * result_low_dpi.shape[1]
    )


def test_viz_chunks_with_different_chunk_types():
    # Test visualization with all available chunk types
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    chunks = []
    y_positions = [0.1, 0.3, 0.5, 0.7]

    for i, chunk_type in enumerate(ChunkType):
        if i >= len(y_positions):
            break

        chunk = Chunk(
            text=f"Test {chunk_type.value}",
            chunk_type=chunk_type,
            chunk_id=f"chunk_{i}",
            grounding=[
                ChunkGrounding(
                    page=0,
                    box=ChunkGroundingBox(
                        l=0.1, t=y_positions[i], r=0.9, b=y_positions[i] + 0.1
                    ),
                )
            ],
        )
        chunks.append(chunk)

    # Test visualization
    result = viz_chunks(img, chunks)
    assert isinstance(result, np.ndarray)
    assert result.shape == (200, 200, 3)


def test_crop_image_boundary_conditions():
    # Test cropping with boundary conditions
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image

    # Test cropping the entire image
    bbox_full = ChunkGroundingBox(l=0.0, t=0.0, r=1.0, b=1.0)
    crop_full = _crop_image(img, bbox_full)
    assert crop_full.shape == (100, 100, 3)

    # Test cropping a very small area
    bbox_small = ChunkGroundingBox(l=0.49, t=0.49, r=0.51, b=0.51)
    crop_small = _crop_image(img, bbox_small)
    assert crop_small.shape[0] >= 1 and crop_small.shape[1] >= 1
    assert crop_small.shape[2] == 3


def test_crop_image_coordinate_clamping():
    # Test coordinate clamping for out-of-bounds coordinates
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image

    # Test with negative coordinates - should be clamped to 0
    bbox_negative = ChunkGroundingBox(l=-0.1, t=-0.2, r=0.5, b=0.5)
    crop_negative = _crop_image(img, bbox_negative)
    assert isinstance(crop_negative, np.ndarray)
    assert crop_negative.shape[2] == 3
    assert crop_negative.shape[0] > 0 and crop_negative.shape[1] > 0

    # Test with coordinates > 1 - should be clamped to 1
    bbox_over_one = ChunkGroundingBox(l=0.5, t=0.5, r=1.2, b=1.3)
    crop_over_one = _crop_image(img, bbox_over_one)
    assert isinstance(crop_over_one, np.ndarray)
    assert crop_over_one.shape[2] == 3
    assert crop_over_one.shape[0] > 0 and crop_over_one.shape[1] > 0

    # Test with mixed invalid coordinates
    bbox_mixed = ChunkGroundingBox(l=-0.5, t=0.2, r=1.5, b=0.8)
    crop_mixed = _crop_image(img, bbox_mixed)
    assert isinstance(crop_mixed, np.ndarray)
    assert crop_mixed.shape[2] == 3
    assert crop_mixed.shape[0] > 0 and crop_mixed.shape[1] > 0

    # Test with all coordinates out of bounds (should still work)
    bbox_all_invalid = ChunkGroundingBox(l=-1.0, t=-1.0, r=2.0, b=2.0)
    crop_all_invalid = _crop_image(img, bbox_all_invalid)
    assert isinstance(crop_all_invalid, np.ndarray)
    assert crop_all_invalid.shape[2] == 3
    # Should crop the entire image when clamped
    assert crop_all_invalid.shape == (100, 100, 3)

    # Test with extreme values that result in valid crops after clamping
    bbox_extreme = ChunkGroundingBox(l=-999.0, t=-500.0, r=0.5, b=1000.0)
    crop_extreme = _crop_image(img, bbox_extreme)
    assert isinstance(crop_extreme, np.ndarray)
    assert crop_extreme.shape[2] == 3
    assert crop_extreme.shape[0] > 0 and crop_extreme.shape[1] > 0

    # Test edge case where clamping results in zero-size crop (top == bottom)
    bbox_zero_height = ChunkGroundingBox(
        l=0.2, t=500.0, r=0.8, b=600.0
    )  # Both t and b clamp to 1.0
    crop_zero_height = _crop_image(img, bbox_zero_height)
    assert isinstance(crop_zero_height, np.ndarray)
    assert crop_zero_height.shape[2] == 3
    # May have zero height when top == bottom after clamping
    assert crop_zero_height.shape[0] >= 0 and crop_zero_height.shape[1] > 0

    # Test edge case where clamping results in zero-size crop (left == right)
    bbox_zero_width = ChunkGroundingBox(
        l=500.0, t=0.2, r=600.0, b=0.8
    )  # Both l and r clamp to 1.0
    crop_zero_width = _crop_image(img, bbox_zero_width)
    assert isinstance(crop_zero_width, np.ndarray)
    assert crop_zero_width.shape[2] == 3
    # May have zero width when left == right after clamping
    assert crop_zero_width.shape[0] > 0 and crop_zero_width.shape[1] >= 0


def test_save_groundings_as_images_with_empty_chunks(temp_dir):
    # Test saving groundings when there are no chunks
    img_path = temp_dir / "test.jpg"
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    img.save(img_path)

    save_dir = temp_dir / "groundings"
    chunks = []  # Empty list

    with patch("agentic_doc.utils.get_file_type", return_value="image"):
        result = save_groundings_as_images(img_path, chunks, save_dir)

        # Should return empty dict for empty chunks
        assert result == {}


def test_viz_parsed_document_with_no_chunks(temp_dir):
    # Test visualization with a document that has no chunks
    img_path = temp_dir / "test_empty.png"
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    img.save(img_path)

    # Create a document with no chunks
    empty_doc = ParsedDocument(
        markdown="", chunks=[], start_page_idx=0, end_page_idx=0, doc_type="image"
    )

    with patch("agentic_doc.utils.get_file_type", return_value="image"):
        images = viz_parsed_document(img_path, empty_doc)

        # Should still return an image even with no chunks
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)


def test_log_retry_failure_with_different_attempt_numbers(monkeypatch):
    # Test retry logging with different attempt numbers

    # Mock retry state for different attempt numbers
    for attempt_num in [1, 5, 10, 50]:
        retry_state = MagicMock()
        retry_state.attempt_number = attempt_num
        outcome = MagicMock()
        outcome.failed = True
        outcome.exception.return_value = Exception(f"Error on attempt {attempt_num}")
        retry_state.outcome = outcome
        retry_state.fn = MagicMock()
        retry_state.fn.__name__ = "test_function"

        # Set retry logging style
        settings.retry_logging_style = "log_msg"

        # Should not raise an exception regardless of attempt number
        log_retry_failure(retry_state)


def test_get_chunk_from_reference():
    chunks = [
        {
            "text": "Name: Bob Johnson",
            "grounding": [
                {
                    "page": 0,
                    "box": {"l": 0.1, "t": 0.1, "r": 0.9, "b": 0.2},
                }
            ],
            "chunk_type": "text",
            "chunk_id": "1",
        },
        {
            "text": "Name: Alice Smith",
            "grounding": [
                {
                    "page": 1,
                    "box": {"l": 0.2, "t": 0.2, "r": 0.8, "b": 0.3},
                }
            ],
            "chunk_type": "text",
            "chunk_id": "2",
        },
    ]

    result = get_chunk_from_reference("1", chunks)
    assert result["text"] == "Name: Bob Johnson"
    assert get_chunk_from_reference("999", chunks) == None


def test_pdf_color_space_conversion(temp_dir):
    """Test that PDF-derived RGB images are correctly converted to BGR for saving."""
    # Create a simple PDF with red colored content using reportlab
    pdf_path = temp_dir / "colored_test.pdf"
    
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import red
    from reportlab.lib.pagesizes import letter
    
    # Create a simple PDF with a red rectangle
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    
    # Draw a red rectangle in the middle of the page
    c.setFillColor(red)
    rect_x = width * 0.3  
    rect_y = height * 0.4
    rect_width = width * 0.4
    rect_height = height * 0.2
    c.rect(rect_x, rect_y, rect_width, rect_height, fill=1, stroke=0)
    c.save()
    
    # Create a chunk that targets the red rectangle area
    chunk = Chunk(
        text="Red Rectangle",
        chunk_type=ChunkType.text,
        chunk_id="test_color",
        grounding=[
            ChunkGrounding(
                page=0, 
                # Target the area where we placed the red rectangle
                box=ChunkGroundingBox(l=0.3, t=0.4, r=0.7, b=0.6)
            )
        ],
    )
    
    # Test the actual save_groundings_as_images function without mocking pymupdf.open
    result = save_groundings_as_images(pdf_path, [chunk], temp_dir)
    
    # Check that the file was saved
    assert "test_color" in result
    saved_file = result["test_color"][0]
    assert saved_file.exists()
    
    # Read back the saved image and verify color space conversion
    saved_img = cv2.imread(str(saved_file))
    assert saved_img is not None
    
    # Look for red pixels in BGR format (where red = [0, 0, 255])
    # Since it's a PDF-derived image, we expect the RGB->BGR conversion to work correctly
    red_pixels = np.all(saved_img == [0, 0, 255], axis=2)
    red_pixel_count = np.sum(red_pixels)
    
    # Also check for "reddish" pixels (high red component, low others) to account for anti-aliasing
    red_mask = (saved_img[:, :, 2] > 200) & (saved_img[:, :, 1] < 100) & (saved_img[:, :, 0] < 100)
    reddish_count = np.sum(red_mask)
    
    # The test passes if we find red pixels, confirming correct RGB->BGR conversion
    assert red_pixel_count > 1000 or reddish_count > 1000, f"Expected to find red pixels indicating correct color space conversion, but found {red_pixel_count} pure red and {reddish_count} reddish pixels"


def test_image_color_space_preservation(temp_dir):
    """Test that image files maintain correct colors when saved through save_groundings_as_images."""
    # Create a test image with known red color using PIL
    img_path = temp_dir / "red_test_image.png"
    
    # Create a 200x200 image with a red rectangle in the center
    from PIL import Image, ImageDraw
    
    # Create white background
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))  # White background
    draw = ImageDraw.Draw(img)
    
    # Draw a red rectangle in the center
    rect_coords = [60, 80, 140, 120]  # x1, y1, x2, y2
    draw.rectangle(rect_coords, fill=(255, 0, 0))  # Pure red rectangle
    
    # Save the image
    img.save(img_path)
    
    # Create a chunk that targets the red rectangle area (normalized coordinates)
    chunk = Chunk(
        text="Red Rectangle",
        chunk_type=ChunkType.text,
        chunk_id="test_image_color",
        grounding=[
            ChunkGrounding(
                page=0,
                # Target the red rectangle area (60,80,140,120) normalized to (200,200) image
                box=ChunkGroundingBox(l=0.3, t=0.4, r=0.7, b=0.6)
            )
        ],
    )
    
    # Test save_groundings_as_images with the image file
    result = save_groundings_as_images(img_path, [chunk], temp_dir)
    
    # Check that the file was saved
    assert "test_image_color" in result
    saved_file = result["test_image_color"][0]
    assert saved_file.exists()
    
    # Read back the saved image and verify colors are preserved
    saved_img = cv2.imread(str(saved_file))
    assert saved_img is not None
    
    # Look for red pixels in BGR format (where red = [0, 0, 255])
    # For image files, cv2.imread reads in BGR format and cv2.imencode expects BGR format,
    # so colors should be preserved correctly
    red_pixels = np.all(saved_img == [0, 0, 255], axis=2)
    red_pixel_count = np.sum(red_pixels)
    
    # Also check for "reddish" pixels (high red component, low others) to account for compression artifacts
    red_mask = (saved_img[:, :, 2] > 200) & (saved_img[:, :, 1] < 100) & (saved_img[:, :, 0] < 100)
    reddish_count = np.sum(red_mask)
    
    # The test passes if we find red pixels, confirming correct color preservation
    assert red_pixel_count > 100 or reddish_count > 100, f"Expected to find red pixels indicating correct color preservation, but found {red_pixel_count} pure red and {reddish_count} reddish pixels"

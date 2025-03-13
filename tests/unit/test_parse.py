import pytest

from agentic_doc.parse import parse_and_save_documents


def test_parse_and_save_documents_with_invalid_file(sample_pdf_path, results_dir):
    # Arrange
    input_files = [
        sample_pdf_path.parent / "invalid.pdf",  # Non-existent file
        sample_pdf_path,
    ]

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        parse_and_save_documents(input_files, result_save_dir=results_dir)


def test_parse_and_save_documents_empty_list(results_dir):
    # Act
    result_paths = parse_and_save_documents([], result_save_dir=results_dir)

    # Assert
    assert result_paths == []

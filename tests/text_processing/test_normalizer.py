"""
Test cases for the TextNormalizer class.
"""
import pytest
from src.text_processing.normalizer import TextNormalizer

@pytest.fixture
def normalizer():
    """Create a TextNormalizer instance for testing."""
    return TextNormalizer()

def test_basic_normalization(normalizer):
    """Test basic text normalization."""
    input_text = "Hello World! This is a TEST."
    expected = "hello world! this is a test"
    assert normalizer.normalize_text(input_text) == expected

def test_abbreviation_expansion(normalizer):
    """Test if abbreviations are correctly expanded."""
    input_text = "Mr. Smith lives on St. John's Dr."
    expected = "mister smith lives on street johns doctor"
    assert normalizer.normalize_text(input_text) == expected

def test_number_conversion(normalizer):
    """Test if numbers are converted to words."""
    input_text = "I have 2 apples and 5 oranges"
    expected = "i have two apples and five oranges"
    assert normalizer.normalize_text(input_text) == expected

def test_sentence_splitting(normalizer):
    """Test sentence splitting functionality."""
    input_text = "Hello! How are you? I am fine."
    expected = ["Hello", "How are you", "I am fine"]
    assert normalizer.split_into_sentences(input_text) == expected

def test_empty_input(normalizer):
    """Test handling of empty input."""
    assert normalizer.normalize_text("") == ""
    assert normalizer.split_into_sentences("") == []

def test_special_characters(normalizer):
    """Test removal of special characters."""
    input_text = "Hello@World#$%^&*()"
    expected = "hello world"
    assert normalizer.normalize_text(input_text) == expected

def test_multiple_spaces(normalizer):
    """Test handling of multiple spaces."""
    input_text = "Hello    World   !  "
    expected = "hello world !"
    assert normalizer.normalize_text(input_text) == expected

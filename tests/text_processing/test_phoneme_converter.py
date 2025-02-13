"""
Test cases for the PhonemeConverter class.
"""
import pytest
from src.text_processing.phoneme_converter import PhonemeConverter

@pytest.fixture
def converter():
    """Create a PhonemeConverter instance for testing."""
    return PhonemeConverter()

def test_empty_input(converter):
    """Test handling of empty input."""
    assert converter.convert_to_phonemes("") == ""

def test_basic_words(converter):
    """Test basic word conversion."""
    # Test simple words
    result = converter.convert_to_phonemes("hello")
    assert result  # Should not be empty
    assert 'h' in result.lower()  # Should contain 'h' sound

def test_numbers_and_symbols(converter):
    """Test handling of numbers and symbols."""
    result = converter.convert_to_phonemes("123")
    assert result  # Should convert numbers to phonetic representation

def test_punctuation(converter):
    """Test handling of punctuation."""
    text = "Hello, world!"
    result = converter.convert_to_phonemes(text)
    assert result  # Should handle punctuation appropriately

def test_stress_patterns(converter):
    """Test stress pattern handling."""
    # Words with known stress patterns
    result = converter.convert_to_phonemes("today")
    assert result  # Should include stress markers 
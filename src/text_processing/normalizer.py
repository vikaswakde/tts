"""
Text normalizer for TTS system.
Handles text cleaning, number conversion, and basic normalization.
"""
import re
from typing import List, Optional

class TextNormalizer:
    def __init__(self):
        # Common abbreviations and their expansions
        self.abbreviations = {
            "mr.": "mister",
            "mrs.": "missus",
            "dr.": "doctor",
            "st.": "street",
            "vs.": "versus",
            "etc.": "etcetera",
        }
        
        # Common contractions and possessives
        self.contractions = {
            "'s": "s",   # Change from "" to "s" to preserve possessive
            "'t": "t",  # don't -> dont
            "'re": "re",  # you're -> youre
            "'ll": "ll",  # you'll -> youll
            "'ve": "ve",  # could've -> couldve
            "'m": "m",   # I'm -> im
        }
        
        # Numbers to words mapping (basic)
        self.numbers = {
            "0": "zero", "1": "one", "2": "two", "3": "three",
            "4": "four", "5": "five", "6": "six", "7": "seven",
            "8": "eight", "9": "nine", "10": "ten"
        }

    def normalize_text(self, text: str) -> str:
        """
        Main function to normalize text.
        Args:
            text: Input text string
        Returns:
            Normalized text string
        """
        if not text:
            return ""
            
        # Step 1: Convert to lowercase
        text = text.lower()
        
        # Step 2: Handle contractions and possessives first
        for contraction, replacement in self.contractions.items():
            text = text.replace(contraction, replacement)
        
        # Step 3: Expand abbreviations
        words = text.split()
        processed_words = []
        for word in words:
            # Check if this word is in our abbreviations
            word_lower = word.lower()
            if word_lower in self.abbreviations:
                processed_words.append(self.abbreviations[word_lower])
            else:
                # Remove any trailing periods that aren't part of abbreviations
                if word.endswith('.') and word_lower not in self.abbreviations:
                    word = word[:-1]
                processed_words.append(word)
        
        text = ' '.join(processed_words)
        
        # Step 4: Handle special characters
        # Replace special characters with space, except essential punctuation
        text = re.sub(r'[^a-z0-9\s.,!?-]', ' ', text)
        
        # Step 5: Convert numbers
        text = self._convert_numbers(text)
        
        # Step 6: Clean up punctuation
        text = text.rstrip('.')
        
        # Step 7: Clean up whitespace and join words
        text = ' '.join(text.split())
        
        return text.strip()

    def _convert_numbers(self, text: str) -> str:
        """Convert numbers to their word representation."""
        words = text.split()
        converted = []
        
        for word in words:
            # If word is a number and in our dictionary
            if word.isdigit() and word in self.numbers:
                converted.append(self.numbers[word])
            else:
                converted.append(word)
                
        return " ".join(converted)

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        Args:
            text: Input text
        Returns:
            List of sentences
        """
        # Basic sentence splitting on punctuation
        sentences = re.split(r'[.!?]+', text)
        # Remove empty sentences and strip whitespace
        return [s.strip() for s in sentences if s.strip()]

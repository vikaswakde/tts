"""
PhonemeConverter for TTS system.
Handles conversion of text to phonemes using a hybrid approach:
1. Uses phonemizer library for initial conversion
2. Applies custom rules and optimizations
"""
from typing import List, Dict, Union
import re
from phonemizer import phonemize

class PhonemeConverter:
    def __init__(self):
        # Basic English phoneme mappings
        self.phoneme_map = {
            # Vowels
            'AA': 'ɑ',  # odd
            'AE': 'æ',  # at
            'AH': 'ʌ',  # hut
            'AO': 'ɔ',  # caught
            'AW': 'aʊ', # cow
            'AY': 'aɪ', # hide
            'EH': 'ɛ',  # red
            'ER': 'ɝ',  # hurt
            'EY': 'eɪ', # say
            'IH': 'ɪ',  # it
            'IY': 'i',  # eat
            'OW': 'oʊ', # go
            'OY': 'ɔɪ', # toy
            'UH': 'ʊ',  # put
            'UW': 'u',  # too

            # Consonants
            'B': 'b',   # be
            'CH': 'tʃ', # cheese
            'D': 'd',   # dee
            'DH': 'ð',  # thee
            'F': 'f',   # fee
            'G': 'g',   # green
            'HH': 'h',  # he
            'JH': 'dʒ', # jee
            'K': 'k',   # key
            'L': 'l',   # lee
            'M': 'm',   # me
            'N': 'n',   # knee
            'NG': 'ŋ',  # ping
            'P': 'p',   # pee
            'R': 'r',   # read
            'S': 's',   # sea
            'SH': 'ʃ',  # she
            'T': 't',   # tea
            'TH': 'θ',  # theta
            'V': 'v',   # vee
            'W': 'w',   # we
            'Y': 'j',   # yield
            'Z': 'z',   # zee
            'ZH': 'ʒ',  # seizure
        }

        # Stress patterns for common word endings
        self.stress_patterns = {
            r'tion$': -2,    # attention -> at-TEN-tion
            r'sion$': -2,    # division -> di-VI-sion
            r'ture$': -2,    # nature -> NA-ture
            r'ate$': -2,     # activate -> AC-ti-vate
            r'ize$': -2,     # realize -> RE-a-lize
            r'ful$': -3,     # beautiful -> BEAU-ti-ful
            r'ing$': -3,     # interesting -> IN-ter-est-ing
            r'ly$': -3,      # completely -> com-PLETE-ly
            r'sion$': -2,     # confusion -> con-FU-sion
            r'ture$': -2,     # furniture -> FUR-ni-ture
        }

    def convert_to_phonemes(self, text: str) -> str:
        """
        Convert text to phonemes.
        Args:
            text: Input text string
        Returns:
            Phonetic representation of the text
        """
        if not text:
            return ""

        # Step 1: Use phonemizer for initial conversion
        phonemes = phonemize(
            text,
            language='en-us',
            backend='espeak',
            strip=True,
            preserve_punctuation=True,
            with_stress=True
        )

        # Handle different return types from phonemizer
        if isinstance(phonemes, (list, tuple)):
            phonemes = ' '.join(str(p) for p in phonemes)
        phonemes = str(phonemes)

        # Step 2: Apply custom rules and cleanup
        phonemes = self._apply_custom_rules(phonemes)
        
        return phonemes.strip()

    def _apply_custom_rules(self, phonemes: str) -> str:
        """
        Apply custom phoneme rules and cleanup.
        Args:
            phonemes: Raw phoneme string
        Returns:
            Processed phoneme string
        """
        # Remove extra spaces
        phonemes = ' '.join(phonemes.split())
        
        # Apply stress markers
        phonemes = self._apply_stress_markers(phonemes)
        
        return phonemes

    def _apply_stress_markers(self, phonemes: str) -> str:
        """
        Apply stress markers to vowel phonemes.
        Args:
            phonemes: Input phoneme string
        Returns:
            Phonemes with stress markers
        """
        # Split into words
        words = phonemes.split()
        marked_words = []

        for word in words:
            # Skip punctuation
            if not any(c.isalpha() for c in word):
                marked_words.append(word)
                continue

            # Find vowel sounds
            vowel_positions = [
                i for i, char in enumerate(word)
                if char in 'ɑæʌɔaʊaɪɛɝeɪɪioʊɔɪʊu'
            ]

            if not vowel_positions:
                marked_words.append(word)
                continue

            # Apply primary stress to the first vowel by default
            # unless word matches a specific pattern
            stress_pos = vowel_positions[0]
            
            # Check for specific word endings
            for pattern, stress_offset in self.stress_patterns.items():
                if re.search(pattern, word):
                    try:
                        stress_pos = vowel_positions[stress_offset]
                    except IndexError:
                        # If pattern doesn't match word length, use default
                        pass

            # Insert stress marker
            word = word[:stress_pos] + 'ˈ' + word[stress_pos:]
            marked_words.append(word)

        return ' '.join(marked_words) 
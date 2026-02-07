"""
Dataset Text Cleaning Module
Reduces vocabulary size by removing rare Unicode characters, emojis, and special symbols
"""

import re
import unicodedata


class DatasetCleaner:
    """Clean text datasets to reduce vocabulary size"""

    def __init__(self, mode='balanced'):
        """
        Initialize cleaner with specified mode

        Args:
            mode: 'strict', 'balanced', or 'lenient'
                - strict: ASCII only (A-Z, a-z, 0-9, basic punctuation)
                - balanced: Extended ASCII + common accents (recommended)
                - lenient: Keep more Unicode, just remove emojis/rare symbols
        """
        self.mode = mode

    def clean_text(self, text, show_stats=True):
        """
        Clean text according to mode

        Args:
            text: Input text string
            show_stats: Whether to print statistics

        Returns:
            Cleaned text string
        """
        original_length = len(text)
        original_vocab = len(set(text))

        if self.mode == 'strict':
            cleaned = self._clean_strict(text)
        elif self.mode == 'balanced':
            cleaned = self._clean_balanced(text)
        elif self.mode == 'lenient':
            cleaned = self._clean_lenient(text)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if show_stats:
            self._print_stats(text, cleaned, original_vocab)

        return cleaned

    def _clean_strict(self, text):
        """
        Strict cleaning: ASCII only
        Keeps: A-Z, a-z, 0-9, basic punctuation, whitespace
        """
        # Convert accented characters to ASCII equivalents
        text = self._remove_accents(text)

        # Keep only ASCII printable characters plus newlines/tabs
        allowed_chars = (
            'abcdefghijklmnopqrstuvwxyz'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            '0123456789'
            ' .,!?;:\'"()-[]{}/*@#$%&+=<>\n\t'
        )

        cleaned = ''.join(char if char in allowed_chars else ' ' for char in text)

        # Normalize whitespace
        cleaned = self._normalize_whitespace(cleaned)

        return cleaned

    def _clean_balanced(self, text):
        """
        Balanced cleaning: Extended ASCII
        Keeps: ASCII + common European accents
        Removes: Emojis, CJK characters, rare symbols
        """
        # Remove emojis
        text = self._remove_emojis(text)

        # Remove CJK (Chinese, Japanese, Korean) characters
        text = self._remove_cjk(text)

        # Remove control characters (except newline, tab)
        text = self._remove_control_chars(text)

        # Keep extended ASCII (0-255) which includes common European accents
        cleaned = ''.join(char if ord(char) < 256 else ' ' for char in text)

        # Normalize whitespace
        cleaned = self._normalize_whitespace(cleaned)

        return cleaned

    def _clean_lenient(self, text):
        """
        Lenient cleaning: Keep most Unicode
        Only removes: Emojis, rare symbols, control chars
        """
        # Remove emojis
        text = self._remove_emojis(text)

        # Remove control characters (except newline, tab)
        text = self._remove_control_chars(text)

        # Remove very rare Unicode blocks
        text = self._remove_rare_unicode(text)

        # Normalize whitespace
        cleaned = self._normalize_whitespace(cleaned)

        return cleaned

    def _remove_accents(self, text):
        """Convert accented characters to ASCII (é → e, ñ → n)"""
        # Normalize to NFD (decomposed form)
        nfd = unicodedata.normalize('NFD', text)
        # Remove combining characters (accents)
        return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')

    def _remove_emojis(self, text):
        """Remove emoji characters"""
        # Emoji Unicode ranges
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub(' ', text)

    def _remove_cjk(self, text):
        """Remove Chinese, Japanese, Korean characters"""
        # CJK Unicode ranges
        cjk_pattern = re.compile(
            "["
            "\u4E00-\u9FFF"      # CJK Unified Ideographs
            "\u3400-\u4DBF"      # CJK Extension A
            "\U00020000-\U0002A6DF"  # CJK Extension B
            "\uF900-\uFAFF"      # CJK Compatibility Ideographs
            "\u3040-\u309F"      # Hiragana
            "\u30A0-\u30FF"      # Katakana
            "\uAC00-\uD7AF"      # Hangul
            "]+",
            flags=re.UNICODE
        )
        return cjk_pattern.sub(' ', text)

    def _remove_control_chars(self, text):
        """Remove control characters except newline and tab"""
        return ''.join(
            char for char in text
            if char in '\n\t' or not unicodedata.category(char).startswith('C')
        )

    def _remove_rare_unicode(self, text):
        """Remove very rare Unicode blocks"""
        # Keep common blocks, replace rare ones with space
        return ''.join(
            char if ord(char) < 0x2000 or ord(char) > 0x2BFF else ' '
            for char in text
        )

    def _normalize_whitespace(self, text):
        """Normalize whitespace: collapse multiple spaces, remove trailing spaces"""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline (paragraph breaks)
        text = re.sub(r'\n\n+', '\n\n', text)
        # Remove spaces at line ends
        text = re.sub(r' +\n', '\n', text)
        # Remove spaces at line starts
        text = re.sub(r'\n +', '\n', text)
        return text.strip()

    def _print_stats(self, original, cleaned, original_vocab):
        """Print cleaning statistics"""
        cleaned_vocab = len(set(cleaned))
        chars_removed = len(original) - len(cleaned)
        vocab_reduction = original_vocab - cleaned_vocab

        print("\n" + "=" * 80)
        print("DATASET CLEANING STATISTICS")
        print("=" * 80)
        print(f"\nCleaning mode: {self.mode}")
        print(f"\nOriginal:")
        print(f"  Characters: {len(original):,}")
        print(f"  Vocabulary: {original_vocab:,} unique characters")
        print(f"\nCleaned:")
        print(f"  Characters: {len(cleaned):,}")
        print(f"  Vocabulary: {cleaned_vocab:,} unique characters")
        print(f"\nReduction:")
        print(f"  Characters removed: {chars_removed:,} ({chars_removed/len(original)*100:.2f}%)")
        print(f"  Vocabulary reduced: {vocab_reduction:,} ({vocab_reduction/original_vocab*100:.1f}%)")
        print(f"  Vocabulary size: {original_vocab:,} → {cleaned_vocab:,}")
        print("=" * 80 + "\n")


def should_clean_dataset(vocab_size, threshold=500):
    """
    Determine if dataset should be cleaned based on vocabulary size

    Args:
        vocab_size: Number of unique characters in dataset
        threshold: Vocabulary size threshold (default: 500)

    Returns:
        bool: True if vocab_size exceeds threshold
    """
    return vocab_size > threshold


def get_cleaning_recommendation(vocab_size):
    """
    Get recommended cleaning mode based on vocabulary size

    Args:
        vocab_size: Number of unique characters

    Returns:
        tuple: (should_clean, recommended_mode, reason)
    """
    if vocab_size < 300:
        return False, None, "Vocabulary size is already optimal"
    elif vocab_size < 500:
        return False, None, "Vocabulary size is acceptable"
    elif vocab_size < 1000:
        return True, 'balanced', "Moderate vocabulary - balanced cleaning recommended"
    elif vocab_size < 3000:
        return True, 'balanced', "Large vocabulary - balanced cleaning recommended"
    else:
        return True, 'strict', "Very large vocabulary - strict cleaning recommended"


if __name__ == '__main__':
    # Test the cleaner
    test_text = """
    Hello world! 你好世界! 😊
    This is a test with accents: café, naïve, résumé
    Special symbols: ™ © † § ¶
    Emojis: 🎉 ❤️ 🚀
    Numbers and punctuation: 123-456-7890, test@email.com
    """

    print("Testing dataset cleaner...\n")

    for mode in ['strict', 'balanced', 'lenient']:
        print(f"\n{'='*80}")
        print(f"MODE: {mode.upper()}")
        print('='*80)

        cleaner = DatasetCleaner(mode=mode)
        cleaned = cleaner.clean_text(test_text, show_stats=True)

        print("Sample output:")
        print(cleaned[:200])
        print()

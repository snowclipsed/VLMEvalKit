import re

def text_to_int(text):
    # Handle non-string inputs
    if isinstance(text, (int, float)):
        if isinstance(text, float) and (text != text):  # NaN check
            return None
        return int(text) if 0 <= text <= 20 else None
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Dictionary for number words (0-20)
    number_words = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
        'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
        'nineteen': 19, 'twenty': 20
    }
    
    # Create regex pattern for word numbers
    word_pattern = r'\b(' + '|'.join(number_words.keys()) + r')\b'
    
    # Create regex pattern for digit numbers (0-20)
    digit_pattern = r'\b([0-9]|1[0-9]|20)\b'
    
    # Combined pattern - digit pattern first for priority
    combined_pattern = f'({digit_pattern}|{word_pattern})'
    
    # Find first match
    match = re.search(combined_pattern, text)
    
    if match:
        found = match.group()
        # Check if it's a digit
        if found.isdigit():
            return int(found)
        # Otherwise it's a word
        return number_words.get(found, None)
    
    return None
    
def safe_convert(pred):
    try:
        return text_to_int(pred)
    except:
        return None
import re
from typing import Union

def text2int(text: str) -> int:
    """
    Convert a number written in English into an integer.
    
    Supports numbers up to the trillions and negative numbers.
    For example:
        "two thousand and nineteen" -> 2019
        "forty-two" -> 42
        "negative seven hundred" -> -700

    Raises:
        ValueError: If an unrecognized word is encountered.
    """
    if isinstance(text, int):
        return text
    # Dictionaries for number conversion
    units = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "seventeen": 17, "eighteen": 18, "nineteen": 19
    }
    tens = {
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
    }
    scales = {
        "hundred": 100,
        "thousand": 1000,
        "million": 1000000,
        "billion": 1000000000,
        "trillion": 1000000000000
    }
    
    # Normalize the input: lowercase, replace hyphens and commas with space.
    text = text.lower().replace('-', ' ').replace(',', ' ')
    # Split the text into tokens and remove the word "and"
    tokens = [token for token in text.split() if token != "and"]
    
    if not tokens:
        raise ValueError("No number words found to convert.")
    
    # Handle negative numbers (e.g., "negative forty-two")
    sign = 1
    if tokens[0] in ("negative", "minus"):
        sign = -1
        tokens = tokens[1:]
        if not tokens:
            raise ValueError("Expected number words after sign indicator.")
    
    current = 0  # Current accumulated value for a segment.
    result = 0   # Overall result.
    for token in tokens:
        if token in units:
            current += units[token]
        elif token in tens:
            current += tens[token]
        elif token in scales:
            factor = scales[token]
            if factor == 100:
                # Multiply the current value by 100. If current is 0, assume 1.
                current = (current or 1) * factor
            else:
                # For scales greater than hundred, update the result and reset current.
                current = (current or 1) * factor
                result += current
                current = 0
        else:
            raise ValueError(f"Unrecognized number word: {token}")
    
    return sign * (result + current)

def int_from_string(value: Union[int, str]) -> int:
    """
    Convert a value into an integer handling several cases:
    
    1. If the input is an int, return it directly.
    2. If the input is a numeric string (e.g., "2019" or "1,234"), convert it.
    3. If the input string contains a number mixed with other text 
       (e.g., "The total is 2019 dollars"), extract and convert the first occurrence.
    4. If the input string represents a number in English words
       (e.g., "two thousand and nineteen"), convert it using text2int().
       
    Raises:
        ValueError: If conversion is not possible.
        TypeError: If the input is neither an int nor a str.
    """
    # Case 1: Already an integer.
    if isinstance(value, int):
        return value
    
    # Only strings are supported beyond ints.
    if not isinstance(value, str):
        raise TypeError("Input value must be an integer or string.")
    
    text = value.strip()
    
    # Case 2: Try direct conversion (handles digits and comma-delimited numbers)
    try:
        normalized = text.replace(',', '')
        return int(normalized)
    except ValueError:
        pass
    
    # Case 3: Extract a digit sequence from text
    match = re.search(r'-?\d+', text)
    if match:
        try:
            return int(match.group())
        except ValueError:
            pass
    
    # Case 4: Fallback to converting English words to an integer.
    try:
        return text2int(text)
    except Exception as e:
        raise ValueError(f"Cannot convert {value!r} to an integer.") from e

# Example usage and testing:
if __name__ == "__main__":
    test_cases = [
        "two thousand and nineteen", 
        "2019",                      
        2019,                        
        "The total is 2019 dollars"  
    ]
    
    for case in test_cases:
        try:
            result = int_from_string(case)
            print(f"{case!r} -> {result}")
        except Exception as err:
            print(f"{case!r} -> Error: {err}")
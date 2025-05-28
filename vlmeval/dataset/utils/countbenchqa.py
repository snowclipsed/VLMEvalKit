import re


def text_to_int(text):
    """
    Detect the first number in a text string and return its integer value.
    Only supports numbers from 0 to 20, both in digit and word form.
    """

    patterns = {
        **{str(i): i for i in range(21)},
        **dict(
            zip(
                [
                    "zero",
                    "one",
                    "two",
                    "three",
                    "four",
                    "five",
                    "six",
                    "seven",
                    "eight",
                    "nine",
                    "ten",
                    "eleven",
                    "twelve",
                    "thirteen",
                    "fourteen",
                    "fifteen",
                    "sixteen",
                    "seventeen",
                    "eighteen",
                    "nineteen",
                    "twenty",
                ],
                range(21),
            )
        ),
    }
    pattern = r"\b(" + "|".join(patterns.keys()) + r")\b"
    match = re.search(pattern, text.lower())

    return patterns.get(match.group(1)) if match else None

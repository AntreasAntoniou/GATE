from typing import Any


def letter_to_index(letter):
    """
    Converts a letter to its index in the English alphabet.

    Args:
      letter: A letter.

    Returns:
      The index of the letter in the English alphabet.
    """

    letter = letter.lower()
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    index = alphabet.index(letter)
    return index


class RACETask:
    def __init__(self):
        super().__init__()

    def __call__(self, inputs) -> Any:
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            Any: _description_

        .. code-block:: python
        An input looks like this before processing
        {
        'example_id': 'middle4558.txt',
        'article': '"I planted a seed. Finally grow fruits. Today is a great day. Pick off the star for you. Pick off the moon for you. Let it rise for you every day. Become candles burning myself. Just light you up, hey!... You are my little little apple. How much I love you, still no enough."\nThis words are from the popular song You Are My Little Dear Apple. Bae Seul-Ki acted as the leading dancer in the MV of the song. She loves dancing. She became crazy about hip-hop when she was a school girl.\nBai Seul-Ki was born on September 27, 1986. She is a South Korean singer and dancer. She is 168cm tall. She loves cooking. Her favourite food is spicy and salty. She like pink and red most. There are five members in her family---father, mother, two younger brothers and herself. She isn\'t married.\nAfter her father and mother broke up, she lived with her mother and new daddy. She enjoys being alone.',
        'answer': 'B',
        'question': 'Bae Seul-Ki   _   in the MV of the song according to the passage.',
        'options': ['sang', 'danced', 'cried', 'laughed']
        }
        """
        return {
            "text": {
                "article": inputs["article"],
                "question": inputs["question"],
                "options": inputs["options"],
            },
            "labels": letter_to_index(inputs["answer"]),
        }


class LambadaTask:
    def __init__(self):
        super().__init__()
        self.domains = {
            "Historical": 0,
            "Young_Adult": 1,
            "Other": 2,
            "Literature": 3,
            "Mystery": 4,
            "New_Adult": 5,
            "Adventure": 6,
            "Science_fiction": 7,
            "Horror": 8,
            "Humor": 9,
            "Themes": 10,
            "Thriller": 11,
            "Fantasy": 12,
            "Teen": 13,
            "Vampires": 14,
            "Romance": 15,
        }

    def __call__(self, inputs) -> Any:
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            Any: _description_

        .. code-block:: python
        An input looks like this before processing
        {
        'example_id': 'middle4558.txt',
        'article': '"I planted a seed. Finally grow fruits. Today is a great day. Pick off the star for you. Pick off the moon for you. Let it rise for you every day. Become candles burning myself. Just light you up, hey!... You are my little little apple. How much I love you, still no enough."\nThis words are from the popular song You Are My Little Dear Apple. Bae Seul-Ki acted as the leading dancer in the MV of the song. She loves dancing. She became crazy about hip-hop when she was a school girl.\nBai Seul-Ki was born on September 27, 1986. She is a South Korean singer and dancer. She is 168cm tall. She loves cooking. Her favourite food is spicy and salty. She like pink and red most. There are five members in her family---father, mother, two younger brothers and herself. She isn\'t married.\nAfter her father and mother broke up, she lived with her mother and new daddy. She enjoys being alone.',
        'answer': 'B',
        'question': 'Bae Seul-Ki   _   in the MV of the song according to the passage.',
        'options': ['sang', 'danced', 'cried', 'laughed']
        }
        """
        if inputs["domain"] != None:
            return {
                "text": inputs["text"],
                "labels": self.domains[inputs["domain"]],
            }
        else:
            return {
                "text": inputs["text"],
                "labels": None,  # Test dataset does not have labels
            }

class LocalAgreement2:
    def __init__(self):
        self.confirmed = []  # To store the confirmed sequence of words
        self.unconfirmed = []  # To store the unconfirmed sequence of words

    def process_stream(self, new_words):
        # Find the longest matching prefix between current_words and _unconfirmed_words
        for a, con_word in reversed(list(enumerate(self.unconfirmed))):
            for b, new_word in reversed(list(enumerate(new_words))):
                if con_word == new_word:
                    # Now we maybe know where the last unconfirmed word is in the new words list
                    # We can now start from here and go backwards to find the common prefix
                    # find the common prefix
                    common_prefix = 0
                    temp_un_word_list = list(reversed(self.unconfirmed[:a + 1]))
                    temp_new_word_list = list(reversed(new_words[:b + 1]))
                    for i in range(min(len(temp_un_word_list), len(temp_new_word_list))):
                        if temp_un_word_list[i] == temp_new_word_list[i]:
                            common_prefix += 1
                        else:
                            break
                        
                    
                    # common_prefix has to be exactly the length of the unconfirmed word - processed unconfirmed
                    processed_unconfirmed = len(temp_un_word_list)
                    if common_prefix == processed_unconfirmed:
                        # We can now confirm all unconfirmed to a
                        self.confirmed.extend(self.unconfirmed[:a + 1])
                        self.unconfirmed = self.unconfirmed[a + 1:] + new_words[b + 1:]
                        return
                    else:
                        break
            
            # If we reach here, it means this unconfirmed word doesnt exist in the new words list
            # or the previous unconfirmed word changed. Remove it
            self.unconfirmed = self.unconfirmed[:a]
        
                    
        # Find the longest matching prefix between current_words and confirmed_words
        for a, con_word in reversed(list(enumerate(self.confirmed))):
            for b, new_word in reversed(list(enumerate(new_words))):
                if con_word == new_word:
                    # Now we maybe know where the last unconfirmed word is in the new words list
                    # We can now start from here and go backwards to find the common prefix
                    # find the common prefix
                    common_prefix = 0
                    temp_un_word_list = list(reversed(self.confirmed[:a + 1]))
                    temp_new_word_list = list(reversed(new_words[:b + 1]))
                    for i in range(min(len(temp_un_word_list), len(temp_new_word_list))):
                        if temp_un_word_list[i] == temp_new_word_list[i]:
                            common_prefix += 1
                        else:
                            break
            
                    if common_prefix > 2:
                        self.unconfirmed = new_words[b + 1:]
                        return
                    
        
        self.unconfirmed = new_words 

    def get_confirmed(self):
        """
        Returns the list of confirmed words.
        """
        return self.confirmed

    def get_unconfirmed(self):
        """
        Returns the list of unconfirmed words.
        """
        return self.unconfirmed
    
    def clear(self):
        self.confirmed = []
        self.unconfirmed = []


# Test the algorithm with the example stream
stream_processor = LocalAgreement2()

# Process each incoming stream of words
streams = [
    {"words": ["Hello", "my", "name"], "confirmed": [], "unconfirmed": ["Hello", "my", "name"]},
    {"words": ["Hello", "my", "name", "is", "Bob"], "confirmed": ["Hello", "my", "name"], "unconfirmed": ["is", "Bob"]},
    {"words": ["Hello", "my", "name", "is", "Bob", "and", "I", "am"], "confirmed": ["Hello", "my", "name", "is", "Bob"], "unconfirmed": ["and", "I", "am"]},
    {"words": ["Hello", "my", "name", "is", "Bob", "and", "I", "need", "to"], "confirmed": ["Hello", "my", "name", "is", "Bob", "and", "I"], "unconfirmed": ["need", "to"]},
    {"words": ["Hello", "my", "name", "is", "Bob", "and", "I", "need", "to", "go"], "confirmed": ["Hello", "my", "name", "is", "Bob", "and", "I", "need", "to"], "unconfirmed": ["go"]},
    {"words": ["Hello", "my", "name", "is", "Rob", "and", "I", "need", "to", "go", "to", "the"], "confirmed": ["Hello", "my", "name", "is", "Bob", "and", "I", "need", "to", "go"], "unconfirmed": ["to", "the"]},
    {"words": ["Hello", "my", "name", "is", "Bob", "and", "I", "need", "to", "go", "to", "a", "store"], "confirmed": ["Hello", "my", "name", "is", "Bob", "and", "I", "need", "to", "go", "to"], "unconfirmed": ["a", "store"]},
    {"words": ["Hello", "my", "name", "is", "Bob", "and", "I", "need", "to", "go", "to", "a", "store", "to", "buy"], "confirmed": ["Hello", "my", "name", "is", "Bob", "and", "I", "need", "to", "go", "to", "a", "store"], "unconfirmed": ["to", "buy"]},
                   {"words": ["name", "is", "Bob", "and", "I", "need", "to", "go", "to", "a", "store", "to", "buy", "some"], "confirmed": ["Hello", "my", "name", "is", "Bob", "and", "I", "need", "to", "go", "to", "a", "store", "to", "buy"], "unconfirmed": ["some"]},
                                 {"words": ["Bob", "and", "I", "need", "to", "go", "to", "a", "store", "to", "buy", "some", "food"], "confirmed": ["Hello", "my", "name", "is", "Bob", "and", "I", "need", "to", "go", "to", "a", "store", "to", "buy", "some"], "unconfirmed": ["food"]},
]

# Iterate through the streams to process
for words in streams:
    stream_processor.process_stream(words["words"])
    confirmed = stream_processor.get_confirmed()
    unconfirmed = stream_processor.get_unconfirmed()
    
    # Check if the confirmed and unconfirmed words match the expected values
    if confirmed == words["confirmed"] and unconfirmed == words["unconfirmed"]:
        print("Test Passed")
    else:
        print(f"Confirmed: {stream_processor.get_confirmed()}, Expected: {words['confirmed']}")
        print(f"Unconfirmed: {stream_processor.get_unconfirmed()}, Expected: {words['unconfirmed']}")
    print("---")


stream_processor.clear()

streams2 = [
    [],
    ['Hello,', 'everyone.'],
    ['.'],
    ['.'],
    ['.'],
    ['.'],
    ['.'],
    ['You', 'know,'],
    ['You', 'know,', 'in', 'the', 'future'],
    ['You', 'know,', 'in', 'the', 'future', 'I', 'think', 'emotional', 'intelligence'],
    ['You', 'know,', 'in', 'the', 'future,', 'I', 'think', 'emotional', 'intelligence'],
    ['You', 'know,', 'in', 'the', 'future,', 'I', 'think', 'emotional', 'intelligence', 'will', 'be', 'one'],
    ['You', 'know', 'in', 'the', 'future,', 'I', 'think', 'emotional', 'intelligence', 'will', 'be', 'one', 'of', 'several', 'abilities'],
    ['You', 'know,', 'in', 'the', 'future,', 'I', 'think', 'emotional', 'intelligence', 'will', 'be', 'one', 'of', 'several', 'abilities', 'that', 'we'],
    ['You', 'know,', 'in', 'the', 'future,', 'I', 'think', 'emotional', 'intelligence', 'will', 'be', 'one', 'of', 'several', 'abilities', 'that', 'we', 'need'],
    ['You', 'know,', 'in', 'the', 'future,', 'I', 'think', 'emotional', 'intelligence', 'will', 'be', 'one', 'of', 'several', 'abilities', 'that', 'we', 'need.', 'Another', 'of', 'course,'],
    ['In', 'the', 'future,', 'I', 'think', 'emotional', 'intelligence', 'will', 'be', 'one', 'of', 'several', 'abilities', 'that', 'we', 'need.', 'Another,', 'of', 'course,', 'is', 'cognitive', 'abilities.'],
    ['I', 'think', 'emotional', 'intelligence', 'will', 'be', 'one', 'of', 'several', 'abilities', 'that', 'we', 'need.', 'Another', 'of', 'course', 'is', 'cognitive', 'ability,', 'IQ,'],
    ['Intelligence', 'will', 'be', 'one', 'of', 'several', 'abilities', 'that', 'we', 'need.', 'Another,', 'of', 'course,', 'is', 'cognitive', 'ability,', 'IQ,', 'and', 'maybe', 'AI'],
    ['One', 'of', 'several', 'abilities', 'that', 'we', 'need.', 'Another', 'of', 'course', 'is', 'cognitive', 'ability,', 'IQ,', 'and', 'maybe', 'AI', 'will', 'take', 'over', 'more.'],
    ['Several', 'abilities', 'that', 'we', 'need.', 'Another', 'of', 'course', 'is', 'cognitive', 'ability,', 'IQ,', 'and', 'maybe', 'AI', 'will', 'take', 'over', 'more', 'and', 'more', 'of', 'that.'],
    ['that', 'we', 'need.', 'Another', 'of', 'course', 'is', 'cognitive', 'ability,', 'IQ,', 'and', 'maybe', 'AI', 'will', 'take', 'over', 'more', 'and', 'more', 'of', 'that.'],
    ['Another,', 'of', 'course,', 'is', 'cognitive', 'ability,', 'IQ,', 'and', 'maybe', 'AI', 'will', 'take', 'over', 'more', 'and', 'more', 'of', 'that.', 'However,', 'emotional', 'intelligence'],
    ['Another,', 'of', 'course,', 'is', 'cognitive', 'ability,', 'IQ,', 'and', 'maybe', 'AI', 'will', 'take', 'over', 'more', 'and', 'more', 'of', 'that.', 'However,', 'emotional', 'intelligence', 'is'],
    ["There's", 'cognitive', 'ability,', 'IQ,', 'and', 'maybe', 'AI', 'will', 'take', 'over', 'more', 'and', 'more', 'of', 'that.', 'However,', 'emotional', 'intelligence', 'is', 'a', 'human', 'ability'],
    ['The', 'IQ', 'and', 'maybe', 'AI', 'will', 'take', 'over', 'more', 'and', 'more', 'of', 'that.', 'However,', 'emotional', 'intelligence', 'is', 'a', 'human', 'ability', 'and', 'will'],
    ['Maybe', 'AI', 'will', 'take', 'over', 'more', 'and', 'more', 'of', 'that.', 'However,', 'emotional', 'intelligence', 'is', 'a', 'human', 'ability', 'and', 'will', 'always', 'remain', 'so.'],
    ['will', 'take', 'over', 'more', 'and', 'more', 'of', 'that.', 'However,', 'emotional', 'intelligence', 'is', 'a', 'human', 'ability', 'and', 'will', 'always', 'remain', 'so.'],
    ['or', 'more', 'of', 'that.', 'However,', 'emotional', 'intelligence', 'is', 'a', 'human', 'ability', 'and', 'will', 'always', 'remain', 'so.'],
    ['However,', 'emotional', 'intelligence', 'is', 'a', 'human', 'ability', 'and', 'will', 'always', 'remain', 'so.', 'IQ', 'predicts', 'how', 'well', "you'll"],
    ['Emotional', 'intelligence', 'is', 'a', 'human', 'ability', 'and', 'will', 'always', 'remain', 'so.', 'IQ', 'predicts', 'how', 'well', "you'll", 'do', 'in', 'your', 'school.'],
    ['Intelligence', 'is', 'a', 'human', 'ability', 'and', 'will', 'always', 'remain', 'so.', 'IQ', 'predicts', 'how', 'well', "you'll", 'do', 'in', 'your', 'school', 'years'],
    ['a', 'human', 'ability', 'and', 'will', 'always', 'remain', 'so.', 'IQ', 'predicts', 'how', 'well', "you'll", 'do', 'in', 'your', 'school', 'years', 'and', 'how', 'much', 'salary'],
    ['and', 'will', 'always', 'remain', 'so.', 'IQ', 'predicts', 'how', 'well', "you'll", 'do', 'in', 'your', 'school', 'years', 'and', 'how', 'much', 'salary', 'you', 'can', 'make', 'over', 'the', 'course'],
    ['always', 'remain', 'so.', 'IQ', 'predicts', 'how', 'well', "you'll", 'do', 'in', 'your', 'school', 'years', 'and', 'how', 'much', 'salary', 'you', 'can', 'make', 'over', 'the', 'course', 'of', 'a', 'career.']
]

for words2 in streams2:
    stream_processor.process_stream(words2)

    print(f"Words: {words2}")
    print(f"Confirmed: {stream_processor.get_confirmed()}")
    print(f"Unconfirmed: {stream_processor.get_unconfirmed()}")
    print("---")
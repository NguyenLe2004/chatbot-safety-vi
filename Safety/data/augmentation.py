import random
from typing import List, Callable

class SafetyAugmenter:
    def __init__(self, augmentation_rate: float = 0.1):
        self.rate = augmentation_rate

    def weighted_randint(self, low: int, high: int) -> int:
        numbers = list(range(low, high + 1))
        weights = [10 - (i - 1) for i in numbers]
        return random.choices(numbers, weights=weights, k=1)[0]

    def random_replace_similar_chars(self, text):
        similar_chars = {
            'l': 'I',
            'I': 'l',
            's': 'x',
            'x': 's',
            'y': 'i',
            'i': 'y'
        }
        
        if random.random() < self.rate:
            text = list(text)
            indices = random.sample(range(len(text)), self.weighted_randint(1, len(text) // 4))
            
            for idx in indices:
                if text[idx] in similar_chars:
                    text[idx] = similar_chars[text[idx]]
        
        return ''.join(text)

    def random_noise(self, text):
        if random.random() < self.rate:
            noise_char = '~!@#$%^&?><+-_=*`;:,./\\)(}{][\'\"'
            noise_alpha = 'abcdefghijklmnopqrstuvwxyg'
            noise_number = '1234567890'
            noise = noise_char + noise_alpha + noise_number
            
            noises = ''.join(random.choices(noise, k = self.weighted_randint(1, len(text)//4)))
            for noise in noises:
                index = random.randint(0, len(text))
                text = text[:index] + noise + text[index:]
        return text

    def random_newline(self, text):
        if random.random() < self.rate:
            words = text.split(" ")
            newline_num = min(self.weighted_randint(1,len(text)//4),len(words))
            choices = random.sample(words, newline_num) 
            for choice in choices:
                text = text.replace(choice,"\n" + choice, 1) 
        return text
        
    def random_crop_word(self, text):
        if random.random() < self.rate:
            words = text.split(" ")
            word_num = min(self.weighted_randint(1,len(text)//3),len(words))
            choices = random.sample(words, word_num) 
            for choice in choices:
                if len(choice) >1 :
                    num_crop = self.weighted_randint(1,len(choice)-1)
                    text = text.replace(choice, choice[:len(choice) - num_crop])
        return text
        
    def random_remove_space(self, text):
        num_space = min(self.weighted_randint(1,len(text)//3), text.count(' '))
        if random.random() < self.rate:
            words = text.split()
            indices_to_remove = random.sample(range(len(words) - 1), num_space)
            text = ''.join(words[i] + (' ' if i not in indices_to_remove else '') for i in range(len(words)))
        return text
            
    def random_cut_and_paste(self, text):
        if random.random() < self.rate:  
            words = text.split()
            if len(words) > 2:
                word_to_cut = random.choice(words)
                words.remove(word_to_cut)
                insert_position = random.randint(0, len(words))
                words.insert(insert_position, word_to_cut)
            return ' '.join(words)
        return text
    
def get_augmentation_functions(rate = 0.1) -> List[Callable]:
    augmenter = SafetyAugmenter( augmentation_rate = rate)
    return [
        augmenter.random_noise,
        augmenter.random_newline,
        augmenter.random_crop_word,
        augmenter.random_remove_space,
        augmenter.random_cut_and_paste,
        augmenter.random_replace_similar_chars
    ]
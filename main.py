import re
import sys
import math
import random
import pandas as pd
import names as names_gen
from pathlib import Path
from statistics import mean
from string import ascii_lowercase


class MinMaxNameLen:
    def __init__(self, names):
        self.min = 4
        self.max = max(len(name) for name in names)
        self.max += math.ceil(0.25 * self.max)  # 25% margem de segurança

    def match(self, str_):
        return self.min <= len(str_) <= self.max


class MinMaxPartNameLen:
    def __init__(self, names):
        part_lens = []
        for name in names:
            part_lens.extend(len(part_name) for part_name in name.split(' '))

        self.min = 2
        self.max = max(part_lens)
        self.max += math.ceil(0.25 * self.max)  # 25% margem de segurança

    def match(self, str_):
        part_lens = [len(part_str) for part_str in str_.split(' ')]
        return all(self.min <= part_len <= self.max for part_len in part_lens)


class MinMaxWordCount:
    def __init__(self, names):
        word_count = [len(name.split(' ')) for name in names]
        self.min_count = 2
        self.max_count = max(word_count)
        self.max_count += math.ceil(0.25 * self.max_count)  # 25% margem de segurança

    def match(self, str_):
        return self.min_count <= len(str_.split(' ')) <= self.max_count


class CharacterSet:
    def __init__(self, lower_names):
        self.characters = set(ascii_lowercase)
        self.characters.add(' ')
        for lower_name in lower_names:
            for character in lower_name:
                self.characters.add(character)

    def match(self, str_):
        return all(character in self.characters for character in str_)


class NonNameWords:
    def __init__(self, name_db):
        part_names = set()
        for name in name_db:
            for part_name in name.split(' '):
                part_names.add(part_name)

        # https://en.wikipedia.org/wiki/Most_common_words_in_English
        self.non_name_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for', 'not', 'on',
                               'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we',
                               'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
                               'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when',
                               'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into',
                               'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now',
                               'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two',
                               'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any',
                               'these', 'give', 'day', 'most', 'us', 'person', 'last', 'long', 'thing', 'great', 'man',
                               'little', 'world', 'own', 'life', 'hand', 'old', 'part', 'right', 'child', 'big', 'eye',
                               'high', 'woman', 'different', 'place', 'small', 'large', 'week', 'find', 'next', 'case',
                               'tell', 'early', 'point', 'ask', 'young', 'government', 'important', 'company', 'seem',
                               'few', 'number', 'feel', 'public', 'group', 'try', 'bad', 'problem', 'leave', 'same',
                               'fact', 'call', 'able', 'is', 'was', 'are', 'had', 'word', 'were', 'said', 'each',
                               'many', 'has', 'more', 'write', 'water', 'been', 'oil', 'down', 'did', 'made', 'may'}

        self.non_name_words = self.non_name_words - part_names

    def match(self, str_):
        return all(str_part not in self.non_name_words for str_part in str_.split(' '))


class ProbabilisticMinMaxPartNameLen:
    def __init__(self, names):
        part_lens = []
        for name in names:
            part_lens.extend(len(part_name) for part_name in name.split(' '))

        self.min = 2
        self.max = max(part_lens)

    def compute_probability(self, _, str_lower):
        parts = str_lower.split(' ')
        parts_found = 0
        for part in parts:
            if self.min <= len(part) <= self.max:
                parts_found += 1

        return parts_found * (1 / len(parts))


class ProbabilisticOccurrence:
    def __init__(self, name_db):
        self.part_occurrences = set()
        for name in name_db:
            for part_name in name.split(' '):
                self.part_occurrences.add(part_name)

    def compute_probability(self, _, str_lower):
        parts = str_lower.split(' ')
        parts_found = 0
        for part in parts:
            if part in self.part_occurrences:
                parts_found += 1

        return parts_found * (1 / len(parts))


class NameDetector:
    def __init__(self, name_db):
        self.name_db = set(re.sub(' +', ' ', name).lower() for name in name_db)

        self.statistic_rules = [
            MinMaxNameLen(self.name_db),
            MinMaxPartNameLen(self.name_db),
            MinMaxWordCount(self.name_db),
            CharacterSet(self.name_db),
            NonNameWords(self.name_db)
        ]

        self.probabilistic_rules = [
            ProbabilisticMinMaxPartNameLen(self.name_db),
            ProbabilisticOccurrence(self.name_db),
        ]

    def compute_probability(self, str_):
        str_ = re.sub(' +', ' ', str_)
        str_lower = str_.lower()

        if str_lower in self.name_db:
            return 1

        for statistic_rule in self.statistic_rules:
            if not statistic_rule.match(str_lower):
                return 0

        probabilities = []

        for probabilistic_rule in self.probabilistic_rules:
            probability = probabilistic_rule.compute_probability(str_, str_lower)
            if probability is not None:
                probabilities.append(probability)

        return round(mean(probabilities), 2)


def main(argv):
    # params
    RND_SEED = 99
    RND_NAMES_SIZE = 20000
    THRESHOLD = 0.5

    src_dir = Path(__file__).resolve().parent
    random.seed(RND_SEED)

    print('## Generating a database of random names... ##')
    name_db = []
    for i in range(RND_NAMES_SIZE):
        if (i % 500) == 0:
            print(f'{i} of {RND_NAMES_SIZE}')
        name_db.append(names_gen.get_full_name())
    print('Done\n')

    name_detector = NameDetector(name_db)

    if len(argv) > 1:
        for sentence in argv[1:]:
            probability = name_detector.compute_probability(sentence)
            print(f'Probability of "{sentence}" be a name={probability}')
        return

    print('## Evaluating test_database.csv... ##')
    test_db = pd.read_csv(src_dir / 'test_database.csv', sep=';')
    test_db['probability'] = -1
    test_db['predicted'] = -1



    test_total_count = len(test_db)
    test_names_count = len(test_db[test_db.name == 1])
    test_non_names_count = len(test_db[test_db.name == 0])

    for i, row in test_db.iterrows():
        if (i % 50) == 0:
            print(f'{i} of {test_total_count}')
        probability = name_detector.compute_probability(row.sentence)
        predicted = 1 if probability > THRESHOLD else 0
        test_db.at[i, 'probability'] = probability
        test_db.at[i, 'predicted'] = predicted

    print('Done\n')

    true_positives = test_db[(test_db.name == 1) & (test_db.predicted == 1)]
    true_negatives = test_db[(test_db.name == 0) & (test_db.predicted == 0)]
    false_positives = test_db[(test_db.name == 0) & (test_db.predicted == 1)]
    false_negatives = test_db[(test_db.name == 1) & (test_db.predicted == 0)]

    print('## Wrong cases ##')
    print('Names predicted as non name:')
    print(false_negatives.to_string())
    print()
    print('Non names predicted as name:')
    print(false_positives.to_string())
    print()

    print('## Metrics ##')
    print(f'True Positives={len(true_positives)}')
    print(f'True Negatives={len(true_negatives)}')
    print(f'False Positives={len(false_positives)}')
    print(f'False Negatives={len(false_negatives)}\n')

    general_accuracy = (len(true_positives) + len(true_negatives)) / test_total_count
    names_accuracy = len(true_positives) / test_names_count
    non_names_accuracy = len(true_negatives) / test_non_names_count
    print(f'General Accuracy={round(general_accuracy, 2)}')
    print(f'Names Accuracy={round(names_accuracy, 2)}')
    print(f'Non Names Accuracy={round(non_names_accuracy, 2)}\n')

    precision = len(true_positives) / (len(true_positives) + len(false_positives))
    recall = len(true_positives) / (len(true_positives) + len(false_negatives))
    f1 = 2 * ((precision * recall) / (precision + recall))

    print(f'Precision={round(precision, 2)}')
    print(f'Recall={round(recall, 2)}')
    print(f'F1={round(f1, 2)}\n')


if __name__ == '__main__':
    main(sys.argv)

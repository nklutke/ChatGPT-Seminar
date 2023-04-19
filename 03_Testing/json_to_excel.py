import jsonlines
import pandas as pd

file1 = '/Users/nicolaiklutke/Desktop/ChatGPTSeminar/03_Testing/predictions/instructgpt-babbage_predictions_de.jsonl'
file2 = '/Users/nicolaiklutke/Desktop/ChatGPTSeminar/03_Testing/predictions/fine_tuned_bloom_predictions_de2.jsonl'

data = []


def remove_illegal_chars(text):
    return ''.join(c for c in text if c.isprintable())


with jsonlines.open(file1) as reader1, jsonlines.open(file2) as reader2:
    for item1, item2 in zip(reader1, reader2):
        data.append({
            'prompt': remove_illegal_chars(item1['prompt']),
            'response_1': remove_illegal_chars(item1['response']),
            'response_2': remove_illegal_chars(item2['response']),
            'target': remove_illegal_chars(item1['target'])
        })


df = pd.DataFrame(data)
df.to_excel('/Users/nicolaiklutke/Desktop/ChatGPTSeminar/03_Testing/humaneval.xlsx', index=False)
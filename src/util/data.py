import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from keras.preprocessing.sequence import pad_sequences
from util.convenient_funcs import combine_sentence_to_segment
import regex


def process_data(docs_list: list, tokenizer, split_segment=True, segment_len=309):

    essays = docs_list
    essays = [regex.sub(pattern=r'@[A-Z]+[1-9]+', string=essay, repl='MASK') for essay in essays]

    if split_segment:
        essays = [combine_sentence_to_segment(sent_tokenize(essay), max_segment_len=segment_len) for essay
                  in essays]  # split into sentence
    else:
        essays = [sent_tokenize(essay) for essay in essays]

    essay_tokens = [[tokenizer.encode(sentence) for sentence in essay] for essay in essays]

    essay_sent_count = [len(essay) for essay in essay_tokens]
    essay_sent_length = [[len(sent) for sent in essay] for essay in essay_tokens]
    max_len = max([len(sent) for essay in essay_tokens for sent in essay])
    max_count = max(essay_sent_count)

    for essay in essay_tokens:
        for i in range(len(essay), max_count):
            pad_sent = [tokenizer.cls_token_id, tokenizer.sep_token_id]
            essay.append(pad_sent)

    for length in essay_sent_length:
        for i in range(len(length), max_count):
            length.append(0)

    essay_tokens_pad = [pad_sequences(essay, maxlen=max_len if max_len < 512 else 512, dtype='long', padding="post") for essay in essay_tokens]
    essay_tokens_pad = np.array(essay_tokens_pad)

    attention_mask = []
    for essay in essay_tokens_pad:
        essay_attention_mask = []
        for (i, sent) in enumerate(essay):
            sent_mask = [float(i > 0) for i in sent]
            essay_attention_mask.append(sent_mask)

        attention_mask.append(essay_attention_mask)

    attention_mask = np.array(attention_mask)

    return {
        'inputs': essay_tokens_pad,
        'sent_length': essay_sent_length,
        'sent_count': essay_sent_count,
        'attention_mask': attention_mask
    }

import math
import numpy as np
import nltk
from tqdm import tqdm
#nltk.download('punkt')


def pad_sents(sents, pad_token):
    """ Pad list of sentence according to the longest sentence in the batch.
    @param sents (list[list[str]]): list of sentence, where each sentence 
                                    is represented as a list of words.
    @param pad_token (str): pading token
    @return sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token
    """
    MAX_LEN = max([len(sent) for sent in sents])
    sents_padded = [sent[:] for sent in sents]
    for sent in sents_padded:
        if len(sent) < MAX_LEN:
            sent += [pad_token]*(MAX_LEN - len(sent))
    return sents_padded

def read_corpus(file_path, source):
    data = []
    for line in tqdm(open(file_path, mode='r', encoding='utf-8')):
        sent = nltk.word_tokenize(line)
        # Only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)
    return data

def batch_iter(data, batch_size, shuffle=False):
    num_iter = math.ceil(len(data)/batch_size)
    index_array = list(range(len(data)))
    if shuffle:
        np.random.shuffle(index_array)
    
    for i in range(num_iter):
        indices = index_array[i*batch_size: (i + 1)*batch_size]
        examples = [data[idx] for idx in indices]

        # sorted by length of source
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

    
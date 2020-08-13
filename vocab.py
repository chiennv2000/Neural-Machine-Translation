import os
import json
import torch
from itertools import chain
from collections import Counter
from utils import pad_sents, read_corpus

class VocabEntry(object):
    """
    Class that build vocabulary from courpus.
    """
    def __init__(self, word2id=None):
        """ Init VocabEntry Insance
        @param word2id (dict): mapping words to indices.
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0   # Pad token
            self.word2id['<s>'] = 1     # Start token
            self.word2id['</s>'] = 2    # End token
            self.word2id['<unk>'] = 3   # Unknow token
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        """ Retrieve word's index. Return the unk's index if the word is out of vocabulary.
        @param word (str): word to look up
        @return index (int): index of word
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """ Check if the word is captured by VocabEntry.
        @param word (str): word to look up
        @return contains (bool): whether word is contained 
        """      
        return word in self.word2id

    def __setitem__(self, key, value):
        """ Raise Error if you tries to edit VocabEntry.
        """
        raise ValueError("Vocabulary is read only.")

    def __len__(self):
        """ Compute number of words in VocabEntry.
        @return len (int): number of words in VocabEntry.
        """
        return len(self.word2id)

    def int2word(self, word_id):
        """ Return mapping of index to word.
        @param word_id (int): word index
        @return word (str): word corresponding to the index
        """
        return self.id2word[word_id]

    def add(self, word):
        """ Add word to VocabEntry if it is previously unseen.
        @param word (str): word to add the VocabEntry
        @return index (int): index that the word has been assigned
        """
        if word not in self.word2id:
            word_id = self.word2id[word] = len(self.word2id)
            return word_id
        else:
            return self[word]

    def words2indices(self, sents):
        """ Convert list of words or list of sentences of words
        into list or list of list of indices.
        @param sents (list[str] or list[list[str]]): sentence(s) in words
        @return word_ids (list[int] or list[list[int]]): sentence(s) in indices
        """
        if type(sents[0]) == list:
            return [[self[w] for w in sent] for sent in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        """ Convert list of indices into words.
        @param word_ids (list[int]): list of word ids
        @return sents (list[str]): list of words
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents, device) -> torch.Tensor:
        """ Convert list of sentences (words) into tensor with necessary padding for 
        shorter sentences.

        @param sents (List[List[str]]): list of sentences (words)
        @param device: device on which to load the tesnor, i.e. CPU or GPU

        @returns sents_var: tensor of (max_sentence_length, batch_size)
        """ 
        word_ids = self.words2indices(sents)
        sent_t = pad_sents(word_ids, self.word2id['<pad>'])
        sent_var = torch.tensor(sent_t, dtype=torch.long, device=device)
        return torch.t(sent_var)    # shape (src_len, b)

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        """ Given a corpus construct a Vocab Entry.
        @param corpus (list[str]): corpus of text produced by read_corpus function
        @param size (int): # of words in vocabulary
        @param freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word
        @returns vocab_entry (VocabEntry): VocabEntry instance produced from provided corpus
        """
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        top_k_words = sorted(valid_words, key= lambda w: word_freq[w], reverse=True)[: size]
        for w in top_k_words:
            vocab_entry.add(w)
        return vocab_entry

class Vocab(object):
    """ Class that contains source and target language.
    """
    def __init__(self, src_vocab: VocabEntry, tgt_vocab: VocabEntry):
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(src_sents, tgt_sents, vocab_size, freq_cutoff) -> 'Vocab':
        assert len(src_sents) == len(tgt_sents)
        print('initializing source vocabulary...')
        src = VocabEntry.from_corpus(src_sents, vocab_size, freq_cutoff)
        print('initializing target vocabulary...')
        tgt = VocabEntry.from_corpus(tgt_sents, vocab_size, freq_cutoff)
        return Vocab(src, tgt)

    def save(self, file_path):
        json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id), open(file_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    
    @staticmethod
    def load(file_path):
        entry = json.load(open(file_path, 'r', encoding='utf-8'))
        src = entry['src_word2id']
        tgt = entry['src_word2id']
        return Vocab(VocabEntry(src), VocabEntry(tgt))

def main(args):
    if os.path.isfile(os.path.join(args.data_path, args.vocab_file)):
        print('vocab file exist in %s.' % os.path.join(args.data_path, args.vocab_file))
    else:
        print('Reading in source sentence : %s ...' %args.train_src)
        src_sents = read_corpus(os.path.join(args.data_path, args.train_src), source='src')
        print('Reading in target sentence : %s ...' %args.train_tgt)
        tgt_sents = read_corpus(os.path.join(args.data_path, args.train_tgt), source='tgt')

        vocab = Vocab.build(src_sents, tgt_sents, vocab_size=args.size, freq_cutoff=args.freq_cutoff)
        print('generated vocabulary, source %d words, target %d words' % (len(vocab.src), len(vocab.tgt)))

        vocab.save(os.path.join(args.data_path, args.vocab_file))
        print('vocabulary saved to %s' % os.path.join(args.data_path, args.vocab_file))

if __name__ == "__main__":
    main(args)






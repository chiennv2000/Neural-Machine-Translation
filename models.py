from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class ModelEmbedding(nn.Module):
    """
    Class that converts input to embeddings.
    """
    def __init__(self, embed_size, vocab):
        super(ModelEmbedding, self).__init__()
        self.embed_size = embed_size

        src_pad_token_idx = vocab.src['pad']
        tgt_pad_token_idx = vocab.tgt['pad']

        self.src_embedding = nn.Embedding(num_embeddings=len(vocab.src), embedding_dim=embed_size, padding_idx=src_pad_token_idx)
        self.tgt_embedding = nn.Embedding(num_embeddings=len(vocab.tgt), embedding_dim=embed_size, padding_idx=tgt_pad_token_idx)

class NMT(nn.Module):
    """ Neural Machine Translation Model:
        - Bidirectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model 
    """
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate, device):
        super(NMT, self).__init__()
        self.device = device
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.embedding = ModelEmbedding(embed_size, vocab)
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, bidirectional=True)
        self.decoder = nn.LSTMCell(input_size=embed_size+hidden_size, hidden_size=hidden_size)
        self.h_projection = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.c_projection = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.att_projection = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.combined_output_projection = nn.Linear(3*hidden_size, hidden_size, bias=False)
        self.target_vocab_projection = nn.Linear(hidden_size, len(vocab.tgt), bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        source_lengths = [len(s) for s in source]

        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)                  # (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)                  # (tgt_len, b)

        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        enc_masks = self.generate_sent_mask(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)   # (tgt_len - 1, b, h)

        P = F.softmax(self.target_vocab_projection(combined_outputs), dim=-1)                   # (tgt_len - 1, b, vocab_tgt_len)
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()                       # (tgt_len, b)
        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1: ].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        score = target_gold_words_log_prob.sum(0)                                               # (b)
        return score


    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        X = self.embedding.src_embedding(source_padded)                             # [src_len, b, e]
        packed_embed = pack_padded_sequence(X, lengths=source_lengths, batch_first=False)
        output, (last_hidden, last_cell) = self.encoder(packed_embed)               # (src_len, b, 2*h), (2, b, h), (2, b, h)
        # Because using packedsequence, we must pad it
        enc_hiddens = pad_packed_sequence(output)[0].permute(1, 0, 2)                # (b, src_len, 2*h)
        last_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)            # (b, 2*h)
        last_cell = torch.cat((last_cell[0], last_cell[1]), dim=1)                  # (b, 2*h)

        init_decoder_hidden = self.h_projection(last_hidden)                        # (b, h)
        init_decoder_cell = self.c_projection(last_cell)                            # (b, h)
        dec_init_state = (init_decoder_hidden, init_decoder_cell)
        return enc_hiddens, dec_init_state
    
    def decode(self, enc_hidden, enc_masks, dec_init_state, target_padded):
        # Chop of the <END> token for max length sentences
        target_padded = target_padded[:-1]
        dec_state = dec_init_state
        batch_size = enc_hidden.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)
        combined_outputs = []
        
        enc_hidden_proj = self.att_projection(enc_hidden)                           # (b, src_len, h)
        Y = self.embedding.tgt_embedding(target_padded)                             # (tgt_len - 1, b, e)
        # Take each word in sentence by sequence 
        for Y_t in Y.split(1, dim=0):
            Y_t = Y_t.squeeze()                                                     # (b, e)
            Y_t = torch.cat((Y_t, o_prev), dim=1)                                   # (b, e + h)
            dec_state, o_t = self.step(Y_t, dec_state, enc_hidden, enc_hidden_proj, enc_masks)
            combined_outputs.append(o_t)
            o_prev = o_t
        
        combined_outputs = torch.stack(combined_outputs, dim=0)                     # (tgt_len - 1, b, h)

        return combined_outputs
    
    def step(self, Y_t, dec_state, enc_hidden, enc_hidden_proj, enc_masks):
        dec_hidden, cell_hidden = self.decoder(Y_t, dec_state)                      # (b, h), (b, h)
        e_t = torch.bmm(enc_hidden_proj, dec_hidden.unsqueeze(dim=2)).squeeze(dim=2)# (b, src_len)
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))
        a_t = F.softmax(e_t, dim=1)                                                 # (b, src_len)
        a_t = torch.bmm(a_t.unsqueeze(1), enc_hidden).squeeze(1)                    # (b, 2*h)
        u_t = torch.cat((dec_hidden, a_t), dim=1)                                   # (b, 3*h)
        v_t = self.combined_output_projection(u_t)                                  # (b, h)
        o_t = self.dropout(F.tanh(v_t))                                             # (b, h)
        return (dec_hidden, cell_hidden), o_t

    def generate_sent_mask(self, enc_hidden, source_lengths):
        enc_masks = torch.zeros((enc_hidden.size(0), enc_hidden.size(1)), dtype=torch.float) # (b, src_len)
        for e_id, src_len in enumerate(source_lengths, 0):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])
        return model

    def save(self, path: str):
        """ Save the odel to a file.
        """
        print('save model parameters to [%s]' % path)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)







        


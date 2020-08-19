import os
import time
import math

import argparse
import numpy as np

from models import NMT
from vocab import Vocab, VocabEntry
from utils import read_corpus, batch_iter

import torch

parser = argparse.ArgumentParser(description='Neural Machine Translation')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--data-path', type=str, default='data')
parser.add_argument('--save-path', type=str, default='./model/model.bin')
parser.add_argument('--train-src', type=str, default='train.en.txt')
parser.add_argument('--train-tgt', type=str, default='train.vi.txt')
parser.add_argument('--test-src', type=str, default='tst2013.en.txt')
parser.add_argument('--test-tgt', type=str, default='tst2013.vi.txt')
parser.add_argument('--vocab-file', type=str, default='vocab.json')

parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--size', type=int, default=50000)
parser.add_argument('--valid-niter', type=int, default=2000)
parser.add_argument('--freq-cutoff', type=int, default=2)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=bool, default=False)

parser.add_argument('--clip-grad', type=float, default=5.0)
parser.add_argument('--embed-size', type=int, default=256)
parser.add_argument('--hidden-size', type=int, default=256)
parser.add_argument('--dropout-rate', type=float, default=0.3)
parser.add_argument('--uniform-init', type=float, default=0.1)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--max-num-trial', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr-decay', type=float, default=0.5)

if torch.cuda.is_available():       
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def evaluate_ppl(model, dev_data, batch_size=32):
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents) 
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    model.train()

    return ppl

def train(args):
    train_data_src = read_corpus(os.path.join(args.data_path, args.train_src), source='src')
    train_data_tgt = read_corpus(os.path.join(args.data_path, args.train_tgt), source='tgt')

    dev_data_src = read_corpus(os.path.join(args.data_path, args.test_src), source='src')
    dev_data_tgt = read_corpus(os.path.join(args.data_path, args.test_tgt), source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))
    train_batch_size = int(args.batch_size)
    clip_grad = float(args.clip_grad)
    valid_niter = int(args.valid_niter)
    log_every = 10
    model_save_path = args.save_path

    vocab = Vocab.load(os.path.join(args.data_path, args.vocab_file))
    model = NMT(embed_size=args.embed_size,
                hidden_size=args.hidden_size,
                vocab=vocab,
                dropout_rate=args.dropout_rate,
                device=device)
    uniform_init = float(args.uniform_init)
    for p in model.parameters():
        p.data.uniform_(-uniform_init, uniform_init)
    
    model.train()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()

    print("Training...\n")
    while True:
        epoch += 1
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()
            batch_size = len(src_sents)

            example_losses = - model(src_sents, tgt_sents)      # (b,)
            batch_loss = example_losses.sum()
            loss = batch_loss/batch_size

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time))

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0

            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples))
                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1
                print('begin validation ...')

                dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl))

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path)
                    model.save(model_save_path)
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                
                elif patience < int(args.patience):
                    patience += 1
                    print('hit patience %d' % patience)
                    if patience == int(args.patience):
                        num_trial += 1
                        print('hit #%d trial' % num_trial)
                        if num_trial == int(args.max_num_trial):
                            print('early stop!')
                            exit(0)
                        lr = optimizer.param_groups[0]['lr'] * float(args.lr_decay)

                        print('load previously best model and decay learning rate to %f' % lr)

                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0
                
                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!')
                    exit(0)

def predict(args):
    model = NMT.load(args['save_path'])
    text  = ''
    while True:
        text = input("Enter your text (english): \n")
        if text == 'stop':
            break
        output = model.beam_search(text.split())
        print("Translating....\n.")
        print(" ".join(output))

if __name__ == "__main__":
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed * 13 // 7)
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)
    else:
        raise RuntimeError('invalid run mode')



import torch
import config
import torchtext
from torchtext import data, vocab
from torchtext import datasets

from util.mnli import MNLI
import numpy as np
import itertools
from torch.autograd import Variable


'''f = open('data/glove.840B.300d.txt', 'r')
idx_to_embedding = {}
word_to_idx = {}
idx_to_word = {}
i = 0
all_embeddings = []
for line in f:
    splitLine = line.split(sep=' ')
    word = splitLine[0]
    embedding = np.array([float(val) for val in splitLine[1:]])

    word_to_idx[word] = i
    idx_to_word[i] = word
    idx_to_embedding[i] = embedding
    i += 1
    all_embeddings.append(embedding)'''


class RParsedTextLField(data.Field):
    def __init__(self, eos_token='<pad>', lower=False, include_lengths=True):
        super(RParsedTextLField, self).__init__(
            eos_token=eos_token, lower=lower, include_lengths=True, preprocessing=lambda parse: [
                t for t in parse if t not in ('(', ')')],
            postprocessing=lambda parse, _, __: [
                list(reversed(p)) for p in parse])


class ParsedTextLField(data.Field):
    def __init__(self, eos_token='<pad>', lower=False, include_lengths=True):
        super(ParsedTextLField, self).__init__(
            eos_token=eos_token, lower=lower, include_lengths=True, preprocessing=lambda parse: [
                t for t in parse if t not in ('(', ')')])

    def plugin_new_words(self, new_vocab):
        for word, i in new_vocab.stoi.items():
            if word in self.vocab.stoi:
                continue
            else:
                self.vocab.itos.append(word)
                self.vocab.stoi[word] = len(self.vocab.itos) - 1


def load_new_embedding(embd_file=config.DATA_ROOT + "/saved_embd_new.pt"):
    embd = torch.load(embd_file)
    return embd


def load_data_sm(data_root, embd_file, reseversed=True, batch_sizes=(32, 32, 32, 32, 32), device=-1, shuffle=False):
    if reseversed:
        testl_field = RParsedTextLField()
    else:
        testl_field = ParsedTextLField()

    transitions_field = datasets.snli.ShiftReduceField()
    y_field = data.Field(sequential=False)
    g_field = data.Field(sequential=False)

    train_size, dev_size, test_size, m_dev_size, m_test_size = batch_sizes

    snli_train, snli_dev, snli_test = datasets.SNLI.splits(testl_field, y_field, transitions_field, root=data_root)

    mnli_train, mnli_dev_m, mnli_dev_um = MNLI.splits(testl_field, y_field, transitions_field, g_field, root=data_root,
                                                      train='train.jsonl',
                                                      validation='dev_matched.jsonl',
                                                      test='dev_mismatched.jsonl')

    mnli_test_m, mnli_test_um = MNLI.splits(testl_field, y_field, transitions_field, g_field, root=data_root,
                                            train=None,
                                            validation='test_matched_unlabeled.jsonl',
                                            test='test_mismatched_unlabeled.jsonl')

    testl_field.build_vocab(snli_train, snli_dev, snli_test,
                            mnli_train, mnli_dev_m, mnli_dev_um, mnli_test_m, mnli_test_um)

    g_field.build_vocab(mnli_train, mnli_dev_m, mnli_dev_um, mnli_test_m, mnli_test_um)
    y_field.build_vocab(snli_train)
    print('Important:', y_field.vocab.itos)
    print(embd_file)

    testl_field.vocab.vectors = torch.load(embd_file)
    # will be assigned to model.embeddings.weight.data
    #testl_field.vocab.load_vectors()

    snli_train_iter, snli_dev_iter, snli_test_iter = data.Iterator.splits(
        (snli_train, snli_dev, snli_test), batch_sizes=batch_sizes, device=-1, shuffle=False, sort=False)

    mnli_train_iter, mnli_dev_m_iter, mnli_dev_um_iter, mnli_test_m_iter, mnli_test_um_iter = data.Iterator.splits(
        (mnli_train, mnli_dev_m, mnli_dev_um, mnli_test_m, mnli_test_um),
        batch_sizes=(train_size, m_dev_size, m_test_size, m_dev_size, m_test_size),
        device=-1, shuffle=shuffle, sort=False)

    return (snli_train_iter, snli_dev_iter, snli_test_iter), (mnli_train_iter, mnli_dev_m_iter, mnli_dev_um_iter, mnli_test_m_iter, mnli_test_um_iter), testl_field.vocab.vectors


def load_data_embd_vocab_snli(data_root, embd_file, reseversed=True, batch_sizes=(32, 32, 32, 32, 32), device=-1):
    if reseversed:
        testl_field = RParsedTextLField()
    else:
        testl_field = ParsedTextLField()

    transitions_field = datasets.snli.ShiftReduceField()
    y_field = data.Field(sequential=False)
    g_field = data.Field(sequential=False)

    train_size, dev_size, test_size, m_dev_size, m_test_size = batch_sizes

    snli_train, snli_dev, snli_test = datasets.SNLI.splits(testl_field, y_field, transitions_field, root=data_root)

    mnli_train, mnli_dev_m, mnli_dev_um = MNLI.splits(testl_field, y_field, transitions_field, g_field, root=data_root,
                                                      train='train.jsonl',
                                                      validation='dev_matched.jsonl',
                                                      test='dev_mismatched.jsonl')

    mnli_test_m, mnli_test_um = MNLI.splits(testl_field, y_field, transitions_field, g_field, root=data_root,
                                            train=None,
                                            validation='test_matched_unlabeled.jsonl',
                                            test='test_mismatched_unlabeled.jsonl')

    testl_field.build_vocab(snli_train, snli_dev, snli_test,
                            mnli_train, mnli_dev_m, mnli_dev_um, mnli_test_m, mnli_test_um)

    testl_field.vocab.vectors = torch.load(embd_file)

    return testl_field.vocab.vectors, testl_field.vocab, testl_field


def combine_two_set(set_1, set_2, rate=(1, 1), seed=0):
    np.random.seed(seed)
    len_1 = len(set_1)
    len_2 = len(set_2)
    # print(len_1, len_2)
    p1, p2 = rate
    c_1 = np.random.choice([0, 1], len_1, p=[1 - p1, p1])
    c_2 = np.random.choice([0, 1], len_2, p=[1 - p2, p2])
    iter_1 = itertools.compress(iter(set_1), c_1)
    iter_2 = itertools.compress(iter(set_2), c_2)
    for it in itertools.chain(iter_1, iter_2):
        yield it


if __name__ == '__main__':
    snli, mnli, embd = load_data_sm(config.DATA_ROOT, config.EMBD_FILE, reseversed=False,
                                    batch_sizes=(32, 32, 32))

    s_train, s_dev, s_test = snli
    m_train, m_dev_m, m_dev_um, m_test_m, m_test_um = mnli

    train = combine_two_set(s_train, m_train, rate=[0.15, 1])

    print(len(list(train)))
    print(len(m_train))
    print(len(s_train))

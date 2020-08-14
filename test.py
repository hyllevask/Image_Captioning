from vocab import Vocabulary

vocab = Vocabulary(vocab_threshold = 2)

for value, count in lengths:
    print('value: %2d --- count: %5d' % (value, count))
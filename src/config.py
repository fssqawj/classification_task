
category2id = {'entertainment': 0, 'sports': 1, 'car': 2, 'society': 3, 'tech': 4,
               'world': 5, 'finance': 6, 'game': 7, 'travel': 8, 'military': 9,
               'history': 10, 'baby': 11, 'fashion': 12, 'food': 13, 'discovery': 14,
               'story': 15, 'regimen': 16, 'essay': 17}

id2category = {index:label for label, index in category2id.items()}
# category2id = {'__label__{}'.format(i): i for i in range(18)}

ROOT = '../data/nlpcc_data'

stopwords_file = ROOT + '/stopwords.txt'
word_embed_file = ROOT + '/emb/emb_wd/headlines.vec'
vocab2id_file = ROOT + '/vocab2id.p'
we_file = ROOT + '/we.p'


DATA_DIR = '../data/nlpcc_data/word/'

train_file = DATA_DIR + 'train-exp-fix-2.txt'

dev_file = DATA_DIR + 'test-exp-fix-2.txt'
dev_predict_file = './outputs/{}-test-predicts.txt'

test_file = DATA_DIR + 'test-exp-fix-2.txt'
test_predict_file = './outputs/{}-test-predicts.txt'

word_dim = 100
max_sent_len = 50

save_dir = '../generate/save'

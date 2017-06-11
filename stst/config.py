DICT_LABEL_TO_INDEX = {
    "entertainment": 0,
    "sports": 1,
    "car": 2,
    "society": 3,
    "tech": 4,
    "world": 5,
    "finance": 6,
    "game": 7,
    "travel": 8,
    "military": 9,
    "history": 10,
    "baby": 11,
    "fashion": 12,
    "food": 13,
    "discovery": 14,
    "story": 15,
    "regimen": 16,
    "essay": 17
}

DICT_INDEX_TO_LABEL = {DICT_LABEL_TO_INDEX[label]: label for label in DICT_LABEL_TO_INDEX}


GENERATE_DIR = './generate'

EMB_WORD_DIR = './data/nlpcc_data/emb/emb_wd'
EMB_CHAR_DIR = './data/nlpcc_data/emb/emb_wd'

''' parse config'''
PARSE_DIR = GENERATE_DIR + '/parse'

''' result out '''
OUTPUT_DIR = GENERATE_DIR + '/outputs'

''' feature config '''
FEATURE_DIR = GENERATE_DIR + '/features'

''' model config '''
MODEL_DIR = GENERATE_DIR + '/models'

''' record config '''
RECORD_DIR = GENERATE_DIR + '/records'

''' dictionary config '''
DICTIONARY_DIR = GENERATE_DIR + '/dicts'


''' tmp config '''
TMP_DIR = '/generate' + '/tmp'

RESOURCE = './stst/resources'
''' dict config '''
DICT_DIR = RESOURCE

#
# ''' nn feature config '''
# NN_FEATURE_PATH = RESOURCE + '/iclr2016-test/data/features'


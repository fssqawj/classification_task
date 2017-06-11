# STST for NLPCC 2017 Task2

## ./stst/data_tools

1. define the sent object
  - get_sent
  - get_label **required in model **
  - get_instance_string **required in model **

2. load the data
  - load data
  - load parse data

## ./features/features_unigram.py

1. Some Modification
- extract idf during train, and load the dict during dev and test
- extract can return str object (for sparse features), but must return the dimension in the first info list
Next is a example of the Unigram Feature.
 ```
  class TFFeature(Feature):

    def __init__(self, type='word', convey='count', **kwargs):
        super(TFFeature, self).__init__(**kwargs)
        self.type = type
        self.convey = convey
        self.feature_name = '{}-{}#{}'.format(self.feature_name, type, self.convey)

    def extract_information(self, train_instances):
        if self.is_training:
            sents = []
            for train_instance in train_instances:
                sent = train_instance.get_sent(self.type)
                sents.append(sent)
            idf_dict = utils.idf_calculator(sents)
            with utils.create_write_file(config.DICTIONARY_DIR + '/{}_idf_dict.txt'.format(self.type)) as fw:
                idf_dict_tuple = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)
                for key, value in idf_dict_tuple:
                    print('{}\t{}'.format(key, value), file=fw)
        else:
            with utils.create_read_file(config.DICTIONARY_DIR + '/{}_idf_dict.txt'.format(self.type)) as fr:
                idf_dict = {}
                for line in fr:
                    line = line.strip().split('\t')
                    idf_dict[line[0]] = float(line[1])
        self.unigram_dict = idf_dict
        word_keys = sorted(idf_dict.keys(), reverse=True)
        self.word2index = {word: i for i, word in enumerate(word_keys)}

    def extract(self, train_instance):
        sent = train_instance.get_sent(self.type)
        feats = utils.sparse_vectorize(sent, self.unigram_dict, self.word2index, self.convey)
        infos = [len(self.unigram_dict), 'tf']
        feats = Feature._feat_dict_to_string(feats)
        return feats, infos
 ```

## main_lr.py


## main_stack.py

- Idea:
   1. 将数据分成5折，四份做train，一份做test
   2. 将得到的每一份test进行合并，得到数据集大小的用于stack的训练数据stack_train
   3. 训练stack的分类器

- Steps:
   1和2. => model.cross_validation, 写入model.output_file
   3. => model.train() 训练一个分类器

   在测试阶段，将用5份数据训练好的1级分类器测试得到第一部分结果，再通过二级分类器得到最终结果, model.test会自动完成:
   ```
   elif isinstance(feature_class, Model):
        if dev:
            feature_class.test(train_instances, train_file)
            feature_string = feature_class.load_model_score(train_file)
        else:
            ''' seperate to train for speed up '''
            # feature_class.train(train_instances, train_file)
            feature_string = feature_class.load_model_score(train_file)
   ```

## Other tools

- ./eval.py gold_file predict_file
- ./error_analysis.py gold_file predict_file output_file 按csv格式保存到output_file
- ./statistic.py 用于数据统计
- ./test.py 将生成的prob变成json格式
- ./test_super.py 将最后的结果进行ensemble，并输出


## ./data

数据集为NLPCC 2017 Shared Task2
官方数据: https://github.com/FudanNLP/nlpcc2017_news_headline_categorization
我们整理的数据: 百度云盘(176MB) http://pan.baidu.com/s/1nu8hUhb
# coding: utf8
from __future__ import print_function
from stst.config import DICT_INDEX_TO_LABEL

class Sent(object):
    def __init__(self, sent, label):
        """
        :param  sent: list
        :param  label: str
        :return SentPair: tuple, (sa, sb)
        """
        self.sent = sent  # str
        self.label = label

    def get_sent(self, type):
        sent = None
        if type == 'word':
            sent = self.get_word()
        elif type == 'char':
            sent = self.get_char()
        return sent

    def get_word(self):
        return self.sent

    def get_char(self):
        sent = ''.join(self.sent)
        return list(sent)

    def get_label(self):
        return self.label

    def get_instance_string(self):
        sent = ' '.join(self.sent)
        instance_string = '{}\t{}'.format(DICT_INDEX_TO_LABEL[self.label], sent)
        return instance_string


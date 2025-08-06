import re
import math
import os
import pandas as pd
from collections import Counter, defaultdict

from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

class MyArray:
    def __init__(self):
        self.data = []

    def add_many(self, values):  # 一次加很多个
        self.data.extend(values)  # 比 append 强，能加整个列表
        return self.data

    def print_all(self):
        for v in self.data:
            print(v)

    def save_to_file(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for v in self.data:
                f.write(v)


class SMSClassifier:
    def __init__(self):
        self.ham_array = MyArray()
        self.spam_array = MyArray()

    def read (self ) :
        print(os.getcwd())
        path = "/Volumes/Samsung T7/Visual Studio Code/python_language/machine_learning/sms+spam+collection/SMSSpamCollection"
        email = os.open(path , os.O_RDWR)

        with open(path , 'r' , encoding = 'utf-8') as f:
             text = f.read()

        words = re.findall(r'\b\w+\b', text.lower())
        print(words)
        print(len(words))
        return words

    def classification(self, words):
        i = 0
        while i < len(words):
            if words[i] == 'ham':
                # 收集一条 ham 消息
                message = []
                i += 1
                while i < len(words) and words[i] != 'ham' and words[i] != 'spam':
                    message.append(words[i])
                    i += 1
                self.ham_array.add_many(message)

            elif words[i] == 'spam':
                # 收集一条 spam 消息
                message = []
                i += 1
                while i < len(words) and words[i] != 'ham' and words[i] != 'spam':
                    message.append(words[i])
                    i += 1
                self.spam_array.add_many(message)

            else:
                # 出现异常字符，跳过
                i += 1

    def print_all(self):
        print("==== HAM Messages ====")
        self.ham_array.print_all()

        print("\n==== SPAM Messages ====")
        self.spam_array.print_all()



if __name__ == '__main__':
    clf = SMSClassifier()
    words = clf.read()             # 读取所有词语
    clf.classification(words)     # 分类
    clf.print_all()               # 打印各自分类结果
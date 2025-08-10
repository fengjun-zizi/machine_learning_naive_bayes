import re
from math import log, lgamma
import os
import pandas as pd
from collections import Counter, defaultdict
import numpy as np
from bottleneck import bench
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray
from pygments.lexer import words
from Proccess_mails import load_tokenized_texts


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
        # self.MyArray = MyArray()
        # self.ham_array = MyArray()
        # self.spam_array = MyArray()
        # self.arraays = []
        self.ham = []
        self.spam = []
        self.unique_ham = []
        self.unique_spam = []
        self.i= 0
        self.w = 1
        self.count_ham = 1
        self.count_spam = 1
        self.array_ham = []
        self.array_spam = []
        self.count = 1
        self.data_ham = []
        self.data_spam = []
        self.span_words_probs_dic = {}
        self.ham_words_probs_dic = {}
        self.words_probs_dic = {}
        self.ham_dic = {}
        self.spam_dic = {}

    def read (self ) :
        # print(os.getcwd())
        path = "/Volumes/Samsung T7/Visual Studio Code/python_language/machine_learning/sms+spam+collection/SMSSpamCollection"
        email = os.open(path , os.O_RDWR)

        with open(path , 'r' , encoding = 'utf-8') as f:
             text = f.read()

        words = re.findall(r'\b\w+\b', text.lower())
       #  print(type(words))
        # print(len(words))
        return words



    def classification(self,words) :
        q = 0
        data = {}
        self.i = 0
        while q < len(words) :
            if words [self.i] == 'ham' :
                key = f"data{self.count}"
                # 初始化
                data[key] = [words[self.i]]
                # data[key].append("ham")
                q = self.i + 1
                while q < len(words) and words [q] != 'ham' :
                    data[key].append(words[q])
                    key = f"data{self.count}"

                    # data[key] = []


                    if self.i + 1 < len(words) and words [self.i + 1 ] == 'ham' :
                        self.count += 1

                    q += 1
                    self.i = q

                    if self.i + 1 < len(words) and words [self.i  ] == 'spam' :
                        self.count += 1
                        break



            elif self.i < len(words) and words [self.i] == 'spam' :

                key = f"data{self.count}"
                data[key] = [words[self.i]]
                q = self.i + 1
                while q < len(words) and words [q] != 'spam' :
                    data[key].append(words[q])
                    key = f"data{self.count}"

                    if self.i + 1 < len(words) and words [self.i + 1 ] == 'spam' :
                        self.count += 1
                    q += 1
                    self.i = q
                    if self.i + 1 < len(words) and  words[self.i] == 'ham':
                        self.count += 1
                        break

            else :
                q += 1
                self.i = q

        # print(data)



        return data

    def print_data(self , data  , words) :
        print("______________________________")
        print(data["data100"][1])
        print("______________________________")
        print(f"data的长度",len(data))
        print("______________________________")
        print(f"words的长度",len(words))
        print("______________________________")

        print("______________________________")
        print(len(data["data100"]))
        print("______________________________")
        count = 0
        x = 0
        key = f"data{self.count_ham}"
        print(data[key][count])
        print("______________________________")

        while self.w < len(data): # self.w = 1
            key = f"data{self.w}"
            if self.w < len(data) and data[key][count] == 'ham':
                while self.w < len(data):
                    key = f"data{self.w}"
                    if data[key][count] == 'ham' :
                        self.array_ham.append(data[key])
                        self.count_ham += 1
                        self.w += 1
                    else:
                        break
            elif self.w < len(data) and data[key][count] == 'spam':
                key = f"data{self.w}"
                while self.w < len(data):
                    key = f"data{self.w}"
                    if data[key][count] == 'spam' :
                        self.array_spam.append(data[key])
                        self.count_spam += 1
                        self.w += 1
                    else :
                        break

        return self.array_ham, self.array_spam



    def process_data(self) :
        self.data_ham = [row[1:] for row in self.array_ham]
        self.data_spam = [row[1:] for row in self.array_spam]
        #print("____________________________________________________")
        #print(f"ham处理后的数据：",data_ham)
        #print("____________________________________________________")
        #print(f"spam处理后的数据：",data_spam)
        # 把一个多维的数组转换成一个一维数组
        self.ham = np.concatenate(self.array_ham).tolist()
        self.spam = np.concatenate(self.array_spam).tolist()



        self.unique_ham = list(set(self.ham))
        self.unique_spam = list(set(self.spam))

        return self.unique_ham, self.unique_spam

    def calculate_probability(self, data):

        ham_count = len(self.unique_ham)
        spam_count = len(self.unique_spam)
        count = ham_count + spam_count

        # 展平成一维列表
        flat_ham = [word for row in self.data_ham for word in row]
        flat_spam = [word for row in self.data_spam for word in row]

        # 合并为一维 NumPy 数组
        all_words = np.array(flat_ham + flat_spam)
        all_words_count = Counter(all_words)

        # 单独统计 ham 和 spam 的词频 Counter
        all_words_ham_counter = Counter(flat_ham)
        all_words_spam_counter = Counter(flat_spam)

        # 统计总词数
        all_words_ham_count = sum(all_words_ham_counter.values())
        all_words_spam_count = sum(all_words_spam_counter.values())
        all = sum(all_words_count.values())

        vocab = set(all_words)  # 所有出现过的词
        vocab_size = len(vocab)

        print("HAM 词总数:", all_words_ham_count)
        print("SPAM 词总数:", all_words_spam_count)
        print("总词数:", all_words_ham_count + all_words_spam_count)

        for word in all_words:
            """P(word | ham )"""
            prob_word_spam = (all_words_spam_counter.get(word, 0) + 1) / (all_words_spam_count + vocab_size)
            self.span_words_probs_dic[word] = prob_word_spam
            # print(f"P({word} | spam) = {prob_word_spam:.6f}")

        for word in all_words:
            """P(word | spam ) with laplace smoothing"""
            prob_word_ham = (all_words_ham_counter.get(word, 0) + 1 ) / ( all_words_ham_count + vocab_size)
            self.ham_words_probs_dic[word] = prob_word_ham
            # print(f"P({word} | ham) = {prob_word_ham:.6f}")


        for word in all_words:
            """P(word)"""
            prob_word = ( all_words_count.get(word, 0) + 1 )/ ( all + vocab_size )
            self.words_probs_dic[word] = prob_word
            # print(f"P({word} ) = {prob_word:.6f}")

        # P(spam)
        prob_spam = all_words_ham_count / all
        # print(f"prob_spam: {prob_spam}")


        # P(ham)
        prob_ham = all_words_spam_count / all
        # print(f"prob_ham: {prob_ham}")

        return prob_spam, prob_ham

    def calculate_log(self, prob_spam, prob_ham) :
        log_prob_spam = np.log(prob_spam)
        log_prob_ham = np.log(prob_ham)

        all = self.data_ham + self.data_spam
        test_line = all[5500]
        #print(test_line)
        print(f"总行数",len(all))

        data_1 = []
        labels_1 = []

        root = "/Volumes/Samsung T7/Visual Studio Code/python_language/machine_learning/sms+spam+collection/test-mails"
        data_1, labels_1 = load_tokenized_texts(root)

        print(f"共读取 {len(data_1)} 封邮件")
        print(f"训练数据", len(labels_1))

        wo = self.read()
        data = self.classification(wo)

        # 标签
        labels = []

        #内容
        features = []


        # print(data)

        for key in data:
            lst = data[key]
            labels.append(lst[0])  # 第一个是标签
            features.append(lst[1:])

        # print("标签列表：", labels)
        # print("内容列表：", features)

        # print(f"|||||||",len(features))
        q = 0
        test_mail = []
        while q < len(labels_1):
            log_prob_spam = 0
            log_prob_ham = 0
            test_line = labels_1[q]
            for word in test_line:
                # print(type(self.span_words_probs_dic))  # 应该输出 <class 'dict'>
                p_word_given_spam = self.span_words_probs_dic.get(word, 1e-10)
                p_word_given_ham = self.ham_words_probs_dic.get(word, 1e-10)

                log_prob_ham += np.log(p_word_given_ham)

                log_prob_spam += np.log(p_word_given_spam)

            if log_prob_spam > log_prob_ham:
                mail = "spam"
            else :
                mail = "ham"

            test_mail.append(mail)
            # print(test_mail)

            q += 1

        print(f"log(P(words|spam)) = {log_prob_spam}")
        print(f"log(P(words|ham )) = {log_prob_ham}")

        # test_mail 预测结果
        # labels 标签

        #print(len(test_mail))
        #print(len(labels))

        x = 1
        right_count = 0
        right_probability = 0.00
        while x < len(test_mail):
            if test_mail[x] == labels[x]:
                right_count += 1


            x += 1
        #print(f"test_mail",test_mail)
        #print(f"labels",labels)
        right_probability = right_count / len(test_mail)
        #print(right_count)
        #print(len(test_mail))
        print(f"预测准确率为：" , right_probability)







































if __name__ == '__main__':
    clf = SMSClassifier()
    reed = clf.read()
    result = clf.classification(reed)
    clf.print_data(result , reed)
    clf.process_data()
    prob_spam, prob_ham = clf.calculate_probability(result)
    clf.calculate_log(prob_spam, prob_ham)

    #clf.MyArray.print_all()



'''
    def array (self) :
        array = self.MyArray.data
        for array in self.arraays :
            array.print_all()
            
            
            
            
                    while self.w < len(data) :
            if self.w < len(data) and data[self.w][count] == 'ham' :
                while x < len(data[self.w])  :
                    self.array_ham.append(data[self.w][count])
                    self.w += 1
'''



'''
            if words [self.i] == 'spam' :
                self.count_spam += 1
                t = self.i
                if self.i == 0 :
                    q = self.i + 1
                print(q)
                print(words [q])
                while q < len(words) and words [q] != 'spam' :
                    if words [self.i] == 'spam' :
                        data.insert(self.i , "spam")
                    data.append(words [q])
                    t += 1
                    q += 1
                    self.MyArray.add_many (data)

                    self.i += 1
                self.MyArray.add_many(data)



            self.i += 1

        print(f"ham的数量：",self.count_ham)
        print(f"spam的数量：",self.count_spam)
        print(f"总共",self.i)

        return self.MyArray.data
'''
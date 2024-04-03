#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import re
from sklearn.model_selection import train_test_split
from datasets import Dataset
from datasets.dataset_dict import DatasetDict

class Train_dataset:
    
    def __init__(self, df):
        self.test_text = df['text'].values
        self.test_label = df['label'].values
        #self.token_label = []
        #self.flat_list = []
        self.text = []
        self.tokenized_text = []
        self.tokenized_text_label = [] 
        self.sequences = []
        self.labels = []
        self.dataset = None
    
    
    def label_tokens(self, token_lists, seq):
        #Labeling as 1 only for the token, which is the last token of the setence labeled as 1        
        token_label = []
        for i in range(len(token_lists)):
            if seq !=0 and i == len(token_lists)-1:
                token_label.append(self.test_label[seq]) 
            else :
                token_label.append(0)  
        return token_label

    
    def split_token_sentences(self):
        #Splitting sentence to tokens
        #Labeling as 1 only for the token, which is the last token of the setence labeled as 1

        i=0
        for seq, t in enumerate(self.test_text):
            for match in re.finditer(r"\,|\.|\;|\?|\!", t):
                a = t[i:match.start()+1]
                a.strip()
                b = a.split()
                self.text.append(a)
                self.tokenized_text.append(b)
                if match.start() != len(t) - 1:
                    i = match.start()+1
                    self.tokenized_text_label.append(self.label_tokens(b, 0))
                else:
                    i=0
                    self.tokenized_text_label.append(self.label_tokens(b, seq))
                    
     
    def flatten_list(self, nested_list):        
        #Changing the nested list to a one-dimensional list express
        flat_list = []
        for sublist in nested_list:
            flat_list.extend(sublist)
        return flat_list

    
    def generate_test_dataset(self):
        #Generating sequences to generate different sequences of speaking by combining tokens
        #Labeling sequences to identify the moments when each speaker takes their turn

        t=''
        s=0
        c=1
        first_1 = self.tokenized_text_label.index(1)
        for i in range(len(self.tokenized_text)):
            if self.tokenized_text_label[i] == 1 and c==0 :
                self.sequences.append(self.tokenized_text[i])
                self.labels.append(self.tokenized_text_label[i])
                t = ''
            elif self.tokenized_text_label[i] == 1 and c!=0 :
                t = ' '.join(self.tokenized_text[s:i+1])
                self.sequences.append(t)
                self.labels.append(self.tokenized_text_label[i])
                s = i+1
                t = ''
            elif self.tokenized_text_label[i] == 0 and c == 0:
                s = i
                c = 1
            elif self.tokenized_text_label[i] == 0 and c != 0:
                t = ' '.join(self.tokenized_text[s:i+1])
                self.sequences.append(t)
                self.labels.append(self.tokenized_text_label[i])
            else:# labels[i] == 1 and c != 0:
                t = ' '.join(self.tokenized_text[s:i+1])
                self.sequences.append(t)
                self.abels.append(self.tokenized_text_label[i])
                s = 0
                c = 0
                t = ''

    
    
    def datasets_for_training(self):
        #Splitting sequences and lables for traning and evaluating models
    
        x_train, x_test1, y_train, y_test1 = train_test_split(self.sequences, self.labels, test_size=0.4)
        x_test1, x_test2, y_test1, y_test2 = train_test_split(x_test1,y_test1, test_size=0.5)

        d = {'train':Dataset.from_dict({'label':y_train,'text':x_train}),
         'test1':Dataset.from_dict({'label':y_test1,'text':x_test1}),
         'test2':Dataset.from_dict({'label':y_test2,'text':x_test2})
         }

        self.dataset = DatasetDict(d)

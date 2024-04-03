import pandas as pd
import numpy as np
import json
import re

class Data:
#This class is to get a dataframe with each sentence of an utterance from a json file. The dataframe extracted from the json file will be basic data to be preprocessed for modeling.
    
    def __init__(self, path):
        self.path = path
        self.l_file = None
        self.json_d = None
        self.init_df = None
        self.df = None
        self.raw_text = None
        self.sentences = []
        self.labels = []

    def handle_file(self):
        self.l_file = open(self.path)
        self.json_d = json.load(self.l_file)

    def convert_json_to_dataframe(self):
        #converting loaded json to a data frame with values of a key('utterances)
        #Extracting two columns text and speaker to be used for modeling
        #Digitizing the sepaker column to integer: 0 for Assistant, 1 for user
        self.df = pd.json_normalize(self.json_d, record_path = ['utterances'])
        self.df = self.df[['text','speaker']]
        with pd.option_context('mode.chained_assignment', None):
            self.df['t_label'] = self.df.loc[:, 'speaker'].apply(lambda x: 0 if x == 'ASSISTANT' else 1)
        self.init_df = self.df
        #print("parts of the data frame with labels for sentences")
        #display(self.df[:10])

    def get_next_value(self):
        #make a column with the value of next row of 't-label'(digitized 'speaker')
        #to checkt that the lable of the sentence is same with the next sentence or not
        for i in self.df.index:
            if i < len(self.df)-1:
                self.df.loc[i, 'next'] = self.df.loc[i+1,'t_label']
            else:
                self.df.loc[i, 'next'] = self.df.loc[i,'t_label']
                self.df = self.df.astype({'next':int})
        #print("parts of the data frame with next value of t_label ")
        #print(self.df[:10])

    def compare_values(self):
        #comparing the speaker with the next speaker when its coninuous utterance labelled as 0, while others labelled as 1
        for i in range(len(self.df)):
            self.df.loc[i,'match'] = 0 if self.df.loc[i,'t_label'] == self.df.loc[i,'next'] else 1
        self.df = self.df.astype({'match':int})
        self.init_df = self.df.copy()
        #print("Parts of the dataframe with a column having compared speakers")
        #print(self.df[:10])

    def label_sentences(self):
        #splitting each sentence in an utterance into a column and labeling each sentence whether it is from the same speaker or not
        #labelling 1 only for the sentence that changes the speaker, which is the last sentence in the 'text' column and 'match' label is 1

        self.raw_text = self.df['text'].to_list()
        #print(data_texts[:10])
        self.sentences = []
        self.lables = []
        i=0
        for n, text in enumerate(self.raw_text):
            for match in re.finditer(r"\,|\.|\;|\?|\!", text):
                a = text[i:match.end()]
                a = a.strip()
                self.sentences.append(a)
                if (match.end() != len(text)) or (self.df.loc[n,'match'] != 1):
                    i = match.end()
                    self.labels.append(0)
                else:
                    i=0
                    self.labels.append(1)
        #print(self.sentences[:10])
        #print(self.labels[:10])
        #print(len(self.sentences))
        #print(len(self.labels))
    
    def initial_df(self):
        self.df = pd.DataFrame(list(zip(self.sentences, self.labels)))
        self.df.rename(columns={0:'text', 1:'label'}, inplace=True)
        self.df['count'] = self.df['text'].apply(lambda x: len(x.split()))
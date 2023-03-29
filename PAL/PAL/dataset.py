import torch
from torch.utils.data import DataLoader, Dataset
import pdb

class Counselling_Dataset(Dataset):
    def __init__(self, data, emotion, tokenizer, use_empathy_labels=False, use_politeness_labels=False):

        self.data = data
        self.emotion = emotion
        self.use_empathy_labels = use_empathy_labels
        self.use_politeness_labels = use_politeness_labels
        self.tokenizer = tokenizer
        self.tokenizer.max_len = 1500
        # tokenizer weird behavior
        self.turn_ending = [628, 198]
        # tokenizer.encode("\n\n\n")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):         
        if self.use_empathy_labels and not self.use_politeness_labels: # emp
            dial_tokens = []
            empathy_labels = [item[1] for item in self.data[index]]
            for i in range(len(self.data[index])):
              item1 = self.data[index][i][0]
              sep = "<emotion>"
              item2 = self.emotion[index][i]
              dial_tokens.append(tokenizer.encode(item1)+tokenizer.encode(sep)+tokenizer.encode(item2)+self.turn_ending)
            role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
            return role_ids, dial_tokens, empathy_labels
        elif self.use_empathy_labels and self.use_politeness_labels: # emp and pol
            dial_tokens = []
            empathy_labels = [item[1] for item in self.data[index]]
            politeness_labels = [item[2] for item in self.data[index]]
            for i in range(len(self.data[index])):
              item1 = self.data[index][i][0]
              sep = "<emotion>"
              item2 = self.emotion[index][i]
              dial_tokens.append(tokenizer.encode(item1)+tokenizer.encode(sep)+tokenizer.encode(item2)+self.turn_ending)
            role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
            return role_ids, dial_tokens, empathy_labels, politeness_labels
        elif not self.use_empathy_labels and not self.use_politeness_labels: # no emp no pol
            dial_tokens = []
            for i in range(len(self.data[index])):
              item1 = self.data[index][i][0]
              sep = "<emotion>"
              item2 = self.emotion[index][i]
              dial_tokens.append(tokenizer.encode(item1)+tokenizer.encode(sep)+tokenizer.encode(item2)+self.turn_ending)
            role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
            return role_ids, dial_tokens
        elif not self.use_empathy_labels and self.use_politeness_labels: # pol 
            dial_tokens = []
            politeness_labels = [item[1] for item in self.data[index]]
            for i in range(len(self.data[index])):
              item1 = self.data[index][i][0]
              sep = "<emotion>"
              item2 = self.emotion[index][i]
              dial_tokens.append(tokenizer.encode(item1)+tokenizer.encode(sep)+tokenizer.encode(item2)+self.turn_ending)
            role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
            return role_ids, dial_tokens, politeness_labels

    def collate(self, unpacked_data):
        return unpacked_data

    def get_turn_ending(self):
        return self.turn_ending

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='GPU_NUMBER' 

import numpy as np
import torch.nn as nn
from nltk.translate.meteor_score import meteor_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import functools
import oempator
import os
import pdb
import spacy
import pandas as pd
import json
import tqdm
import datetime
from tqdm.notebook import tqdm_notebook
import random
import pdb
from rlutils import collect_samples, ppo_step, generate_n_candidates, convert_sentences_to_strings, expand_inputs_for_N_candidates
from torch.utils.data import DataLoader, Dataset
from loss import SequenceCrossEntropyLoss
from ppo import PPOMemory
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup, RobertaForSequenceClassification, RobertaTokenizer
# from simpletransformers.classification import ClassificationModel,    
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
from dataset import Counselling_Dataset

class Trainer():
    def __init__(self,
                 modelname,
                 csvfile,
                 n_epochs,
                 print_every,
                 learning_rate,
                 epsilon,
                 human_reward,
                 average_sent_loss,
                 device,
                 alpha,
                 gamma,
                 delta,
                 num_candidates,
                 max_candidate_length,
                 top_p,
                 warmup_steps,
                 pad_token_id,
                 evaluate_every,
                 use_jacc,
                 use_uep_cons,
                 use_uee_cons,
                 use_dial_flow_cons,
                 emp_num_labels,
                 mini_batch,
                 temempature,
                 use_recent_past,
                 recompute_log_prob,
                 use_empathy_classifier,
                 emp_classifier_filename,
                 use_ea_empathy_classifier,
                 ea_emp_classifier_filename,
                 pa_pol_classifier_filename,
                 use_pa_politeness_classifier,
                 beta1,
                 beta2,
                 beta3,
                 beta4,
                 beta5,
                 beta6,
                 beta7,
                 beta8,
                 train_single_model=False,
                 single_model_to_train=None,
                 loadModel=False,
                 batch_size=None,
                 loadFilename=None,
                 use_politeness_classifier=None,
                 pol_classifier_filename=None,
                 pol_num_labels=None,
                 seedvalue=10):

        self.seedvalue = seedvalue
        self.train_single_model = train_single_model
        self.single_model_to_train = single_model_to_train
        self.nlp = spacy.load("en_core_web_sm")
        self.human_reward = human_reward
        self.seed(seedvalue)
        self.use_recent_past = use_recent_past
        self.temempature=temempature
        self.use_jacc=use_jacc,
        self.use_uep_cons=use_uep_cons,
        self.use_uee_cons=use_uee_cons,
        self.use_dial_flow_cons=use_dial_flow_cons,

        self.average_sent_loss = average_sent_loss
        self.mini_batch = mini_batch
        self.evaluate_every = evaluate_every
        self.csvfile = csvfile
        self.modelname = modelname
        self.n_epochs = n_epochs
        self.print_every = print_every
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
        self.recompute_log_prob = recompute_log_prob
        self.num_candidates = num_candidates
        self.pad_token_id = pad_token_id
        self.max_candidate_length = max_candidate_length
        
        self.alpha=alpha,
        self.gamma=gamma,
        self.delta=delta,
        
        self.top_p = top_p
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cuda:4')
        # torch.cuda.set_device(4)
        self.device = device

        self.num_labels = emp_num_labels
        self.num_pol_labels = pol_num_labels
        
        self.loadModel = loadModel
        self.loadFilename = loadFilename
        self.make_model_save_dir()
        self.make_stats_dir()
        

        self.use_ea_empathy_classifier = use_ea_empathy_classifier
        if self.use_ea_empathy_classifier and emp_classifier_filename:
            model_dict = torch.load(ea_emp_classifier_filename)
            self.ea_empathy_classifier = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=self.num_labels)
            self.politeness_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
            self.ea_empathy_classifier.config.problem_type = 'single_label_classification'
            self.ea_empathy_classifier.load_state_dict(model_dict['state_dict'])
            self.ea_empathy_classifier = self.ea_empathy_classifier.to(self.device)
            self.ea_empathy_classifier.eval()
            print('ea_empathy Classifier Loaded! (in Evaluation Mode)')
            self.binary_classifier = None
        elif self.use_ea_empathy_classifier and not ea_emp_classifier_filename:
            raise ValueError('ea_empathy classifier use set to True, but filename to load from not defined.')
        else:
            self.ea_empathy_classifier = None
            self.politeness_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

        self.use_pa_politeness_classifier = use_pa_politeness_classifier
        if self.use_pa_politeness_classifier and pol_pa_classifier_filename:
            model_dict = torch.load(pa_pol_classifier_filename)
            self.pa_politeness_classifier = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=self.num_pol_labels)
            self.pa_politeness_classifier.config.problem_type = 'single_label_classification'
            self.pa_politeness_classifier.load_state_dict(model_dict['state_dict'])
            self.pa_politeness_classifier = self.pa_politeness_classifier.to(self.device)
            self.pa_politeness_classifier.eval()
        elif self.use_politeness_classifier and not pol_classifier_filename:
            raise ValueError('pa_Politeness classifier use set to True, but filename to load from not defined.')
        else:
            self.politeness_classifier = None

        self.use_empathy_classifier = use_empathy_classifier
        if self.use_empathy_classifier and emp_classifier_filename:
            model_dict = torch.load(emp_classifier_filename)
            self.empathy_classifier = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=self.num_labels)
            self.politeness_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
            self.empathy_classifier.config.problem_type = 'single_label_classification'
            self.empathy_classifier.load_state_dict(model_dict['state_dict'])
            self.empathy_classifier = self.empathy_classifier.to(self.device)
            self.empathy_classifier.eval()
            print('empathy Classifier Loaded! (in Evaluation Mode)')
            self.binary_classifier = None
        elif self.use_empathy_classifier and not emp_classifier_filename:
            raise ValueError('empathy classifier use set to True, but filename to load from not defined.')
        else:
            self.empathy_classifier = None
            self.politeness_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

        self.use_politeness_classifier = use_politeness_classifier
        if self.use_politeness_classifier and pol_classifier_filename:
            model_dict = torch.load(pol_classifier_filename)
            self.politeness_classifier = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=self.num_pol_labels)
            self.politeness_classifier.config.problem_type = 'single_label_classification'
            self.politeness_classifier.load_state_dict(model_dict['state_dict'])
            self.politeness_classifier = self.politeness_classifier.to(self.device)
            self.politeness_classifier.eval()
        elif self.use_politeness_classifier and not pol_classifier_filename:
            raise ValueError('Politeness classifier use set to True, but filename to load from not defined.')
        else:
            self.politeness_classifier = None

        self.getDataset()
        
        self.initialize_models()
        self.configure_optimizer()
        
        self.buffer_memory = PPOMemory()
        
        self.saveModelConfig()
        self.criterion = SequenceCrossEntropyLoss()

        self.delta=delta,
        self.beta1=beta1,
        self.beta2=beta2,
        self.beta3=beta3,
        self.beta4=beta4,
        self.beta5=beta5,
        self.beta6=beta6,
        self.beta7=beta7,
        self.beta8=beta8,

        self.x_num = {}

        
        self.emp_classifier_filename = emp_classifier_filename
        self.pol_classifier_filename = pol_classifier_filename

        self.ea_emp_classifier_filename = ea_emp_classifier_filename
        self.pa_pol_classifier_filename = pa_pol_classifier_filename



    def initialize_classifier_models(self):
        model_dict = torch.load(self.emp_classifier_filename)
        self.empathy_classifier = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=self.num_labels)
        self.politeness_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.empathy_classifier.config.problem_type = 'single_label_classification'
        self.empathy_classifier.load_state_dict(model_dict['state_dict'])
        self.empathy_classifier = self.empathy_classifier.to(self.device)
        self.empathy_classifier.eval()
        print('empathy Classifier Loaded! (in Evaluation Mode)')


    def saveModelConfig(self):
        if self.train_single_model:
            config_model_train = self.single_model_to_train
            print('Training Only :', self.single_model_to_train)
        else:
            config_model_train = 'Both Models being Trained.'
            print('Both Models being Trained.')
        config = {'Basic Info': [datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S")],
                  'NOTES': 'gpt2-medium',
                  'modelname': self.modelname,
                  'Training only one Model': self.train_single_model,
                  'Training Models': config_model_train,
                  'empathy_classifier': self.use_empathy_classifier,
                  'politeness_classifier': self.use_politeness_classifier,
                  'ea_empathy_classifier': self.use_ea_empathy_classifier,
                  'pa_politeness_classifier': self.use_pa_politeness_classifier,
                  'num_labels_empathy': self.num_labels,
                  'num_labels_politeness': self.num_pol_labels,
                  'alpha': self.alpha,
                  'gamma': self.gamma,
                  'delta': self.delta,
                  'device': self.device,
                  'use_jacc': self.use_jacc,
                  'use_uep_cons': self.use_uep_cons,
                  'use_uee_cons': self.use_uee_cons,
                  'use_dial_flow_cons': self.use_dial_flow_cons,
                  'modelLoaded': self.loadFilename,
                  'human_reward': self.human_reward,
                  'average_sent_loss' : self.average_sent_loss,
                  'n_epochs': self.n_epochs,
                  'use_recent_past': self.use_recent_past,
                  'temempature': self.temempature,
                  'learning_rate': self.learning_rate,
                  'epsilon': self.epsilon,
                  'num_candidates': self.num_candidates,
                  'pad_token_id': self.pad_token_id,
                  'max_candidate_length': self.max_candidate_length,
                  'recompute_log_prob': self.recompute_log_prob,
                  'evaluate_every': self.evaluate_every,
                  'top_p': self.top_p,
                  'warmup_steps': self.warmup_steps,
                  'batch_size':self.batch_size,
                  'seed': self.seedvalue}
        configfilename = os.path.join(self.savefolder, self.modelname, 'config')
        if not os.path.exists(configfilename):
            os.makedirs(configfilename)
        configfilename = configfilename + '/config' + '_' + self.modelname + '.json'
        with open(configfilename, 'w') as f:
            json.dump(config, f)

    def seed(self,seed=10):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def extract_data(self, csvfile):
        df = pd.read_csv(csvfile)
        data = []
        emotion = []
        for i in tqdm.trange(len(df)):
            if df['speaker'][i] == 'T':
                text = "A:" + str(df["text"][i])
                emotion.append(str(df["emotion"][i]))
                if self.empathy_classifier and self.politeness_classifier:
                    empathy_id = int(df['Empathy'][i])
                    politeness_id = int(df['Politeness'][i])
                    tup = (text, empathy_id, politeness_id)
                elif self.empathy_classifier and not self.politeness_classifier:
                    empathy_id = int(df['Empathy'][i])
                    tup = (text, empathy_id)
                else:
                    tup = (text)
            else:
                text = "B:" + str(df["text"][i])
                emotion.append(str(df["emotion"][i]))
                if self.empathy_classifier and self.politeness_classifier:
                    empathy_id = None
                    politeness_id = None
                    tup = (text, empathy_id, politeness_id)
                elif self.empathy_classifier and not self.politeness_classifier:
                    empathy_id = None
                    tup = (text, empathy_id)
                else:
                    tup = (text)
            data.append(tup)
        return data, emotion
        
    def utteranceToConversation(self, csvfile, data):
      df = pd.read_csv(self.csvfile)
      values=df['conv_id'].unique().tolist()
      conv_ids = df['conv_id'].tolist()

        dataset = []
        conversation = []
        emotion_set = []
        emotion_conversation = []
        for conv in values:
          for i in range(0, df.shape[0]):
            if(conv_ids[i]==conv):
              conversation.append(data[i])
              emotion_conversation.append(emotion[i])
            else:
              continue
          dataset.append(conversation)
          emotion_set.append(emotion_conversation)
          conversation = []
          emotion_conversation = []
        
      return dataset, emotion_set 
          
    def convertDicttoList(self, data: dict):
        return list(data.values())

    def random_split_data(self, data):
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        train_data = [data[idx] for idx in indices[:nc]] # nc: number of conversations in trrain set 
        val_data = [data[idx] for idx in indices[nc:]]

        train_emotion = [emotion[idx] for idx in indices[:nc]]
        val_emotion = [emotion[idx] for idx in indices[nc:]]
        
        return train_data, train_emotion, val_data, val_emotion

    def getDataset(self):
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

        data, emotion = self.extract_data(self.csvfile)
        data, emotion = self.utteranceToConversation(self.csvfile, data)
        
        self.traindata, self.trainemotion, self.valdata, self.valemotion = self.random_split_data(data)
        

        if self.empathy_classifier and self.politeness_classifier:
            use_empathy_labels=True
            use_politeness_labels=True
        elif  not self.empathy_classifier and self.politeness_classifier:
            use_empathy_labels=False
            use_politeness_labels=True
        elif self.empathy_classifier and not self.politeness_classifier:
            use_empathy_labels=True
            use_politeness_labels=False
        else:
            use_empathy_labels=False
            use_politeness_labels=False
        
        traindata_ = Counselling_Dataset(self.traindata, self.trainemotion,
                                     self.tokenizer,
                                     use_empathy_labels=use_empathy_labels,
                                     use_politeness_labels=use_politeness_labels)
        
        self.turn_ending = traindata_.get_turn_ending()
        
        valdata_ = Counselling_Dataset(self.valdata, self.valemotion,
                                   self.tokenizer,
                                   use_empathy_labels=use_empathy_labels,
                                   use_politeness_labels=use_politeness_labels)
        
        self.train_dataloader = DataLoader(dataset=traindata_,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           collate_fn=traindata_.collate)
        
        self.val_dataloader = DataLoader(dataset=valdata_,
                                         shuffle=False,
                                         batch_size=self.batch_size,
                                         collate_fn=valdata_.collate)

    def initialize_models(self):
        if not self.train_single_model:
            self.model_A = GPT2LMHeadModel.from_pretrained("gpt2-medium")
            self.model_B = GPT2LMHeadModel.from_pretrained("gpt2-medium")
            self.model_A_ref = GPT2LMHeadModel.from_pretrained("gpt2-medium")
            self.model_B_ref = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        else:
            if self.single_model_to_train == 'counsellor':
                self.model_A = GPT2LMHeadModel.from_pretrained("gpt2-medium")
                self.model_A_ref = GPT2LMHeadModel.from_pretrained("gpt2-medium")
            else:
                self._model_B = GPT2LMHeadModel.from_pretrained("gpt2-medium")
                self.model_B_ref = GPT2LMHeadModel.from_pretrained("gpt2-medium")

        if self.loadModel:
            if self.loadFilename:
                model_A_state_dict, model_B_state_dict = torch.load(self.loadFilename)#, map_location=self.device)
                if not self.train_single_model:
                    self.model_A.load_state_dict(model_A_state_dict)
                    self.model_A_ref.load_state_dict(model_A_state_dict)
                    self.model_B.load_state_dict(model_B_state_dict)
                    self.model_B_ref.load_state_dict(model_B_state_dict)
                    self.model_A = self.model_A.to(self.device)
                    self.model_A_ref = self.model_A_ref.to(self.device)
                    self.model_B = self.model_B.to(self.device)
                    self.model_B_ref = self.model_B_ref.to(self.device)
                    self.model_A.train()
                    self.model_B.train()
                    self.model_A_ref.eval()
                    self.model_B_ref.eval()
                else:
                    if self.single_model_to_train == 'counsellor':
                        self.model_A.load_state_dict(model_A_state_dict)
                        self.model_A_ref.load_state_dict(model_A_state_dict)
                        self.model_A = self.model_A.to(self.device)
                        self.model_A_ref = self.model_A_ref.to(self.device)
                        self.model_A.train()
                        self.model_A_ref.eval()
                        #self.model_B.load_state_dict(model_B_state_dict) 
                        #self.model_B = self.model_B.to('cuda')
                        #self.model_B.eval()
                        self.model_B = None
                        self.model_B_ref = None
                    else:
                        self.model_B.load_state_dict(model_B_state_dict)
                        self.model_B_ref.load_state_dict(model_B_state_dict)
                        self.model_B = self.model_B.to(self.device)
                        self.model_B_ref = self.model_B_ref.to(self.device)
                        self.model_B.train()
                        self.model_B_ref.eval()
                        self.model_A = None
                        self.model_A_ref = None
                print('\n')
                print("Models loaded from file ", self.loadFilename)
            else:
                print('Models not loaded since directory not provided.')
        print(f"Models Initalized!")
        print('\n')


    def configure_optimizer(self):
        
        self.num_train_optimization_steps = self.n_epochs * len(self.traindata) # // self.batch_size

        if not self.train_single_model:
            param_optimizer = list(self.model_A.named_parameters()) + list(self.model_B.named_parameters())
        else:
            if self.single_model_to_train == 'counsellor':
                param_optimizer = list(self.model_A.named_parameters())
        no_decay = ['bias', 'ln', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = optimizer = AdamW(optimizer_grouped_parameters,
                                           lr=self.learning_rate,
                                           eps=1e-06)

        #self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
        #                                                 num_warmup_steps=self.warmup_steps,
        #                                                 num_training_steps=self.num_train_optimization_steps)

        '''self.scheduler = WarmupLinearSchedule(self.optimizer,
                                                 warmup_steps=self.warmup_steps,
                                                 t_total=self.num_train_optimization_steps)'''


    def get_candidate_lengths(self, candidate_dict):

        avg_iter_length = []
        for i in candidate_dict:
            for j in candidate_dict[i]:
                 avg_iter_length.append(len(j.split()))
        #print(f"Average Candidate Length for {self.num_candidates} candidates generated at each utterance is {np.mean(avg_iter_length)}.")
        return avg_iter_length


    def validate_model(self, dataloader):

        with torch.no_grad():
            if not self.train_single_model:
                self.model_A.eval()
                self.model_B.eval()
            else:
                if self.single_model_to_train == 'counsellor':
                    self.model_A.eval()
                else:
                    self.model_B.eval()

            with torch.no_grad():
                
                progress_bar = tqdm_notebook
                pbar = progress_bar(dataloader)
               
                total_ppl = []
                total_loss = []
                candidates_dict = {}

                for batch in pbar:

                    if sum([len(item) for item in batch[0][1]]) > 1024:
                        continue

                    if not self.empathy_classifier and not self.politeness_classifier:
                        role_ids, dialog_tokens = batch[0]
                    elif self.empathy_classifier and self.politeness_classifier:
                        role_ids, dialog_tokens, empathy_label, politeness_label = batch[0]
                    elif self.empathy_classifier and not self.politeness_classifier:
                        role_ids, dialog_tokens, empathy_label = batch[0]
                    elif not self.empathy_classifier and self.politeness_classifier:
                        role_ids, dialog_tokens, politeness_label = batch[0]
                    
                    #dial_inputs = [torch.LongTensor(item).unsqueeze(0).to(self.device) for item in dialog_tokens]
                    dial_inputs = [torch.LongTensor(item).unsqueeze(0) for item in dialog_tokens]
                    past = None
                    past_ = None
                    all_logits = []
                    target = []

                    for turn_num, dial_turn_inputs in enumerate(dial_inputs):

                        if not self.train_single_model:
                            if role_ids[turn_num] == 0:
                                outputs = self.model_A(dial_turn_inputs, past_key_values=past, return_dict=False)
                                past = outputs[1]
                                all_logits.append(outputs[0])
                            else:
                                outputs = self.model_B(dial_turn_inputs, past_key_values=past, return_dict=False)
                                past = outputs[1]
                                all_logits.append(outputs[0])
                        else:
                            if self.single_model_to_train == 'counsellor':
                                if role_ids[turn_num] == 0:
                                    # dial_turn_str = convert_sentences_to_strings([dial_turn_inputs], self.tokenizer)[0].split('\t')[0]
                                    dial_turn_inputs = dial_turn_inputs.to(self.device)
                                    outputs = self.model_A(dial_turn_inputs, past_key_values=past, return_dict=False)
                                    past = outputs[1]
                                    all_logits.append(outputs[0])
                                    target.append(dial_turn_inputs)

                    all_logits = torch.cat(all_logits, dim=1)
                    all_logits = all_logits[:, :-1].contiguous()

                    if not self.train_single_model:
                        target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()
                    else:
                        target = torch.cat(target, dim=1)[:, 1:].contiguous()
                    
                    target_mask = torch.ones_like(target).float()

                    loss = self.criterion(all_logits, target, target_mask, label_smoothing=-1, reduce='sentence')
                    total_loss.extend(loss.tolist())

                    ppl = torch.exp(loss)
                    total_ppl.extend(ppl.tolist())
                    
                print('\n')
                print(f"Validation perplexity: {np.mean(total_ppl)}")

                # average_lengths = self.get_candidate_lengths(candidates_dict)
                # print(f"Average candidate length: {np.mean(average_lengths)}")
                

        # return np.mean(total_ppl), np.mean(total_loss), np.mean(average_lengths), num_strategy*100
        return np.mean(total_ppl), np.mean(total_loss)
    

    def make_stats_dir(self):
        
        self.statsfolder = os.path.join(os.getcwd(), self.savefolder, self.modelname, 'stats')
        if not os.path.exists(self.statsfolder):
            os.makedirs(self.statsfolder)


    def make_model_save_dir(self):
        
        self.savefolder = os.path.join(os.getcwd(), 'models/RL_ARDM_medium')
        if not os.path.exists(self.savefolder):
            print("Model save folder doesn't exist.")
            os.makedirs(self.savefolder)
            print(f"Created folder {self.savefolder} to save the models.")


    def save_models(self, num_iter):
        
        modeldir = os.path.join(self.savefolder, self.modelname)
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)
            print('Created Directory for saving models!')
        filename = modeldir + '/' + self.modelname + '_' + str(num_iter) + ".pth"
        #torch.save([self.model_A.state_dict(), self.model_B.state_dict()], filename)
        torch.save(self.model_A.state_dict(), filename)

    def modified_train_one_iter(self, batch):
        dial_inputs, role_ids, scores_dict = collect_samples(batch,
                                                             model_A=self.model_A_ref,
                                                             model_B=self.model_B,
                                                             top_p=self.top_p,
                                                             eos_token_id=self.turn_ending[0],
                                                             pad_token_id=self.turn_ending[1],
                                                             average_sent_loss=self.average_sent_loss,
                                                             max_gen_length=self.max_candidate_length,
                                                             buffer_memory=self.buffer_memory,
                                                             use_jacc=self.use_jacc,
                                                             use_uep_cons=self.use_uep_cons,
                                                             use_uee_cons=self.use_uee_cons,
                                                             use_dial_flow_cons=self.use_dial_flow_cons,
                                                             device=self.device,
                                                             num_candidates=self.num_candidates,
                                                             human_reward=self.human_reward,
                                                             politeness_tokenizer=self.politeness_tokenizer,
                                                             empathy_classifier=self.empathy_classifier,
                                                             politeness_classifier=self.politeness_classifier,
                                                             ea_empathy_classifier=self.ea_empathy_classifier,
                                                             pa_politeness_classifier=self.pa_politeness_classifier,
                                                             alpha=self.alpha,
                                                             gamma=self.gamma,
                                                             delta=self.delta
                                                             tokenizer=self.tokenizer,
                                                             criterion=self.criterion,
                                                             temempature=self.temempature,
                                                             use_recent_past=self.use_recent_past,
                                                             recompute_log_prob=self.recompute_log_prob,
                                                             nlp=self.nlp,
                                                             train_single_model=self.train_single_model,
                                                             model_to_train=self.single_model_to_train,
                                                             beta1 = self.beta1,
                                                             beta2 = self.beta2,
                                                             beta3 = self.beta3,
                                                             beta4 = self.beta4,
                                                             beta5 = self.beta5,
                                                             beta6 = self.beta6,
                                                             beta7 = self.beta7,
                                                             beta8 = self.beta8)

        log_dict = ppo_step(model_A=self.model_A,
                            model_B=self.model_B,
                            buffer_memory=self.buffer_memory,
                            train_single_model=self.train_single_model,
                            dial_inputs= dial_inputs,
                            model_to_train=self.single_model_to_train,
                            device=self.device,
                            ppo_epsilon=self.epsilon,
                            num_candidates=self.num_candidates,
                            use_recent_past=self.use_recent_past,
                            average_sent_loss=self.average_sent_loss,
                            criterion=self.criterion,
                            optimizer=self.optimizer,
                            role_ids=role_ids)

        self.buffer_memory.clear_memory()

        return log_dict, scores_dict 
 
    def train(self):

        update_count = 0
        progress_bar = tqdm_notebook

        val_ppl = []
        val_loss = []

        rewards = []
        kl = []
        clip_frac = []


        uep_consistency_scores = []
        uee_consistency_scores = []    
        dial_flow_consistency_scores = []
        jacc_scores = []
        pa_politeness_scores = []
        ea_empathy_scores = []
        empathy_scores = []
        politeness_scores = []
        
        emp_actual_probs = []
        emp_other_probs = []

        pol_actual_probs = []
        pol_other_probs = []
        


        best_ppl = None
        
        #length = None
        
        iters = None
        


        pbar = progress_bar(self.train_dataloader)

        for i in range(self.n_epochs):
            if not self.train_single_model:
                self.model_A.train()
                self.model_B.train()
            else:
                if self.single_model_to_train == 'counsellor':
                    self.model_A.train()
            for batch in pbar:
                if sum([len(item) for item in batch[0][1]]) > 1024 - self.max_candidate_length:
                    continue

                print(f"ITERATION: {update_count}")

                batch = batch[0]
                log_dict, scores_dict  = self.modified_train_one_iter(batch)

                clip_frac.append(log_dict['clip_frac'])
                kl.append(log_dict['approx_kl'])
                rewards.append(log_dict['reward'])


                uep_consistency_scores.extend(scores_dict['uep_consistency_scores'])
                uee_consistency_scores.extend(scores_dict['uee_consistency_scores'])    
                dial_flow_consistency_scores.extend(scores_dict['dial_flow_consistency_scores'])
                jacc_scores.extend(scores_dict['jacc_scores'])
                pa_politeness_scores.extend(scores_dict['pa_politeness_scores'])
                ea_empathy_scores.extend(scores_dict['ea_empathy_scores'])

                empathy_scores.extend(scores_dict['empathy_scores'])
                politeness_scores.extend(scores_dict['politeness_scores'])

                
                emp_actual_probs.extend(scores_dict['empathy_actual_prob'])
                emp_other_probs.extend(scores_dict['empathy_other_prob'])

                pol_actual_probs.extend(scores_dict['politeness_actual_prob']) 
                pol_other_probs.extend(scores_dict['politeness_other_prob'])

                # np.save(self.statsfolder + '/' + 'cos_sim_scores.npy', np.array(cos_sim_scores))
                np.save(self.statsfolder + '/' + 'uep_consistency_scores.npy', np.array(uep_consistency_scores))
                np.save(self.statsfolder + '/' + 'pa_politeness_scores.npy', np.array(pa_politeness_scores))
                np.save(self.statsfolder + '/' + 'uee_consistency_scores.npy', np.array(uee_consistency_scores))
                np.save(self.statsfolder + '/' + 'ea_empathy_scores.npy', np.array(ea_empathy_scores))
                np.save(self.statsfolder + '/' + 'diversity_scores.npy', np.array(jacc_scores))
                np.save(self.statsfolder + '/' + 'dial_flow_consistency_scores.npy', np.array(dial_flow_consistency_scores))

                if self.empathy_classifier:
                    np.save(self.statsfolder + '/' + 'empathy_scores.npy', np.array(empathy_scores))
                    np.save(self.statsfolder + '/' + 'empathy_actual_prob.npy', np.array(emp_actual_probs))
                    np.save(self.statsfolder + '/' + 'empathy_other_prob.npy', np.array(emp_other_probs))
                if self.politeness_classifier:
                    np.save(self.statsfolder + '/' + 'politeness_scores.npy', np.array(politeness_scores))
                    np.save(self.statsfolder + '/' + 'politeness_actual_prob.npy', np.array(pol_actual_probs))
                    np.save(self.statsfolder + '/' + 'politeness_other_prob.npy', np.array(pol_other_probs))
                
                update_count += 1

                if  update_count % self.evaluate_every == 0:
                    #ppl, loss, average_length, empcent_strategy = self.validate_model(self.val_dataloader)
                    
                    ppl, loss = self.validate_model(self.val_dataloader)
                    
                    if best_ppl is None:

                        best_ppl = ppl
                        iters = update_count
                        
                        #strategies = empcent_strategy
                        #length = average_length
                        
                        if update_count > 20 and update_count < 22:
                          self.save_models(iters)
                          print(f'Saving Model at {iters}')
                        
                        '''filename = self.statsfolder + '/strategy_count_num_dict.json'
                        with open(filename, 'w') as f:
                            json.dump(self.count_dict_num, f)

                        filename = self.statsfolder + '/strategy_count_str_dict.json'
                        with open(filename, 'w') as f:
                            json.dump(self.count_dict_str, f)'''
                    else:
                        if ppl < best_ppl:
                            best_ppl = ppl
                            iters = update_count
                            
                            #strategies = empcent_strategy
                            #length = average_length
                            
                        if update_count > 20 and update_count < 22:
                          self.save_models(iters)
                          print(f'Saving Model at {iters}')
                
                    print('\n')
                    print(f'Best perplexity Found so far {best_ppl} for iteration: {iters}')
                    print('\n')
                    
                    val_ppl.append(ppl)
                    val_loss.append(loss)
                    
                    #candidate_lengths.append(average_length)
                    #empcent_candidates_with_strategy.append(empcent_strategy)
                                
                    np.save(self.statsfolder + '/' + 'val_PPL_iter'  + '.npy', np.array(val_ppl))
                    
                    #np.save(self.statsfolder + '/' + 'val_cand_length'  + '.npy', np.array(average_length)) 
                    #np.save(self.statsfolder + '/' + 'val_empcent_strategy' + '.npy', np.array(empcent_candidates_with_strategy))
                    
                    np.save(self.statsfolder + '/' + 'train_rewards' + '.npy', np.array(rewards))
                    np.save(self.statsfolder + '/' + 'train_kl' + '.npy', np.array(kl))
                    np.save(self.statsfolder + '/' + 'train_clip_frac' + '.npy', np.array(clip_frac))
                    np.save(self.statsfolder + '/' + 'best_ppl_iteration_value' + '.npy', np.array(iters))
                    
                    #np.save(self.statsfolder + '/' + 'best_ppl_empcent_strategy' + '.npy', np.array(strategies))

                    #self.initialize_strategy_count()
    
                    if not self.train_single_model:
                        self.model_A.train()
                        self.model_B.train()
                    else:
                        if self.single_model_to_train == 'counsellor':
                            self.model_A.train()
                #if update_count == 17:
                #    return best_ppl, iters
        return best_ppl, iters

if __name__ == '__main__':
    trainer = Trainer(modelname='PATH_TO_TRAINED_RL_MODEL',
                      csvfile="PATH_TO_DATASET",
                      device='cuda',
                      n_epochs=1,
                      batch_size=1,
                      mini_batch=20,
                      train_single_model=True,
                      single_model_to_train= 'counsellor',
                      num_candidates=3,
                      recompute_log_prob=True,
                      average_sent_loss=True,
                      max_candidate_length=50,
                      human_reward=10,
                      alpha=2.0,
                      gamma=2.0,
                      delta=2.0,
                      top_p=0.9,
                      temempature=0.8,
                      use_recent_past=True,
                      warmup_steps=10,
                      print_every=1,
                      evaluate_every=1,
                      learning_rate=2e-5,
                      epsilon=0.2,
                      loadModel=True,
                      loadFilename="PATH_TO_CELDM_TRAINED_MODEL",
                      pad_token_id=2,
                      seedvalue=10, # 10 should be the seed value since pre trained on the same seed. 
                      use_empathy_classifier=True,
                      use_ea_empathy_classifier=True,
                      use_politeness_classifier=True,
                      use_pa_politeness_classifier=True,
                      emp_classifier_filename= "PATH_TO_EMPATHETIC_CLASSIFIER",
                      pol_classifier_filename="PATH_TO_POLITENESS_CLASSIFIER",
                      ea_emp_classifier_filename= "PATH_TO_EA_EMPATHETIC_CLASSIFIER",
                      pa_pol_classifier_filename="PATH_TO_PA_POLITENESS_CLASSIFIER",
                      emp_num_labels=3,
                      pol_num_labels=3,
                      use_jacc=True,
                      use_uep_cons=True,
                      use_uee_cons=True,
                      use_dial_flow_cons=True,
                      beta1=0.1,
                      beta2=0.1,
                      beta3=0.2,
                      beta4=0.2,
                      beta5=0.1,
                      beta6=0.1,
                      beta7=0.1,
                      beta8=0.1)
    trainer.train()

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import logging

import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# In[5]:


train_df = pd.read_csv("Path_of_train_data.csv")


# In[7]:


eval_df = pd.read_csv("Path_of_val_data.csv")


model_args = Seq2SeqArgs()
model_args.num_train_epochs = 6
model_args.no_save = False
model_args.use_multiprocessing = False
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.overwrite_output_dir = True
model_args.train_batch_size = 4
model_args.eval_batch_size = 4
model_args.max_length = 128

# Initialize model
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-large",
    args=model_args,
    use_cuda=True

)

# In[11]:

def count_matches(labels, preds):
    print(labels)
    print(preds)
    return sum(
        [
            1 if label == pred else 0
            for label, pred in zip(labels, preds)
        ]
    )


# Train the model
model.train_model(
    train_df, eval_data=eval_df, matches=count_matches
)

# # Evaluate the model
results = model.eval_model(eval_df)


The structure of the entire setup is as follows:

```
|___________/Codes
|	       |___ /CELDM
|	       		|___ CELDM.py				# script to train cross-entropy loss dialogue model.
|
|	       |___ /PAL
|	       		|___ dataset.py				# script to load the custom dataset by creation of a pytorch Dataset class
|	       		|___ rlmain.py				# script to fine-tune CELDM with RL loss
|	       		|___ rlutils.py             # script containing utility functions and reward functions for the RL fine-tuning task
|               |___ ppo.py 				# script containing implementation of buffer memory
|               |___ loss.py 				# script containing implementation of Sequence Cross Entropy Loss
|               |___ rltest.py 				# script to interact with the fine-tuned model.
|
|
|	       |___ /CLASSIFIER
|                       |___ classifier.py 		 # script to train the classifiers
|          |___ /SEQ2SEQ
|                       |___ seq2seq.py          # script to train the seq2seq models
|

```

****REQUIREMENTS****
1. numpy: version '1.21.2'
2. pandas: version '1.3.4'
3. transformers: version '4.11.2'
4. tqdm: version: version '4.62.3'
5. torch: version '1.10.0'


****FINE-TUNING RL MODEL****

1. Provide all the arguments in the "rlmain.py" file.
2. Go to terminal window and enter "python rlmain.py" for WindowsOS or "python3 rlmain.py" for UNIX-based OS to start the RL fine-tuning.

Args:

modelname:str, 'the desired modelname',
csvfile:str, the csv file to load the annotated dataset from
device:str, Default='cuda'
n_epochs:int, Default=1
batch_size:int, Default=1
mini_batch=int, Default=1
train_single_model:bool, Whether to fine-tune both agent and user or either one of them during RL fine tuning, Default=True
single_model_to_train:str, Which model of train 'agent' or 'user', Default:'agent',
num_candidates:int, number of candidates to generate at a turn for the agent, Default=3
recompute_log_prob:bool, Whether to recompute the log probability of the generated candidates, Default= True
average_sent_loss:bool, Whether to average the loss the over the entire sentence for the generated candidates, Default=True
max_candidate_length:int, Maximum length of generated candidates, Default=50
human_reward:int, Default=10
alpha:float, Default=2
gamma:float, Default=2
delta:float, Default=2
top_p:float, The probability sum threshold to consider when generating tokens for the candidates,  Default=0.9
temperature:float, The temprate value when calculating the loss, Default=0.8
use_recent_past:bool, Whether to consider the recent past
warmup_steps:int, number of warm up step to be given to the scheduler, Default=10
print_every:int, number of steps before printing the loss Default=1
evaluate_every:int, Iterations before evaluation, Default=1
learning_rate:float, Default=2e-05
epsilon:float, Default=0.2
loadModel:bool, Whether to load the pretrained language model for fine-tuning, Default=True
loadFilename:str, path to the saved pretrained language model
pad_token_id:int, Default=2
seedvalue:int, Default=10
use_jacc:bool, Whether to use diversity reward or not, Default=True
use_uep_cons:bool, Whether to use utterance-emotion-politeness consistency reward or not, Default=True
use_uee_cons:bool, Whether to use utterance-emotion-empathy consistency reward or not, Default=True
use_dial_flow_cons:bool, Whether to use dialogue flow consistency reward or not, Default=True
use_empathy_classifier:bool, Whether to use empathy classifier and respective reward or not, Default = True 
use_ea_empathy_classifier:bool, Whether to use ea_empathy classifier and respective reward or not, Default = True
use_politeness_classifier:bool, Whether to use politeness classifier and respective reward or not, Default = True
use_pa_politeness_classifier:bool, Whether to use pa_politeness classifier and respective reward or not, Default = True
emp_classifier_filename:PATH_TO_EMPATHETIC_CLASSIFIER
pol_classifier_filename:PATH_TO_POLITENESS_CLASSIFIER
ea_emp_classifier_filename:PATH_TO_EA_EMPATHETIC_CLASSIFIER
pa_pol_classifier_filename:PATH_TO_PA_POLITENESS_CLASSIFIER
beta1:float, weight for the utterance-emotion-politeness consistency reward, Default=0.1
beta2:float, weight for the utterance-emotion-empathy reward, Default=0.1
beta3:float, weight for the politeness adaptiveness reward, Default=0.2
beta4:float, weight for the empathy adaptiveness reward, Default=0.2
beta5:float, weight for the politeness correctness reward, Default=0.1
beta6:float, weight for the empathy correctness reward, Default=0.1
beta7:float, weight for the dialogue flow consistency reward, Default=0.1
beta8:float, weight for the diversity reward, Default=0.1


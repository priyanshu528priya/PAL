import numpy as np
import torch.nn as nn
from nltk.translate.meteor_score import meteor_score
import nltk
#nltk.download('wordnet')
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import functools
import operator
import pdb
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

warnings.filterwarnings("ignore")

model_pol = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name = "PATH_TO_THE_TRAINED_UEP_SEQ2SEQ_MODEL",
    use_cuda=True

)

model_emp = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name = "PATH_TO_THE_TRAINED_UEE_SEQ2SEQ_MODEL",
    use_cuda=True

)

model_cos = SentenceTransformer('bert-base-nli-mean-tokens')

def convert_sentences_to_strings(sentences:list, tokenizer):
    str_sentences = []
    for i in sentences:
        str_sentences.append(tokenizer.decode(i.tolist()[0][2:-2])) # Excludeqs the zero shot tokens: {A:, B:} and the End of turn tokens: [628, 198]
    return str_sentences

def normalize(text, nlp):
    sent = ''
    doc = nlp(text)
    for token in doc:
        if not token.is_punct:
            sent += token.lemma_
            sent += ' '
    return sent

def jacc_sim(context_sentence_list, generated_sentence, tokenizer, nlp):
    str1 = context_sentence_list[0]
    str1 = normalize(str1, nlp)
    str1 = set(str1.split())
    jacc_dis = []
    generated_sentences = convert_sentences_to_strings(generated_sentence, tokenizer)
    for i in generated_sentences:
        str2 = i
        str2 = normalize(str2, nlp)
        str2 = set(str2.split())
        sim_score = 1-(float(len(str1 & str2)) / len(str1 | str2))
        jacc_dis.append(sim_score)
    return jacc_dis

def dialogue_length(candidates, tokenizer, length, num_turn, dial_inputs):
    num = 0
    for i in candidates:
        # Responses of A
        candidate = tokenizer.decode(i.tolist()[0][2:]).split('\n')[0]
        candidate_sentence = candidate.split('\t')[0]
        
        if(num_turn>4):
            prev_sentence = tokenizer.decode(dial_inputs[num_turn-2].tolist()[0][2:]).split('\n')[0].split('\t')[0]
            prev_sentence2 = tokenizer.decode(dial_inputs[num_turn-4].tolist()[0][2:]).split('\n')[0].split('\t')[0]
        else:
            length[num] = num_turn
            num = num+1
            return length
        
        # with (i, i-1)
        turn = []
        turn.append(candidate_sentence)
        turn.append(prev_sentence)
        turn=model.encode(turn)
        cos_sim_1 = cosine_similarity([turn[0]], turn[1:])[0][0]

        # with (i,i-2)
        turn = []
        turn.append(candidate_sentence)
        turn.append(prev_sentence2)
        turn=model.encode(turn)
        cos_sim_2 = cosine_similarity([turn[0]], turn[1:])[0][0]

        cos_sim_A = 0.5*(cos_sim_1+cos_sim_2)

        # Responses of B
        if(num_turn>=1):
            candidate_sentence = tokenizer.decode(dial_inputs[num_turn-1].tolist()[0][2:]).split('\n')[0].split('\t')[0]
        else:
            length[num] = num_turn
            num = num+1
            return length

        if(num_turn>5):
            prev_sentence = tokenizer.decode(dial_inputs[num_turn-3].tolist()[0][2:]).split('\n')[0].split('\t')[0]
            prev_sentence2 = tokenizer.decode(dial_inputs[num_turn-5].tolist()[0][2:]).split('\n')[0].split('\t')[0]
        else:
            length[num] = num_turn
            num = num+1
            return length

        # with (i, i-1)
        turn = []
        turn.append(candidate_sentence)
        turn.append(prev_sentence)
        turn=model.encode(turn)
        cos_sim_1 = cosine_similarity([turn[0]], turn[1:])[0][0]

        # with (i,i-2)
        turn = []
        turn.append(candidate_sentence)
        turn.append(prev_sentence2)
        turn=model.encode(turn)
        cos_sim_2 = cosine_similarity([turn[0]], turn[1:])[0][0]

        cos_sim_B = 0.5*(cos_sim_1+cos_sim_2)
        
        if(cos_sim_A<0.9 and cos_sim_B<0.9):
            length[num] = num_turn
            num = num+1
            return length
        else:
            num = num=1
            return length

def dialogue_flow_consistency(candidates, context_sentence_list, tokenizer, nlp):
    dialogue_flow_consistency_scores = []

    for i in candidates:
        candidate_sentence = tokenizer.decode(i.tolist()[0][2:]).split('\n')[0].split('\t')[0]
        ctxt1 = context_sentence_list[0]

        ctxt2 = context_sentence_list[0] + context_sentence_list[1]

        # with (i, r)
        turn = []
        turn.append(candidate_sentence)
        turn.append(ctxt1)
        turn=model_cos.encode(turn)
        cos_sim_1 = cosine_similarity([turn[0]], turn[1:])[0][0]

        # with (i-1, r)
        turn = []
        turn.append(candidate_sentence)
        turn.append(ctxt2)
        turn=model_cos.encode(turn)
        cos_sim_2 = cosine_similarity([turn[0]], turn[1:])[0][0]

        cos_sim_r = (cos_sim_1+0.5*(cos_sim_2))

        score = 0.5 * (min(cos_sim_r, 0.75))
        dialogue_flow_consistency_scores.append(score)
    
    return dialogue_flow_consistency_scores


def calculate_utt_emotion_politeness_consistency(candidates, current_sentence, next_sentence, num_turn, dial_inputs, tokenizer):
    utt_emo_pol_scores = []
    scores = []
    for i in candidates:
        candidate = tokenizer.decode(i.tolist()[0][2:]).split('\n')[0]

        list1 = []
        list1.append(current_sentence)
        list2=[]
        list2.append(next_sentence)
        list3=[]
        list3.append(candidate)

        eval_data1 = []
        eval_data1.append(list1)
        eval_data1.append(list2)
        eval_data2= []
        eval_data2.append(list1)
        eval_data2.append(list3)
        eval_loss1 = model_pol.eval_model(eval_data1)
        eval_loss2 = model_pol.eval_model(eval_data2)

        nll1 = eval_loss1['eval_loss']
        nll2 = eval_loss2['eval_loss']

        score1 = nll1 - alpha * nll2
        score = -(torch.tanh(score))
        scores.append(score)
        utt_emo_pol_scores.append(score)

    return utt_emo_pol_scores

def calculate_utt_emotion_empathy_consistency(candidates, current_sentence, next_sentence, num_turn, dial_inputs, tokenizer):
    utt_emo_emp_scores = []
    scores = []
    for i in candidates:
        candidate = tokenizer.decode(i.tolist()[0][2:]).split('\n')[0].split('\t')[0]

        list1 = []
        list1.append(current_sentence)
        list2=[]
        list2.append(next_sentence)
        list3=[]
        list3.append(candidate)

        eval_data1 = []
        eval_data1.append(list1)
        eval_data1.append(list2)
        eval_data2= []
        eval_data2.append(list1)
        eval_data2.append(list3)
        eval_loss1 = model_emp.eval_model(eval_data1)
        eval_loss2 = model_emp.eval_model(eval_data2)

        nll1 = eval_loss1['eval_loss']
        nll2 = eval_loss2['eval_loss']

        score1 = nll1 - alpha * nll2
        score = -(torch.tanh(score))
        scores.append(score)
        utt_emo_emp_scores.append(score)

    return utt_emo_emp_scores



def politeness_adaptiveness(current_sentence, candidates, pa_politeness_classifier, politeness_classifier, device, tokenizer, politeness_tokenizer):
    
    pa_pol_scores = []

    input1 = politeness_tokenizer(current_sentence, return_tensors='pt', padding=True, truncation=True)
    
    output1 = pa_politeness_classifier(input1['input_ids'].to(device), input1['attention_mask'].to(device))
    
    probs1 = F.softmax(output1.logits, dim=-1)

    pa_pol_class_prob = max(probs1)

    pa_pol_class = argmax(probs1)

    for i in candidates:
        candidate = tokenizer.decode(i.tolist()[0][2:]).split('\n')[0].split('\t')[0]

        inputs = politeness_tokenizer(candidate, return_tensors='pt', padding=True, truncation=True)

        output = politeness_classifier(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))

        probs = F.softmax(output.logits, dim=-1)

        pol_class_prob = probs[pa_pol_class]

        score = pa_pol_class_prob - gamma * pol_class_prob
        scores.append(score)
        pa_pol_scores.append(score)

    return pa_pol_scores


def empathy_adaptiveness(current_sentence, candidates, ea_empathy_classifier, emmpathy_classifier, device, tokenizer, empathy_tokenizer):
    
    ea_emp_scores = []

    input1 = empathy_tokenizer(current_sentence, return_tensors='pt', padding=True, truncation=True)
    
    output1 = ea_empathy_classifier(input1['input_ids'].to(device), input1['attention_mask'].to(device))
    
    probs1 = F.softmax(output1.logits, dim=-1)

    ea_emp_class_prob = max(probs1)

    ea_emp_class = argmax(probs1)

    for i in candidates:
        candidate = tokenizer.decode(i.tolist()[0][2:]).split('\n')[0].split('\t')[0]

        inputs = empathy_tokenizer(candidate, return_tensors='pt', padding=True, truncation=True)

        output = empathy_classifier(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))

        probs = F.softmax(output.logits, dim=-1)

        emp_class_prob = probs[ea_emp_class]

        score = ea_emp_class_prob - gamma * emp_class_prob
        scores.append(score)
        ea_emp_scores.append(score)

    return ea_emp_scores

def politeness_correctness(generated_sentences, politeness_classifier,
                          actual_politeness_label,
                          device,
                          tokenizer,
                          politeness_tokenizer,
                          delta):
    
    inputs = politeness_tokenizer(generated_sentences, return_tensors='pt', padding=True, truncation=True)
    
    output = politeness_classifier(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))
    
    probs = F.softmax(output.logits, dim=-1)
    actual_label_prob = probs[:, actual_politeness_label]
    other_label_prob = probs.sum(-1) - actual_label_prob
    
    reward_dict = {'actual_prob': actual_label_prob.tolist(),
                   'other_prob': other_label_prob.tolist()}
    
    reward = actual_label_prob - delta * other_label_prob
    
    return reward_dict, reward.tolist()

def empathy_correctness(generated_sentences, empathy_classifier,
                          actual_empathy_label,
                          device,
                          tokenizer,
                          empathy_tokenizer,
                          delta):
    
    inputs = empathy_tokenizer(generated_sentences, return_tensors='pt', padding=True, truncation=True)
    
    output = empathy_classifier(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))
    
    probs = F.softmax(output.logits, dim=-1)
    actual_label_prob = probs[:, actual_empathy_label]
    other_label_prob = probs.sum(-1) - actual_label_prob
    
    reward_dict = {'actual_prob': actual_label_prob.tolist(),
                   'other_prob': other_label_prob.tolist()}
    
    reward = actual_label_prob - delta * other_label_prob
    
    return reward_dict, reward.tolist()

def calculate_rewards(model_A,
                      current_sentence,
                      next_sentence,
                      num_turn,
                      dial_inputs,
                      length,
                      generated_sentences,
                      source_list,
                      tokenizer,
                      criterion,
                      use_jacc,
                      use_uep_cons,
                      use_uee_cons,
                      use_dial_flow_cons,
                      nlp,
                      device,
                      alpha,
                      gamma,
                      delta,
                      politeness_tokenizer,
                      beta1,
                      beta2,
                      beta3,
                      beta4,
                      beta5,
                      beta6,
                      beta7,
                      beta8,
                      pa_politenes_classifier,
                      ea_empathy_classifier,
                      politenes_classifier,
                      actual_politeness_label,
                      counsellor=False,
                      empathy_classifier=None,
                      actual_empathy_label=None):
    
    scores = {}

    scores['uep_consistency'] = []
    scores['uee_consistency'] = []    
    scores['dial_flow_consistency'] = []
    scores['jacc'] = []
    scores['pa_politeness'] = []
    scores['ea_empathy'] = []
    scores['politeness'] = []
    scores['empathy'] = []
    scores['politeness_actual_prob'] = []
    scores['politeness_other_prob'] = []
    scores['empathy_actual_prob'] = []
    scores['empathy_other_prob'] = []

    if len(generated_sentences) >= 1:
        
        rewards = np.zeros((len(generated_sentences)))
        
        # if (len(source_list) ==2):
            
        if use_jacc: 
            non_rep = jacc_sim(source_list, generated_sentences, tokenizer, nlp)
            dial_length = np.array(length)

            non_rep = np.array(non_rep)

            diversity = dial_length - non_rep

            rewards -= beta8*(diversity)
        else: 
            diversity = None
        

        if use_uep_cons:
            uep_cons_scores = calculate_utt_emotion_politeness_consistency(generated_sentences, current_sentence, next_sentence, num_turn, dial_inputs, tokenizer)
            rewards+= beta1*np.array(uep_cons_scores)
        
        if not use_uep_cons:
            uep_cons_scores = None

        if use_uee_cons:
            uee_cons_scores = calculate_utt_emotion_empathy_consistency(generated_sentences, current_sentence, next_sentence, num_turn, dial_inputs, tokenizer)
            rewards+= beta2*np.array(uee_cons_scores)
        
        if not use_uee_cons:
            uee_cons_scores = None

        if use_dial_flow_cons:
            dial_flow_cons_scores = dialogue_flow_consistency(generated_sentences, source_list, tokenizer, nlp)
            rewards += beta7*np.array(dial_flow_cons_scores)
        
        if not use_dial_flow_cons:
            dial_flow_cons_scores = None
        
        if pa_politeness_classifier:   
            pa_politeness_probs = politeness_adaptiveness(current_sentence, generated_sentences, pa_politeness_classifier, politeness_classifier, device, tokenizer, politeness_tokenizer)
            rewards += beta3* np.array(pa_politeness_probs)
        
        if not pa_politeness_classifier:
            pa_politeness_probs = [None]

        if ea_empathy_classifier:   
            ea_empathy_probs = empathy_adaptiveness(current_sentence, generated_sentences, ea_empathy_classifier, empathy_classifier, device, tokenizer, empathy_tokenizer)
            rewards += beta4* np.array(ea_empathy_probs)
        
        if not ea_empathy_classifier:
            ea_empathy_probs = [None]
        
        if politeness_classifier:
            politeness_dict, politeness_probs = politeness_correctness(generated_sentences,
                                                                      politeness_classifier,
                                                                      actual_politeness_label,
                                                                      device, tokenizer, 
                                                                      politeness_tokenizer,
                                                                      delta)
            rewards += beta5* np.array(politeness_probs)

        if not politeness_classifier:
            politeness_dict =  {'actual_prob': [None], 'other_prob': [None]}
            politeness_probs = [None]

        if empathy_classifier:
            empathy_dict, empathy_probs = empathy_correctness(generated_sentences,
                                                                      empathy_classifier,
                                                                      actual_empathy_label,
                                                                      device, tokenizer, 
                                                                      politeness_tokenizer,
                                                                      delta)
            rewards += beta6* np.array(empathy_probs)

        if not empathy_classifier:
            empathy_dict =  {'actual_prob': [None], 'other_prob': [None]}
            empathy_probs = [None]


    else:
        rewards = 0
        jacc_dist = jacc_seim(current_sentence, generated_sentences, tokenizer, nlp)
        
        rewards -= jacc_dist
    try:
        scores['jacc'].extend(jacc_dist)
    except:
        pass
    
    scores['jacc'].extend(diversity.tolist())
    scores['uep_consistency'].extend(uep_cons_scores)
    scores['uee_consistency'].extend(uee_cons_scores)    
    scores['dial_flow_consistency'].extend(dial_flow_cons_scores)
    scores['pa_politeness'].extend(pa_politeness_probs)
    scores['ea_empathy'].extend(ea_empathy_probs)
    scores['politeness'] = []
    scores['empathy'] = []
    scores['politeness'].extend(politeness_probs)
    scores['empathy'].extend(empathy_probs)
    scores['politeness_actual_prob'].extend(politeness_dict['actual_prob'])
    scores['politeness_other_prob'].extend(politeness_dict['other_prob'])
    scores['empathy_actual_prob'].extend(empathy_dict['actual_prob'])
    scores['empathy_other_prob'].extend(empathy_dict['other_prob'])

    return list(rewards), scores

def append(generated_list, context_sentence, tokenizer):
    
    if len(generated_list) == 2:
        generated_list.pop(0)
        cntx = tokenizer.decode(context_sentence.tolist()[0][2:]).split('\n')[0]
        generated_list.append(cntx)
    else:
        cntx = tokenizer.decode(context_sentence.tolist()[0][2:]).split('\n')[0]
        generated_list.append(cntx)
    
    return generated_list

def expand_inputs_for_N_candidates(inputs, num_candidates):
    # inputs = inputs[None, ...]
    return inputs.repeat((num_candidates, 1))

def modify_generated_sequence(generated_sequences, generated_log_probs):
    
    final_generated_sequences = []
    final_generated_log_probs = []
    
    for i in range(generated_sequences.shape[0]):
        
        batch_tokens = []
        batch_log_probs = []
        
        for j in range(len(generated_sequences[i])):
            if generated_sequences[i][j] != 628 and generated_sequences[i][j] != -1:
                batch_tokens.append(generated_sequences[i][j])
                batch_log_probs.append(generated_log_probs[i][j])
            elif generated_sequences[i][j] == 628:
                batch_tokens.append(generated_sequences[i][j])
                batch_log_probs.append(generated_log_probs[i][j])
                batch_tokens.append(198)
                break
            else:
                break
        final_generated_sequences.append(torch.tensor(batch_tokens).unsqueeze(0))
        ### BE CAREFUL WHEN USING THIS, SINCE IT DOESN NOT AVERAGES THE LOG PROBS INSTEAD IT JUST TAKES THE SUM.
        final_generated_log_probs.append(torch.tensor(batch_log_probs).sum().item())
    
    return final_generated_sequences, final_generated_log_probs

def top_p_candidates(logits, prob=0.92, filter_value=-float('Inf')):
    
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cum_sum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cum_sum > prob
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter_(1, index=sorted_indices, src=sorted_indices_to_remove.clone())
    logits[indices_to_remove] = filter_value
    
    return logits

def generate_n_candidates(model,
                          inputs,
                          top_p,
                          temperature,
                          num_candidates,
                          max_gen_length,
                          past,
                          device,
                          eos_token_id=628,
                          pad_token_id=198):

    curr_len = 2

    inputs = expand_inputs_for_N_candidates(inputs, num_candidates)
    inputs_ = inputs
    
    generated_sequences = torch.ones((inputs.shape[0], max_gen_length), dtype=torch.long) * -1
    generated_sequences[:, 0:2] = inputs.cpu()
    
    generated_token_log_prob = torch.zeros((inputs.shape[0], max_gen_length), dtype=torch.float)
    
    unfinished_sequences = inputs.new(inputs.shape[0]).fill_(1) #.cpu()
    
    i = 0
    
    while True:
        if past:
            if past[0][0].shape[-2] > 1024:
                if not torch.all(generated_sequences==-1):
                    final_generated_sequence, final_generated_log_probs = modify_generated_sequence(generated_sequences, generated_token_log_prob)
                    return final_generated_sequence, final_generated_log_probs, past_to_return
                else:
                    return None, None
        
        outputs = model(inputs, past, return_dict=False)
        logits, past = outputs[0], outputs[1]
        
        next_token_logits = logits[:, -1, :].contiguous() / temperature
        
        if top_p and top_p > 0.0:
            # This returns score after performing softmax function.
            next_token_logits = top_p_candidates(next_token_logits, top_p)
            next_token_log_probs = F.log_softmax(next_token_logits, -1)
            probs = F.softmax(next_token_logits, dim=-1)
            
            next_tokens = torch.multinomial(probs, num_samples=1)
            next_token_log_probs = next_token_log_probs.gather(-1, next_tokens)
            next_tokens = next_tokens.squeeze(1)
            
            if eos_token_id is not None:
                assert pad_token_id is not None # "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # NOTE: SAVE LOG PROBS AS WELL
            generated_sequences[:, curr_len] = next_tokens.cpu()
            inputs = next_tokens.unsqueeze(1).to(device)
            #inputs_ = torch.cat((inputs_, next_tokens[:, None]), dim=-1)
            
            curr_len = curr_len + 1
            
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
            if unfinished_sequences.max() == 0:
                break
            if curr_len >= max_gen_length:
                break
    
    final_generated_sequences, final_generated_log_probs =  modify_generated_sequence(generated_sequences, generated_token_log_prob)
    
    return final_generated_sequences, final_generated_log_probs

def compute_log_probs(target_token_ids, logits, mask, average_sent_loss=False):
    logits = logits[:, :-1, :].contiguous() # (batch, sequence_length, vocab_size)
    
    target_token_ids = target_token_ids[:, 1:].contiguous() # (batch, sequence_length)
    

    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, -1, target_token_ids.unsqueeze(-1)).squeeze(-1)
    mask = mask[:, 1:].contiguous()
    
    if average_sent_loss:
        log_probs = (log_probs * mask).sum(-1) / mask.sum(-1)
    else:
        log_probs = (log_probs * mask).sum(-1)
    return {'log_probs': log_probs}

def ppo_step(model_A,
             model_B,
             buffer_memory,
             device,
             ppo_epsilon,
             num_candidates,
             criterion,
             optimizer,
             dial_inputs,
             role_ids,
             scheduler=None,
             train_single_model=False,
             model_to_train=None,
             average_sent_loss=False,
             use_recent_past=False):

    optimizer.zero_grad()
    
    log_dict = {}
    
    new_log_prob = []
    old_log_prob = []
    
    rewardlist = []
    
    ratios = []
    
    policy_loss = []
    advantages  = []

    if use_recent_past:
        print('USING RECENT PAST')
    else:
        print('NOT USING RECENT PAST')


    if use_recent_past:
        
        batches = buffer_memory.get_batch(shuffle=False)
        
        past = None
        
        i = 1
        
        for idx, batch in enumerate(batches):
            
            action = torch.tensor(batch['action'], device=device).unsqueeze(0)
            #pdb.set_trace()      
            if batch['human_response']:
                
                if idx == 0:
                    logits, past = model_A(action, past, return_dict=False)
                
                if idx > 0 and idx % (num_candidates + 1) == 0:
                    try:
                        past = out
                    except:
                        pass
                    
                    #history_indices = idx // (num_candidates + 1)
                    #history = dial_inputs[history_indices]
                    
                    history = dial_inputs[i]
                    
                    _, past = model_A(history.to(device), past_key_values=past, return_dict=False)
                    logits, out = model_A(action, past_key_values=past, return_dict=False)
                    
                    i += 2
            else:
                history_indices = idx // (num_candidates + 1)  # {A:(1,2,3,4,5),B, C:(7,8,9,10,11), D, E: (13,14,15,16,17)}
                
                if history_indices == 0:
                    logits, _ = model_A(action, past_key_values=None, return_dict=False)
                else:
                    logits, _ = model_A(action, past_key_values=past, return_dict=False)
            
            new_log_probs = compute_log_probs(target_token_ids=action,
                                              logits=logits,
                                              mask=torch.ones_like(action).to(device),
                                              average_sent_loss=average_sent_loss)['log_probs']

            old_log_probs = torch.tensor(batch['log_prob'], device=device).unsqueeze(0)
            old_log_prob.append(old_log_probs)

            rewards = torch.tensor(batch['reward'], device=device).unsqueeze(0)
            rewardlist.append(batch['reward'])
            advantages.append(rewards)

            new_log_prob.append(new_log_probs)

        if new_log_prob:
            new_log_prob = torch.cat(new_log_prob, dim=-1)
            old_log_prob = torch.cat(old_log_prob, dim=-1)
        
            advantages = torch.cat(advantages, dim=-1)
        
            ratio = (new_log_prob - old_log_prob).exp()
        
            policyloss1 = - advantages * ratio
            policyloss2 = - advantages * ratio.clamp(1 - ppo_epsilon, 1 + ppo_epsilon)
        
            policyloss = torch.min(policyloss1, policyloss2).mean()
        
            policyloss.backward()

            with torch.no_grad():
                log_dict['policy_loss'] = policyloss.item()
                print('Policy Loss: ', log_dict['policy_loss'])
                
                # (r-1) - logr, where r = p(x)/q(x); p(x) = new distribution and q(x) is old distribution
                log_dict['approx_kl'] = torch.mean(((new_log_prob - old_log_prob).exp() - 1)\
                                                - (new_log_prob - old_log_prob)).item()
                #log_dict['approx_kl'] = 0.5 * np.mean(np.power((np.array(new_log_prob) - np.array(old_log_prob)), 2))
                print('approx KL div: ', log_dict['approx_kl'])
                
                log_dict['clip_frac'] = torch.mean((torch.abs(ratio-1) > ppo_epsilon).float()).item()
                print('clip frac: ', log_dict['clip_frac'])
                
                log_dict['reward'] = np.mean(rewardlist)
                print('rewards: ', log_dict['reward'])
        else:
            log_dict['policy_loss'] = 0
            print('Policy Loss: ', log_dict['policy_loss'])
                
            # (r-1) - logr, where r = p(x)/q(x); p(x) = new distribution and q(x) is old distribution
            log_dict['approx_kl'] = 0
            
            #log_dict['approx_kl'] = 0.5 * np.mean(np.power((np.array(new_log_prob) - np.array(old_log_prob)), 2))
            print('approx KL div: ', log_dict['approx_kl']) 

            log_dict['clip_frac'] = 0
            print('clip frac: ', log_dict['clip_frac'])
                
            log_dict['reward'] = 0
            print('rewards: ', log_dict['reward'])
        

    if not train_single_model:
        nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)
        nn.utils.clip_grad_norm_(model_B.parameters(), 1.0)
    else:
        if model_to_train =='counsellor':
            nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)

    optimizer.step()
    #scheduler.step()

    return log_dict


@torch.no_grad()
def collect_samples(batch,
                    model_A,
                    model_B,
                    top_p,
                    eos_token_id,
                    pad_token_id,
                    max_gen_length,
                    num_candidates,
                    human_reward, 
                    use_jacc,
                    use_uep_cons,
                    use_uee_cons,
                    use_dial_flow_cons,
                    buffer_memory,
                    device,
                    tokenizer,
                    criterion,
                    temperature,
                    use_recent_past,
                    average_sent_loss,
                    nlp,
                    alpha,
                    gamma,
                    delta,
                    ea_empathy_classifier,
                    pa_politeness_classifier,
                    empathy_classifier,
                    beta1,
                    beta2,
                    beta3,
                    beta4,
                    beta5,
                    beta6,
                    beta7,
                    beta8,
                    politeness_tokenizer,
                    train_single_model=False,
                    model_to_train=None,
                    recompute_log_prob=True,
                    politeness_classifier=None,
                    fp16=False):

    scores_dict = {}


    scores_dict['jacc_scores'] = []
    scores_dict['uep_consistency_scores'] = []
    scores_dict['uee_consistency_scores'] = []
    scores_dict['dial_flow_consistency_scores'] = []
    scores_dict['pa_politeness_scores'] = []
    scores_dict['ea_empathy_scores'] = []
    scores_dict['politeness_scores'] = []
    scores_dict['politeness_actual_prob'] = []
    scores_dict['politeness_other_prob'] = []
    scores_dict['empathy_actual_prob'] = []
    scores_dict['empathy_other_prob'] = []
    scores_dict['empathy_scores'] = []

    if not empathy_classifier and not politeness_classifier:# no labels
        role_ids, dialog_tokens = batch 
    elif empathy_classifier and politeness_classifier: # All labels
        role_ids, dialog_tokens, empathy_label, politeness_label = batch
    elif empathy_classifier and not politeness_classifier:
        role_ids, dialog_tokens, empathy_label = batch # empathy
    elif not empathy_classifier and politeness_classifier:
        role_ids, dialog_tokens, politeness_label = batch # politeness
    
    # print(dialog_tokens)

    dial_inputs = [torch.LongTensor(item).unsqueeze(0) for item in dialog_tokens]

    past = None
    past_ = None
    
    context = None
    cntxt = None

    counsellor_generated_list, client_generated_list = [], []
    length = np.zeros(num_candidates)
    length = length.tolist()

    for num_turn, dialog_turn_inputs in enumerate(dial_inputs):
        
        assert not np.any(np.isnan(dialog_turn_inputs).cpu().numpy()), 
        
        dialog_turn_inputs = dialog_turn_inputs.to(device)

        if empathy_classifier and politeness_classifier:
            # All Labels
            actual_empathy_label = empathy_label[num_turn]
            actual_politeness_label = politeness_label[num_turn]
        elif empathy_classifier and not politeness_classifier:
            # empathy
            actual_politeness_label = None
            actual_empathy_label = empathy_label[num_turn]
        elif not empathy_classifier and politeness_classifier:
            # politeness
            actual_empathy_label = None
            actual_politeness_label = politeness_label[num_turn]
        else:
            actual_empathy_label = None
            actual_politeness_label = None


        if model_to_train == 'counsellor':

            if role_ids[num_turn] == 0:
                
                '''if use_recent_past:
                    if cntxt is not None:
                        past = prepare_inputs(cntxt, model_A)
                    else:
                        past = None'''
                
                #dial_turn_str = convert_sentences_to_strings([dialog_turn_inputs], tokenizer)[0]

                outputs = model_A(dialog_turn_inputs, past, return_dict=False)
                logits = outputs[0]

                mask = torch.ones_like(dialog_turn_inputs).to(device)
                
                log_probs = compute_log_probs(target_token_ids=dialog_turn_inputs,
                                              logits=logits,
                                              mask=mask,
                                              average_sent_loss=average_sent_loss)
                
                buffer_memory.update_buffer(state=dialog_turn_inputs.tolist()[0],
                                            context=context,
                                            action=dialog_turn_inputs.tolist()[0],
                                            action_log_probs=log_probs['log_probs'].item(),
                                            reward=human_reward,
                                            counsellor=True,
                                            human_response=True)
                if not use_recent_past:
                    '''In this case, first we generate sentence using the entire past. And then we update the past with
                    the current utterance.'''
                    generated_sequence, generated_log_probs  = generate_n_candidates(model_A,
                                                                                     torch.tensor(tokenizer.encode("A:")).unsqueeze(0).to(device),
                                                                                     top_p,
                                                                                     eos_token_id=eos_token_id,
                                                                                     pad_token_id=pad_token_id,
                                                                                     num_candidates=num_candidates,
                                                                                     max_gen_length=max_gen_length,
                                                                                     temperature=temperature,
                                                                                     past=past_,
                                                                                     device=device)
                    output = model_A(expand_inputs_for_N_candidates(dialog_turn_inputs,num_candidates),
                                     past_,
                                     return_dict=False)

                    past_ = output[1]
                
                else:
                    '''Here first we calculate the past based on the context sentence and then we generate candidates.'''
                    '''if cntxt is not None:
                        past_ = prepare_inputs(expand_inputs_for_N_candidates(cntxt, num_candidates), model_A)
                    else:
                        past_ = None'''

                    generated_sequence, generated_log_probs = generate_n_candidates(model_A,
                                                                                    torch.tensor(tokenizer.encode("A:")).unsqueeze(0).to(device), top_p,
                                                                                    eos_token_id=eos_token_id,
                                                                                    pad_token_id=pad_token_id,
                                                                                    num_candidates=num_candidates,
                                                                                    max_gen_length=max_gen_length,
                                                                                    temperature=temperature,
                                                                                    past=past_,
                                                                                    device=device)

                #gen_sent = convert_sentences_to_strings(generated_sequence, tokenizer)
                counsellor_generated_list = append(counsellor_generated_list, dialog_turn_inputs, tokenizer)

                current_sentence = tokenizer.decode(dialog_turn_inputs.tolist()[0][2:])
                if num_turn+1<len(dial_inputs):
                    next_sentence = tokenizer.decode(dial_inputs[num_turn+1].tolist()[0][2:])
                else:
                    next_sentence = ''


                # calculation of dialogue length:
                length = dialogue_length(generated_sequence, tokenizer, length, num_turn, dial_inputs)

                reward, scores = calculate_rewards(current_sentence=current_sentence,
                                                   next_sentence = next_sentence,
                                                   num_turn=num_turn,
                                                   dial_inputs=dial_inputs,
                                                   generated_sentences= generated_sequence,
                                                   length=length,
                                                   actual_empathy_label=actual_empathy_label,
                                                   actual_politeness_label=actual_politeness_label,
                                                   source_list=counsellor_generated_list,
                                                   tokenizer=tokenizer,
                                                   criterion=criterion,
                                                   politeness_tokenizer=empathy_tokenizer,
                                                   empathy_classifier=empathy_classifier,
                                                   politeness_classifier=politeness_classifier,
                                                   counsellor=True,
                                                   use_jacc=use_jacc,
                                                   use_uep_cons=use_uep_cons,
                                                   use_uee_cons=use_uee_cons,
                                                   use_dial_flow_cons=use_dial_flow_cons,
                                                   nlp=nlp,
                                                   device=device,
                                                   alpha=alpha,
                                                   gamma=gamma,
                                                   delta=delta,
                                                   beta1=beta1,
                                                   beta2=beta2,
                                                   beta3=beta3,
                                                   beta4=beta4,
                                                   beta5=beta5,
                                                   beta6=beta6,
                                                   beta7=beta7,
                                                   beta8=beta8,
                                                   pa_politenes_classifier=pa_politenes_classifier,
                                                   ea_empathy_classifier=ea_empathy_classifier,
                                                   model_A=model_A)

                #candidate_dict[dial_turn_str] = convert_sentences_to_strings(generated_sequence, tokenizer)



                scores_dict['jacc_scores'].extend(scores['jacc'])
                scores_dict['uep_consistency_scores'].extend(scores['uep_consistency'])
                scores_dict['uee_consistency_scores'].extend(scores['uee_consistency'])
                scores_dict['dial_flow_consistency_scores'].extend(scores['dial_flow_consistency'])
                scores_dict['pa_politeness_scores'].extend(scores['pa_politeness'])
                scores_dict['ea_empathy_scores'].extend(scores['ea_empathy'])
                
                scores_dict['empathy_actual_prob'].extend(scores['empathy_actual_prob'])
                scores_dict['empathy_other_prob'].extend(scores['empathy_other_prob'])
                scores_dict['empathy_scores'].extend(scores['empathy'])

                scores_dict['politeness_actual_prob'].extend(scores['politeness_actual_prob'])
                scores_dict['politeness_other_prob'].extend(scores['politeness_other_prob'])
                scores_dict['politeness_scores'].extend(scores['politeness'])

                if recompute_log_prob:

                    for i in range(len(generated_sequence)):
                        
                        # NOTE: STILL USING THE PAST FROM PREVIOUS UTTERANCE, SINCE WE DO NOT NEED PAST FROM
                        #       CONTAINING CURRENT UTTERANCE for GENERATED CANDIDATES
                        
                        output = model_A(generated_sequence[i].to(device), past_key_values=past, return_dict=False)
                        logits = output[0]
                        
                        log_probs = compute_log_probs(target_token_ids=generated_sequence[i].to(device),
                                                      logits=logits,
                                                      mask=torch.ones_like(generated_sequence[i]).to(device),
                                                      average_sent_loss=average_sent_loss)['log_probs'].item()
                        
                        buffer_memory.update_buffer(state=dialog_turn_inputs.tolist()[0],
                                                    context=context,
                                                    action= generated_sequence[i].tolist()[0],
                                                    action_log_probs=log_probs,
                                                    reward=reward[i],
                                                    counsellor=True,
                                                    human_response=False)
                else:
                    for i in range(len(generated_sequence)):
                        buffer_memory.update_buffer(state=dialog_turn_inputs.tolis()[0],
                                                    action=generated_sequence[i].tolist()[0],
                                                    action_log_probs=generated_log_probs[i],
                                                    counsellor=True,
                                                    human_response=False)
                past = outputs[1]
                outputs = model_A(expand_inputs_for_N_candidates(dialog_turn_inputs, num_candidates), past_, return_dict=False)
                past_ = outputs[1]
            else:
                #NOTE: Context will always be client's utterance since, because candidates are generated in response to this utterance.
                outputs = model_A(dialog_turn_inputs, past, return_dict=False)
                past = outputs[1]
                outputs = model_A(expand_inputs_for_N_candidates(dialog_turn_inputs, num_candidates), past_, return_dict=False)
                past_ = outputs[1]
        
        context = dialog_turn_inputs.tolist()[0]
        cntxt = dialog_turn_inputs

    return dial_inputs, role_ids, scores_dict, #candidate_dict

def get_past(batches, model, device):
    
    states = torch.cat(batches, dim=-1).to(device)
    outputs = model(states, past_key_values=None, return_dict=False)
    
    return outputs[1]

def prepare_inputs_for_model(batches, model, num_candidates, device):
    
    states = get_history_utterances(batches, num_candidates)
    states = torch.cat(states, dim=1, device=device)
    outputs = model(states, past_key_values=None, return_dict=False)
    
    return outputs[1]

def get_history_utterances(batches, num_candidates):
    states = []
    for i in range(0, len(batches), num_candidates+1):
        states.append(i)
    return states

def get_recursive_past(dial_inputs, role_ids, model_A, model_B, device):
    '''
    Uses both models alternatively to calculate pasts.
    Used in case of training only the counsellor.
    '''
    past = None
    for num_turn, utter in enumerate(dial_inputs):
        if role_ids[num_turn] == 0:
            _, past = model_A(utter.to(device), past_key_values=past, return_dict=False)
        else:
            _, past = model_B(utter.to(device), past_key_values=past, return_dict=False)
    return past

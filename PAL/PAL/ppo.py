class PPOMemory():
    def __init__(self, momentum=0.90):
        self.initialize_memory()

    def initialize_memory(self):
        self.buffer = []

    def clear_memory(self):
        self.buffer = []

    def update_buffer(self, state, context, action, action_log_probs, reward, counsellor, human_response):
        sample = {}
        sample['state'] = state # dialog_turn_input: list
        sample['context'] = context
        sample['action'] = action  # action similar to dial_turn_inputs: list
        sample['log_prob'] = action_log_probs # single value
        sample['reward'] = reward  # single value
        sample['counsellor'] = counsellor
        sample['human_response'] = human_response
        self.buffer.append(sample)

    def get_batch(self, shuffle=False):
        rewards = []

        if shuffle:
            np.random.shuffle(self.buffer)
            return self.buffer
        else:
            return self.buffer

    def get_batch_rewards(self):
        return self.actual_rewards

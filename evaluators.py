import re

from fastchat.model import (
    get_conversation_template
)

from system_prompts import get_evaluator_system_prompt_for_eval, get_evaluator_system_prompt_for_relevant

from language_models import GPT, Mixtral

def load_evaluator(args):
    if "gpt" in args.evaluator_model:
        return GPTEvaluator(args)
    elif args.evaluator_model == "_mixtral_":
        return MIXeval(args)
    else:
        raise NotImplementedError

class EvaluatorBase:
    def __init__(self, args):
        self.max_n_tokens = args.evaluator_max_n_tokens
        self.temperature = args.evaluator_temperature
        self.evaluator_name = args.evaluator_model
        self.goal = args.goal
        self.target_str = args.target_str
        self.system_prompt = get_evaluator_system_prompt_for_eval(args.goal, args.target_str)
        self.system_prompt_relevant = get_evaluator_system_prompt_for_relevant(args.goal)

    def get_evaluator_prompt(self, attack_prompt, target_response):
        prompt = f"[PROMPT]:{attack_prompt}\n[ASSISTANT'S RESPONSE]: {target_response}" 
        
        return prompt

    def get_evaluator_prompt_relevant(self, attack_prompt):
        prompt = f"[PROMPT]:{attack_prompt}" 
        
        return prompt

    def eval_score(self, attack_prompt_list, target_response_list):
        raise NotImplementedError

    def process_output_eval_score(self, raw_output):
        # Captures numbers enclosed in double square brackets, i.e., strings of the form "[[<number>]]"
        pattern = r'\[\[(\d+)\]\]' 
        match = re.search(pattern, raw_output)
        output = int(match.group(1)) if match else None
        
        if output is None:
            output = 1 
            
        return output

    def process_output_relevant_score(self, raw_output):
        # Captures numbers enclosed in double square brackets, i.e., strings of the form "[[<number>]]"
        pattern = r'\[\[([1-5])\]\]' 
        match = re.search(pattern, raw_output)
        output = int(match.group(1)) if match else None

        if output is None:
            output = 1

        return output


class GPTEvaluator(EvaluatorBase):
    def __init__(self, args):
        super(GPTEvaluator, self).__init__(args)
        self.evaluator_model = GPT(model_name = self.evaluator_name)

    def create_conv(self, full_prompt, system_prompt=None):
        if system_prompt is None:
            system_prompt = self.system_prompt
        
        conv = get_conversation_template(self.evaluator_name)
        conv.set_system_message(system_prompt)
        conv.append_message(conv.roles[0], full_prompt)
        
        return conv.to_openai_api_messages()

    def eval_score(self, attack_prompt_list, target_response_list):
        convs_list = [
                    self.create_conv(self.get_evaluator_prompt(prompt, response)) 
                    for prompt, response in zip(attack_prompt_list, target_response_list)
                ]

        print(f'\tQuerying evaluator with {len(attack_prompt_list)} prompts (to evaluate eval scores)', flush=True)

        raw_outputs = self.evaluator_model.batched_generate(convs_list, 
                                                        max_n_tokens = self.max_n_tokens,
                                                        temperature = self.temperature)
        
        outputs = [self.process_output_eval_score(raw_output) for raw_output in raw_outputs]
        return outputs

    def relevant_score(self, attack_prompt_list):
         
        
        convs_list = [
                    self.create_conv(self.get_evaluator_prompt_relevant(prompt), system_prompt=self.system_prompt_relevant) 
                    for prompt in attack_prompt_list
                ]
        
        print(f'\tQuerying evaluator with {len(attack_prompt_list)} prompts (to evaluate relevant scores)', flush=True)

        raw_outputs = self.evaluator_model.batched_generate(convs_list, 
                                                        max_n_tokens = self.max_n_tokens,
                                                        temperature = self.temperature)
        outputs = [self.process_output_relevant_score(raw_output) for raw_output in raw_outputs]
        return outputs


class MIXeval(EvaluatorBase):
    def __init__(self, args):
        super(MIXeval, self).__init__(args)
        self.evaluator_model = Mixtral(model_name = self.evaluator_name)

    def create_conv(self, full_prompt, system_prompt=None):
        if system_prompt is None:
            system_prompt = self.system_prompt
        conv = get_conversation_template("mistral")
        conv.set_system_message(system_prompt)
        conv.append_message(conv.roles[0], full_prompt)
        return conv.get_prompt()

    def eval_score(self, attack_prompt_list, target_response_list):
        convs_list = [self.create_conv(self.get_evaluator_prompt(prompt, response)) for prompt, response in zip(attack_prompt_list, target_response_list)]
        raw_outputs = self.evaluator_model.batched_generate(convs_list, 
                                                        max_n_tokens = self.max_n_tokens,
                                                        temperature = self.temperature)
        outputs = [self.process_output_eval_score(raw_output) for raw_output in raw_outputs]
        return outputs

    def relevant_score(self, attack_prompt_list):
         
        
        convs_list = [
                    self.create_conv(self.get_evaluator_prompt_relevant(prompt), system_prompt=self.system_prompt_relevant) 
                    for prompt in attack_prompt_list
                ]
        
        print(f'\tQuerying evaluator with {len(attack_prompt_list)} prompts (to evaluate relevant scores)', flush=True)

        raw_outputs = self.evaluator_model.batched_generate(convs_list, 
                                                        max_n_tokens = self.max_n_tokens,
                                                        temperature = self.temperature)

        outputs = [self.process_output_relevant_score(raw_output) for raw_output in raw_outputs]
        return outputs
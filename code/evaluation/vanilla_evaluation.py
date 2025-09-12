import json
import argparse
from collections import defaultdict
import math
from ..utils import ask_model

class Pipeline:

    def __init__(self, args, agents=None):
        self.modelId = args.testing_model_name
        self.num_self_consistency = args.num_self_consistency
        self.max_retries = args.max_retries
        self.num_agents = args.num_agents
        self.max_rounds = args.max_rounds
        self.dataset = args.dataset_name
        self.track_answer_revisions = args.track_answer_revisions
        self.agents = agents
        self.original_question = ''

    def check_answer_semantic(self, question, model_answer, gold_answer, correct_answer=None):
        message = [
            {
                "role": "system",
                "content": """Can you verify if the model's generated answer is correct according to the gold standard answer for the question? If the gold standard answer or its semantically equivalent version is correctly mentioned in the model's generated answer, guess YES.
                            Provide your best guess and the probability that it is correct (0.0 to 1.0).
                            Given ONLY the guess (Yes or No) and probability, no other words or explanation.
                            For example:
                            Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>
                            Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever;
                            just the probability!>."""
            },
            {
                "role": "user",
                "content": "Question: Can you get addicted to heroin after trying it once?\nGold answer: No, it takes repeated uses over a short space of time to become physically addicted to heroin.\nGenerated answer: No, but it can lead to rapid dependence with repeated use."
            },
            {
                "role": "assistant",
                "content": "Guess: Yes\nProbability: 80%"
            },
        ]

        prompt = "Question: " + question + "\nGold answer: " + str(gold_answer) + "\nGenerated answer: " + str(
            model_answer)

        message.append({"role": "user", "content": prompt})

        print(message)

        attempts = 0
        while attempts < self.max_retries:
            response = ask_model(message, use_temp=0.15, modelId=self.modelId)
            try:
                guess = response.split(':')[1].split('\n')[0].strip()
            except (IndexError, json.JSONDecodeError) as e:
                print(f"Failed to generate an extracted answer on attempt {attempts + 1}: {e}")
            attempts += 1

        print(guess, response)

        return guess, response

    def check_unknown(self, answer):

        # if we have a new model, use this prompt, otherwise for claude3 and llama3, the uncertainty phrases are more accuate.

        message = [
            {
                "role": "system",
                "content": """Evaluate the model's response to determine if it indicates uncertainty. Use the following steps:
                1. Check if the response contains any phrases from a predefined list of "uncertainty phrases" or similar expressions. These phrases indicate that the model does not know the answer.
                2. If any uncertainty phrase or similar expressions is present, return "I don't know".
                3. If no uncertainty phrase is present, return the original response exactly as is, without any changes or added explanation.

                Uncertainty phrases include:
                [
                    "unknown", "don't know", "do not know", "I don't know", "cannot be determined with certainty",
                    "is not definitively known", "is uncertain", "does not mention", "there is no information",
                    "does not provide", "there is not enough information provided to determine", "cannot be determined",
                    "is not provided", "is not known", "The response does not provide any information",
                    "The response does not provide enough information", "There is no answer.", "is unclear",
                    "remains uncertain",
                    "Unfortunately, I don't have enough", "Unfortunately, I do not have enough", "Unfortunately, ",
                    "I'm afraid I ", "There is no definitive", "I apologize, but I do not feel comfortable",
                    "there is no definitive",
                    "I do not have enough factual information", "I don't have enough context", "I apologize, but I couldn't",
                    "I do not have enough information", "not explicitly stated", "uncertain", "there is not enough information",
                    "you do not have enough verified information",
                    "I don't have enough information", "I need more context", "I apologize, ",
                    "The question cannot be answered", "no factual information", "not enough factual information",
                    "does not have enough factual information", "does not actually have any factual information", "Unknown", 
                    "couldn't find any information", "There is no information available", "There is no information",
                    "There is not enough information", "I need more information to"
                ]"""
            },
            {
                "role": "user",
                "content": "The response does not provide enough information to determine the answer.",
            },
            {
                "role": "assistant",
                "content": "I don't know"
            },
            {
                "role": "user",
                "content": "It takes repeated uses over a short space of time to become physically addicted to heroin.",
            },
            {
                "role": "assistant",
                "content": "It takes repeated uses over a short space of time to become physically addicted to heroin."
            },
        ]

        print(1, answer)
        message.append({"role": "user", "content": answer})

        response = ask_model(message, use_temp=0.15, modelId=self.modelId)
        print(2, response)

        return response


    def check_answer_semantic_equivalence(self, question, json_data):

        message = [
            {
                "role": "system",
                "content": "Strictly evaluate the semantic equivalence of the keys in the provided JSON given a question. Your response should strictly adhere to the JSON format provided, without additional explanations. Combine keys to the SIMPLER one only when they are definitively equivalent. Each of the 5 answers [1,2,3,4,5] must be included in one, and only one, unique group."
            },
            {
                "role": "user",
                "content": """ Question: Who was the producer of Beaches?
                Json: {
                      "the producers of the film Beaches were Bonnie Bruckheimer and Amanda Gruber.": [1,4,5],
                      "the producers of the 1988 film Beaches were Garry Marshall and Bonnie Bruckheimer.": [2,3]
                }"""
            },
            {
                "role": "assistant",
                "content": """{
                      "the producers of the film Beaches were Bonnie Bruckheimer and Amanda Gruber.": [1,4,5],
                      "the producers of the 1988 film Beaches were Garry Marshall and Bonnie Bruckheimer.": [2,3]
                }"""
            },
            {
                "role": "user",
                "content": """Question: who was the director of avatar?
                Json: {
                "James Cameron was the director of the 2009 film Avatar.": [
                    3,
                    4,
                    5
                ],
                "James Cameron": [
                    1,
                    2
                ]
            }"""
            },
            {
                "role": "assistant",
                "content": """{
                                            "James Cameron": [1,2,3,4,5],
                                        }"""
            },
            {
                "role": "user",
                "content": """Question: Does Izzie Stevens die in Grey's Anatomy?
                Json: {
                                    "Izzie Stevens does not actually die in Grey's Anatomy. She leaves the show at the end of Season 6, but her character is not killed off.": [1,5],
                                    "Izzie Stevens does not actually die in Grey's Anatomy. She leaves the show at the end of Season 6, but her character is still alive.": [2,4],
                                    "Izzie Stevens does not actually die in Grey's Anatomy. She leaves the show at the end of Season 6, and her character is written off as having moved away.": [3]
                                }"""
            },
            {
                "role": "assistant",
                "content": """{
                                                            "Izzie Stevens does not actually die in Grey's Anatomy. She leaves the show at the end of Season 6, but her character is still alive.": [1,2,3,4,5]
                                                        }"""
            },
            {
                "role": "user",
                "content": """Question: In what event was Harold Davis a former record holder, but now is held by Usain Bolt?
                        Json: {
                                                           "There is no information that suggests Harold Davis held a world record in any event that was later broken by Usain Bolt.": [1,2,5],
                                                           "The 100 meters.": [3,4]
                                                       }"""
            },
            {
                "role": "assistant",
                "content": """{
                                                           "There is no information that suggests Harold Davis held a world record in any event that was later broken by Usain Bolt.": [1,2,5],
                                                           "The 100 meters.": [3,4]
                                                       }"""
            },
            {
                "role": "user",
                "content": """Question: When is Season 3 of Emily in Paris's release date?
                Json: {
                                   "Season 3 of Emily in Paris has been confirmed, and it is scheduled to be released on December 21, 2022, on Netflix.": [1],
                                   "Season 3 of Emily in Paris has been confirmed, but the release date has not been officially announced yet.": [2,3,4,5]
                               }"""
            },
            {
                "role": "assistant",
                "content": """{
                                   "Season 3 of Emily in Paris has been confirmed, and it is scheduled to be released on December 21, 2022, on Netflix.": [1],
                                   "Season 3 of Emily in Paris has been confirmed, but the release date has not been officially announced yet.": [2,3,4,5]
                               }"""
            },
        ]

        json_string = json.dumps(json_data, indent=4)  # 'indent' for pretty-printing
        # print(json_string)
        prompt = "Question: " + question + "\nJson: " + json_string
        message.append({"role": "user", "content": prompt})

        attempts = 0
        final_response = dict()
        while attempts < self.max_retries:
            if attempts == 1:
                message.append({"role": "assistant", "content": response})
                message.append({"role": "user",
                                "content": "You cannot generate single quotes in a json. Regenerate with double quotes."})
            response = ask_model(message, use_temp=0.15, modelId=self.modelId, max_token=768)
            try:
                if '{' in response and '}' in response:
                    final_response = json.loads('{' + response.split('{')[1].split('}')[0] + '}')
                    break
            except json.JSONDecodeError:
                print(f"Failed to decode JSON on attempt {attempts + 1}")
            attempts += 1

        if attempts >= 1:
            del message[-2]
            del message[-1]

        # check if all the agent's response are considered.
        visited_response = [0] * self.num_agents

        all_answer_same_tag = False
        for key, value in final_response.items():
            if len(value) == self.num_agents:
                all_answer_same_tag = True
            for v in value:
                if v <= self.num_agents:
                    visited_response[v - 1] = 1


        if all_answer_same_tag:
            print("Checking..")
            sentence = (
                f"Your response indicates that all keys have the same answer. Check if you are correct, keep your answer or regenerate your answer with no other explanations")
            message.append({"role": "assistant", "content": response})
            message.append({"role": "user", "content": sentence})
            final_response = dict()
            attempts = 0
            while attempts < self.max_retries:
                response = ask_model(message, use_temp=0.15, modelId=self.modelId, max_token=768)
                try:
                    if '{' in response and '}' in response:
                        final_response = json.loads('{' + response.split('{')[1].split('}')[0] + '}')
                        break
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON on attempt {attempts + 1}")


                attempts += 1

        indices_of_zeros = [index for index, value in enumerate(visited_response) if value == 0]
        if indices_of_zeros:
            indices_text = ', '.join(str(index + 1) for index in indices_of_zeros)
            sentence = f"However, you are not including the answer {indices_text} in your final response. You should include all 5 answers. Regenerate your answer with no other explanations"
            message.append({"role": "assistant", "content": response})
            message.append({"role": "user", "content": sentence})
            final_response = dict()
            attempts = 0
            while attempts < self.max_retries:
                response = ask_model(message, use_temp=0.15, modelId=self.modelId, max_token=768)
                try:
                    if '{' in response and '}' in response:
                        final_response = json.loads('{' + response.split('{')[1].split('}')[0] + '}')
                        break
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON on attempt {attempts + 1}")

                attempts += 1


        final_response_filterd = defaultdict(list)
        for key, value in final_response.items():
            for v in value:
                if v <= self.num_agents:
                    visited_response[v - 1] = 1
                    final_response_filterd[key].append(v)
        print('check_answer_semantic_equivalence: ', final_response_filterd)
        return final_response_filterd

    def check_answer_consistency(self, question, model_answer):

        message = [
            {
                "role": "system",
                "content": """You are given a question and 5 answers. Your task is to identify unique answers by combining those with different phrasings but similar meanings into a single group. Each of the 5 answers [1,2,3,4,5] must be included in one, and only one, unique group. Your response should strictly adhere to the JSON format provided, without additional explanations. Here’s the format to be used
                                {
                                    "the unique answer": <list the answer number. Each of the 5 answers [1,2,3,4,5] must be included in one, and only one, unique group.>
                                }
                        """
            },
            {
                "role": "user",
                "content": "Question: What is Cecil Aldin's occupation?\n1. Cecil Aldin was a British artist and illustrator, best known for his dog paintings and illustrations.\n2. Cecil Aldin was a British artist and illustrator, best known for his animal illustrations, particularly dogs.\n3. Cecil Aldin was a British artist and illustrator, best known for his drawings of dogs and other animals.\n4. Cecil Aldin was a British artist and illustrator, best known for his dog paintings and illustrations.\n5. Cecil Aldin was a British artist and illustrator, best known for his animal and sporting artwork."
            },
            {
                "role": "assistant",
                "content": """{
                                               "Cecil Aldin was a British artist and illustrator, best known for his animal illustrations.": [1,2,3,4,5]
                                           }"""
            },
            {
                "role": "user",
                "content": "Question: What is Dominick Bellizzi's occupation?\n1. Dominick Bellizzi is a professional wrestler.\n2. Dominick Bellizzi is an actor.\n3. Dominick Bellizzi is an actor.\n4. Dominick Bellizzi is an American professional wrestler.\n5. Dominick Bellizzi is an actor."
            },
            {
                "role": "assistant",
                "content": """{
                                   "Dominick Bellizzi is an actor.": [2,3,5],
                                   "Dominick Bellizzi is a professional wrestler.": [1,4]
                               }"""
            },
            {
                "role": "user",
                "content": "Question: What is Edgar Allan Poe's occupation?\n1. Edgar Allan Poe's occupation was a writer, editor, and literary critic.\n2. Edgar Allan Poe's occupation is a writer, editor, and literary critic.\n3. Edgar Allan Poe's occupation is a writer, editor, and literary critic.\n4. Edgar Allan Poe's occupation is a writer and editor.\n5. Edgar Allan Poe's occupation is a writer, editor, and literary critic."
            },
            {
                "role": "assistant",
                "content": """{
                                                "Edgar Allan Poe's occupation is a writer and editor.": [4],
                                                "Edgar Allan Poe's occupation is a writer, editor, and literary critic": [1,2,3,5]
                                            }"""
            },
        ]

        prompt = "Question: " + question
        for i in range(len(model_answer)):
            prompt += '\n' + str(i + 1) + '. ' + model_answer[i]

        message.append({"role": "user", "content": prompt})
        final_response = dict()
        attempts = 0

        while attempts < self.max_retries:
            response = ask_model(message, use_temp=0.15, modelId=self.modelId, max_token=768)
            try:
                if '{' in response and '}' in response:
                    final_response = json.loads('{' + response.split('{')[1].split('}')[0] + '}')
                    break
            except json.JSONDecodeError:
                print(f"Failed to decode JSON on attempt {attempts + 1}")


            attempts += 1


        print('Answer clustering: ', response)

        # check if all the agent's response are considered.
        visited_response = [0] * self.num_agents
        for key, value in final_response.items():
            for v in value:
                if v <= self.num_agents and isinstance(v, int):
                    visited_response[v - 1] = 1

        indices_of_zeros = [index for index, value in enumerate(visited_response) if value == 0]

        if indices_of_zeros:
            indices_text = ', '.join(str(index + 1) for index in indices_of_zeros)
            sentence = f"However, you are not including the answer {indices_text} in your final response. You should include all 5 answers. Regenerate your answer with no other explanations"
            message.append({"role": "assistant", "content": response})
            message.append({"role": "user", "content": sentence})
            final_response = dict()
            attempts = 0
            while attempts < self.max_retries:
                response = ask_model(message, use_temp=0.15, modelId=self.modelId, max_token=768)
                try:
                    if '{' in response and '}' in response:
                        final_response = json.loads('{' + response.split('{')[1].split('}')[0] + '}')
                        break
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON on attempt {attempts + 1}")
                    # final_response_temp = response

                attempts += 1
            print('Regenerate: ', response)

        final_response_filterd = defaultdict(list)
        for key, value in final_response.items():
            for v in value:
                if v <= self.num_agents and isinstance(v, int):
                    visited_response[v - 1] = 1
                    final_response_filterd[key].append(v)

        if len(final_response_filterd) != 0:
            final_response = self.check_answer_semantic_equivalence(question, final_response_filterd)
        else:
            final_response = defaultdict(list)
        return final_response

    def calculate_uncertainty_score(self, final_response_consistency):

        print('final_response_consistency: ', final_response_consistency)
        final_agent_answer = [-1] * self.num_agents
        answer_tag = 0
        answer_log = defaultdict(dict)
        for key, value in final_response_consistency.items():
            for v in value:
                final_agent_answer[v - 1] = answer_tag
            answer_log['answer_' + str(answer_tag)]['value'] = key
            answer_log['answer_' + str(answer_tag)]['agents'] = value
            answer_tag += 1

        print('Final_agent_answer: ', final_agent_answer)

        # get how many times each agent maintains its answers
        print(self.track_answer_revisions, self.max_rounds)
        if self.track_answer_revisions and self.max_rounds > 1:
            answer_maintain_list = []
            sum_answer_maintain_times = 0
            for i in range(self.num_agents):
                answer_maintain_times = len([index for index, elem in enumerate(
                    self.agents['Agent_' + str(i + 1)]['track_answer_revisions'][0:(self.max_rounds - 1)]) if
                                             elem == "Yes"])
                answer_maintain_list.append(answer_maintain_times + 1)
                sum_answer_maintain_times += answer_maintain_times
            print('Answer maintain times: ', answer_maintain_list)
            answer_maintain_list = [x / (sum_answer_maintain_times + self.num_agents) for x in answer_maintain_list]
            print('Weight: ', answer_maintain_list)
            # get probabilities of each answer and calculate entropy
            probability_list = []
            for answer in range(answer_tag):
                probability = sum([answer_maintain_list[i] for i, final_answer in enumerate(final_agent_answer) if
                                   final_answer == answer])
                probability_list.append(probability)
            probability_list = [x / sum(probability_list) for x in probability_list]
            entropy = 0
            try:
                for i, answer in enumerate(range(answer_tag)):
                    probability = probability_list[i]
                    answer_log['answer_' + str(answer)]['probability'] = probability
                    entropy += -probability * math.log2(probability)
            except ValueError as e:
                print(f"Math domain error occurred: {e}")
                entropy = 2

        else:
            # get probabilities of each answer and calculate entropy
            answer_maintain_list = [v / self.num_agents for v in [1] * self.num_agents]
            print('Weight: ', answer_maintain_list)
            probability_list = []
            for answer in range(answer_tag):
                probability = sum([answer_maintain_list[i] for i, final_answer in enumerate(final_agent_answer) if
                                   final_answer == answer])
                probability_list.append(probability)
            probability_list = [x / sum(probability_list) for x in probability_list]
            entropy = 0
            try:
                for i, answer in enumerate(range(answer_tag)):
                    probability = probability_list[i]
                    answer_log['answer_' + str(answer)]['probability'] = probability
                    entropy += -probability * math.log2(probability)
            except ValueError as e:
                print(f"Math domain error occurred: {e}")
                entropy = 2

        print('Entropy: ', entropy)
        return entropy, answer_log

    def run(self, question, final_answers):

        self.original_question = question
        uncertainty_score = self.calculate_uncertainty_score(final_answers)
        return uncertainty_score



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testing_model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct",
                        help="select the model name")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="select the model name")
    parser.add_argument("--dataset_name", type=str, default="truthfulQA", help="dataset name")
    parser.add_argument("--file_dic", type=str, default="/diverseagententropy",
                        help="data file dictionary")
    parser.add_argument("--save_file", type=str, default="vanilla_qa", help="save datafile name")
    parser.add_argument("--use_temp", type=int, default=0.7, help="LLM's temperature")
    parser.add_argument("--start", type=int, default=0, help="data start index")
    parser.add_argument("--end", type=int, default=10000, help="data end index")
    parser.add_argument("--num_self_consistency", type=int, default=5, help="num_self_consistency")
    parser.add_argument("--max_retries", type=int, default=5, help="max_retries")
    parser.add_argument("--threshold", type=float, default=0.97, help="threshold")
    parser.add_argument("--mode", type=str, default="origin", help="max_retries")
    parser.add_argument("--track_answer_revisions", type=bool, default=False, help="track_answer_revisions")
    parser.add_argument("--num_agents", type=int, default=5, help="num_agents")
    parser.add_argument("--max_rounds", type=int, default=5, help="max_rounds")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    pipe = Pipeline(args)

    with open(
             args.file_dic + "/result/baseline/" + args.dataset_name + "_" + args.save_file + "_"  + args.model_name.replace('/','-') + "_0.json") as f:
        df_all = json.load(f)

    out_objs = []

    for i, df in enumerate(df_all):

        if i >= args.end: break
        if i < args.start: continue
        print(i)
        print(df['question'])
        print(df['gold_answer'])

        df['final_answer'] = pipe.check_answer_consistency(df['question'], df["vanilla_answers"])


        if len(df['final_answer']) != 0:
            predict_answer = max(df['final_answer'], key=lambda k: len(df['final_answer'][k]))
        else:
            predict_answer = ''

        predict_answer = pipe.check_unknown(predict_answer)
        print(predict_answer)

        if predict_answer != "I don't know" and predict_answer != '':
            guess, response = pipe.check_answer_semantic(df['question'], predict_answer, df['gold_answer'])

        else:
            guess = 'Unknown'

        pipe.num_agents = 5
        uncertainty = pipe.run(df['question'], df['final_answer'])
        df['uncertainty_score'], answer_log = uncertainty

        obj = {
            "question": df['question'],
            "gold_answer": df['gold_answer'],
            "agent_final_answer": predict_answer,
            "agent_answer": df['final_answer'],
            "uncertainty_score": df['uncertainty_score'],
            "evaluation": guess
        }
        out_objs.append(obj)
        print('\n')

        json.dump(out_objs, open(
            args.file_dic + "/result/final_answer/baseline/" + args.dataset_name + "_" + args.save_file +  "_"  + args.model_name.replace('/','-') + "_" + str(
                args.start) + ".json",
            "w"), indent=4)


if __name__ == '__main__':
	main()
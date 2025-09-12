# import boto3
import json
import argparse
import random
import math
from collections import defaultdict
import time
from ..utils import ask_model


class Pipeline:

    def __init__(self, args):
        self.modelId = args.model_name
        self.num_agents = args.num_agents
        self.max_rounds = args.max_rounds
        self.max_retries = args.max_retries

        # Initialize the agent log
        self.agents = dict()
        for i in range(self.num_agents):
            self.agents['Agent_' + str(i+1)] = dict()


        # Initialize the final answer log
        self.answer_log = defaultdict(dict)
        self.final_response_consistency = dict()

        self.original_question = ''

        # Initialize the general log
        self.all_rounds_answers = []
        self.all_rounds_consistency = []
        self.chosen_interacted_agent_per_round = defaultdict(list)


    def check_answer_semantic(self, question, model_answer, gold_answer):
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

        prompt = "Question: " + question + "\nGold answer: " + gold_answer + "\nGenerated answer: " + model_answer

        message.append({"role": "user", "content": prompt})

        attempts = 0
        while attempts < self.max_retries:
            response = ask_model(message, use_temp=0.15, modelId=self.modelId)
            try:
                guess = response.split(':')[1].split('\n')[0].strip()
            except (IndexError, json.JSONDecodeError) as e:
                print(f"Failed to generate an extracted answer on attempt {attempts + 1}: {e}")
            attempts += 1

        return guess, response


    def get_interaction_agent(self, agent_id):

        if agent_id not in self.chosen_interacted_agent_per_round.keys():
            visited_agents = self.agents['Agent_' + str(agent_id)]['visited_agents']

            temp_possible_interaction_agents = []
            for key, value_list in self.final_response_consistency.items():
                if agent_id not in value_list:
                    temp_possible_interaction_agents += value_list

            possible_interaction_agents = set(temp_possible_interaction_agents) - set(visited_agents)
            if len(possible_interaction_agents) != 0:
                interaction_agent_id = random.choice(list(possible_interaction_agents))
            else:
                if len(temp_possible_interaction_agents) != 0 :
                    interaction_agent_id = random.choice(list(temp_possible_interaction_agents))
                else:
                    interaction_agent_id = random.choice([i+1 for i in range(self.num_agents) if (i + 1) != agent_id])
            if interaction_agent_id not in self.chosen_interacted_agent_per_round.keys():
                self.chosen_interacted_agent_per_round[interaction_agent_id] = []
            self.chosen_interacted_agent_per_round[interaction_agent_id].append(agent_id)
        else:
            interaction_agent_id = random.choice(self.chosen_interacted_agent_per_round[agent_id])

        return interaction_agent_id



    def extract_atomic_fact_answer(self, response, question):

        # other way based solely on your response method
        message = [
            {
                "role": "system",
                "content": """You will extract the complete answer to the given question using ONLY the information provided in the "Response" section. You will identify the answer directly without using any additional knowledge or explanation. If the response includes a negation to the question, uses those as the answer. """
            },
            {
                "role": "user",
                "content": """Response: The prevalence of the most spoken language in the world, which is Mandarin Chinese, has a significant influence on global media and entertainment in several ways:

                1. **Content creation**: Many Chinese production companies and studios create content specifically for the massive Chinese-speaking audience, which often gets distributed globally. This leads to a increase in Chinese-language content in international markets.
                2. **Dubbing and subtitles**: To cater to the large Chinese-speaking population, many international films and TV shows are dubbed or subtitled in Mandarin, making them more accessible to Chinese audiences.
                3. **Global market appeal**: The massive Chinese market has become a crucial factor in the success of global films, TV shows, and music. Creators often tailor their content to appeal to Chinese audiences, incorporating Chinese themes, actors, or storylines.
                4. **Influence on global trends**: Chinese social media platforms, such as WeChat and Weibo, have become essential channels for promoting global entertainment content. Trends and memes that originate in China can quickly spread globally.
                5. **Cultural exchange**: The prevalence of Mandarin Chinese has facilitated cultural exchange between China and other countries. International artists and creators are increasingly collaborating with Chinese counterparts, leading to a fusion of cultural influences in media and entertainment.

                Overall, the dominance of Mandarin Chinese has reshaped the global media and entertainment landscape, with creators and distributors adapting to cater to the vast and influential Chinese-speaking audience.
                Based solely on the response, What is the most spoken language in the world?"""
            },
            {
                "role": "assistant",
                "content": "The most spoken language in the world is Mandarin Chinese."
            },
            {
                "role": "user",
                "content": """Response: I apologize for another mistake. I don't have any information about the most spoken language in the world, and I should not have assumed it is Chinese. Thank you for correcting me again!
            Based solely on the response, What is the most spoken language in the world?""",
            },
            {
                "role": "assistant",
                "content": "I don't know the answer."
            },
            {
                "role": "user",
                "content": """Response: There is no evidence of an animal landing on the moon. The first humans to walk on the moon were Neil Armstrong and Edwin "Buzz" Aldrin during the Apollo 11 mission in 1969. However, animals have been launched into space as part of space research and exploration.
                        One example is Laika, a Soviet space dog who became the first living creature to orbit the Earth in 1957. Laika's mission, Sputnik 2, provided valuable data on the safety and feasibility of space travel for living organisms. Although Laika did not survive the flight, the mission helped scientists understand the effects of space travel on a living organism's physiology and paved the way for future human spaceflight.
                        Other animals, such as fruit flies, worms, and mice, have also been sent to space as part of scientific experiments to study the effects of microgravity and space radiation on living organisms. These experiments have contributed significantly to our understanding of space travel's effects on living organisms and have helped inform the development of safety measures for human space travelers.
                        Based solely on the response, What is the name of the first animal to land on the moon?"""
            },
            {
                "role": "assistant",
                "content": "There is no evidence of an animal landing on the moon."
            },
            {
                "role": "user",
                "content": """Response: If you traveled to most countries in the world, you would be most likely to encounter English. English is widely spoken and has become a lingua franca, meaning it's often used as a common language for international communication. It's an official language in over 60 countries and is widely spoken in many more.
                       Based solely on the response, What is the most spoken language in the world?"""
            },
            {
                "role": "assistant",
                "content": "The answer cannot be extracted."
            },

        ]

        prompt = 'Response: ' + response + '\nBased solely on the response, ' + question
        message.append({"role": "user", "content": prompt})
        response = ask_model(message, use_temp=0.15, modelId=self.modelId)
        atomic_fact_answer = response.replace('According to the response, ', '').replace('Based solely on the response, ','').replace('Based solely on the response provided, ','').strip() # .split('Answer: ')[1]

        # double check if the answer cannot be extracted
        if 'cannot be extracted' in atomic_fact_answer or "does not mention" in atomic_fact_answer:
            sentence = f"Regenerate your answer. If you still think the answer cannot be extracted, please response with \"The answer cannot be extracted.\""
            message.append({"role": "assistant", "content": response})
            message.append({"role": "user", "content": sentence})
            response = ask_model(message, use_temp=0.15, modelId=self.modelId)
            atomic_fact_answer = response.replace('According to the response, ', '').replace(
                'Based solely on the response, ', '').replace('Based solely on the response provided, ',
                                                              '').strip()  # .split('Answer: ')[1]

        if "The response does not provide any information" in atomic_fact_answer or "The response does not provide enough information" in atomic_fact_answer or "cannot be determined" in atomic_fact_answer or "there is not enough information provided to determine" in atomic_fact_answer or "unknown" in atomic_fact_answer or "don't know" in atomic_fact_answer or "do not know" in atomic_fact_answer or "cannot be determined with certainty" in atomic_fact_answer or "is not definitively known" in atomic_fact_answer or "is uncertain" in atomic_fact_answer  or "there is no information" in atomic_fact_answer or "does not provide" in atomic_fact_answer or "there is not enough information provided to determine" in atomic_fact_answer or "cannot be determined" in atomic_fact_answer or "is not provided" in atomic_fact_answer or "is not known" in atomic_fact_answer:
            atomic_fact_answer = "I don't know"

        return atomic_fact_answer, response


    def first_round(self, agent_id, question):

        message = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people answer questions. Ensure your responses are concise and strictly relevant to the queries presented, avoiding any unrelated content to the question."
            }
        ]

        message.append({"role": "user", "content": question})


        attempts = 0
        answer_flag = 0

        while attempts < self.max_retries:
            response = ask_model(message, use_temp=0.15,modelId=self.modelId)
            try:
                atomic_fact_answer, extraction_response = self.extract_atomic_fact_answer(response,
                                                                                          self.original_question)
                if 'cannot be extracted' not in atomic_fact_answer:
                    answer_flag = 1
                    break
            except json.JSONDecodeError:
                print(f"Failed to generate an extracted answer on attempt {attempts + 1}")
            attempts += 1

        message.append({"role": "assistant", "content": response})


        # deal with situations the answer cannot be extracted
        if answer_flag == 0:
            message_1 = [
                {
                    "role": "system",
                    "content": "You are an AI assistant that helps people answer questions. Ensure your responses are concise and strictly relevant to the queries presented, avoiding any unrelated content to the question."
                }
            ]
            prompt = response + '\nBased on the information from your previous response, ' + question
            message_1.append({"role": "user", "content": prompt})
            response_1 = ask_model(message_1, use_temp=0.15, modelId=self.modelId)
            atomic_fact_answer, extraction_response = self.extract_atomic_fact_answer(response_1,
                                                                                      self.original_question)

        self.agents['Agent_' + str(agent_id)]['question'] = question
        message[0]['content'] = "You are an AI assistant that helps people answer questions. Ensure your responses are concise and strictly relevant to the queries presented, avoiding any unrelated content to the question. Do not change your answer unless you think you are absolutely wrong in your response."
        self.agents['Agent_' + str(agent_id)]['message'] = message
        self.agents['Agent_' + str(agent_id)]['answer'] = [atomic_fact_answer]
        self.agents['Agent_' + str(agent_id)]['response'] = [response]
        self.agents['Agent_' + str(agent_id)]['visited_agents'] = []
        self.agents['Agent_' + str(agent_id)]['track_answer_revisions'] = []
        print('Agent main question: ',question)
        print('Response: ',atomic_fact_answer)


        return atomic_fact_answer

    def interaction_round(self, round, agent_id, interaction_agent_id):

        message = self.agents['Agent_' + str(agent_id)]['message']

        selection_agent_question = self.agents['Agent_' + str(interaction_agent_id)]['question']
        selection_agent_answer = self.agents['Agent_' + str(interaction_agent_id)]['answer'][round-1]

        if len(selection_agent_answer) == 0:
            selection_agent_answer = "I don't know"
        if selection_agent_answer[-1] != '.':
            selection_agent_answer += '.'
        if "I don't know" in selection_agent_answer:
            current_round_message = 'When I asked you in another api call that ' + selection_agent_question + ' You mentioned that you don\'t know the answer to '+ self.original_question[0].lower() + self.original_question[1:-1] +'. Which is your actual answer to ' + self.original_question[0].lower() + self.original_question[1:]
        else:
            current_round_message = 'When I asked you in another api call that ' + selection_agent_question + ' You mentioned that ' + selection_agent_answer[0].lower() + selection_agent_answer[1:] + ' Which is your actual answer to ' + \
                                    self.original_question[0].lower() + self.original_question[1:]

        print('Previous answer: ', message[-1]['content'])
        print('Interaction Prompt: ', current_round_message)
        message.append({"role": "user", "content": current_round_message})

        attempts = 0
        while attempts < self.max_retries:
            response = ask_model(message, use_temp=0.15, modelId=self.modelId)
            try:
                atomic_fact_answer, extraction_response = self.extract_atomic_fact_answer(response,
                                                                                          self.original_question)
                if 'cannot be extracted' not in atomic_fact_answer:
                    break
            except json.JSONDecodeError:
                print(f"Failed to generate an extracted answer on attempt {attempts + 1}")
            attempts += 1

        message.append({"role": "assistant", "content": response})

        print('Response: ', atomic_fact_answer)

        self.agents['Agent_' + str(agent_id)]['message'] = message
        self.agents['Agent_' + str(agent_id)]['answer'].append(atomic_fact_answer)
        self.agents['Agent_' + str(agent_id)]['response'].append(response)
        self.agents['Agent_' + str(agent_id)]['visited_agents'].append(interaction_agent_id)

        # check if the answer is change or not.
        previous_answer = self.agents['Agent_' + str(agent_id)]['answer'][round-1]
        guess, response = self.check_answer_semantic(self.original_question, atomic_fact_answer, previous_answer)
        self.agents['Agent_' + str(agent_id)]['track_answer_revisions'].append(guess)

        return atomic_fact_answer



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
                "content":"""Question: who was the director of avatar?
                Json: {
                "James Cameron was the director of the 2009 film Avatar.": [
                    3,
                    4,
                    5
                ],
                "James Cameron": [
                    1,
                    2
                ]}"""
            },
            {
                "role": "assistant",
                "content": """{
                                            "James Cameron": [1,2,3,4,5],
                                        }"""
            },
            {
                "role": "user",
                "content":"""Question: Does Izzie Stevens die in Grey's Anatomy?
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
                "content":"""Question: When is Season 3 of Emily in Paris's release date?
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

        json_string = json.dumps(json_data, indent=4)

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


        visited_response = [0] * self.num_agents
        print(self.num_agents)
        print(final_response)

        # double-check if all the answers are actually same.
        all_answer_same_tag = False
        for key, value in final_response.items():
            if len(value) == self.num_agents:
                all_answer_same_tag =True
            for v in value:
                if v <= self.num_agents:
                    visited_response[v - 1] = 1

        if all_answer_same_tag:
            print("Checking...")
            sentence = (f"Your response indicates that all keys have the same answer. Check if you are correct, keep your answer or regenerate your answer with no other explanations")
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

        # check if all the agent's response are considered.
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
                "content":"Question: What is Edgar Allan Poe's occupation?\n1. Edgar Allan Poe's occupation was a writer, editor, and literary critic.\n2. Edgar Allan Poe's occupation is a writer, editor, and literary critic.\n3. Edgar Allan Poe's occupation is a writer, editor, and literary critic.\n4. Edgar Allan Poe's occupation is a writer and editor.\n5. Edgar Allan Poe's occupation is a writer, editor, and literary critic."
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
            prompt += '\n' + str(i+1) + '. ' + model_answer[i]

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
            indices_text = ', '.join(str(index+1) for index in indices_of_zeros)
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
            print('Regenerate: ', response)


        # double-check for answer semantic equivalence
        final_response_filterd = defaultdict(list)
        for key, value in final_response.items():
            for v in value:
                if v <= self.num_agents and isinstance(v, int):
                    visited_response[v - 1] = 1
                    final_response_filterd[key].append(v)

        if len(final_response_filterd) != 0:
            final_response = self.check_answer_semantic_equivalence(question, final_response_filterd)
        return final_response


    def check_interaction_necessity(self,round_number):

        # # All answers are the same, stop interaction
        self.final_response_consistency = self.check_answer_consistency(self.original_question, self.all_rounds_answers[-1])
        self.all_rounds_consistency.append(self.final_response_consistency)
        if len(self.final_response_consistency) == 1:
            return False

        # All agents don't update their answers for three rounds, stop interaction
        if round_number > 2:
            label_count = 0
            for i in range(self.num_agents):
                if set(self.agents['Agent_' + str(i+1)]['track_answer_revisions'][-2:]) == "Yes":
                    label_count += 1
            if label_count == self.num_agents:
                return False

        # For any other situation, continue
        return True


    def calculate_uncertainty_score(self):
        print('final_response_consistency: ', self.final_response_consistency)
        final_agent_answer = [-1] * self.num_agents
        answer_tag = 0
        for key,value in self.final_response_consistency.items():
            for v in value:
                final_agent_answer[v-1] = answer_tag

            self.answer_log['answer_' + str(answer_tag)]['value'] = key
            self.answer_log['answer_' + str(answer_tag)]['agents'] = value
            self.answer_log['general_agent_answer'] = final_agent_answer
            answer_tag += 1


        # get how many times each agent maintains its answers
        answer_maintain_list = []
        sum_answer_maintain_times = 0
        for i in range(self.num_agents):
           answer_maintain_times = len([index for index, elem in enumerate(self.agents['Agent_' + str(i+1)]['track_answer_revisions']) if elem == "Yes"])
           answer_maintain_list.append(answer_maintain_times+1)
           sum_answer_maintain_times += answer_maintain_times
        print('Answer maintain times: ', answer_maintain_list)
        answer_maintain_list = [x /  (sum_answer_maintain_times+self.num_agents) for x in answer_maintain_list]
        print('Weight: ', answer_maintain_list)
        print('Final_agent_answer: ', final_agent_answer)

        self.answer_log['weight'] = answer_maintain_list
        self.answer_log['final_agent_answer'] = final_agent_answer

        # get probabilities of each answer and calculate entropy
        probability_list = []
        for answer in range(answer_tag):
            probability = sum([answer_maintain_list[i] for i, final_answer in enumerate(final_agent_answer) if final_answer == answer])
            probability_list.append(probability)
        probability_list = [x / sum(probability_list) for x in probability_list]
        entropy = 0
        try:
            for i,answer in enumerate(range(answer_tag)):
                probability = probability_list[i]
                self.answer_log['answer_' + str(answer)]['probability'] = probability
                entropy += -probability * math.log2(probability)
        except ValueError as e:
            # assign a large entropy if the probability cannot be properly calculated
            print(f"Math domain error occurred: {e}")
            entropy = 2

        return entropy







    def run(self, question, questions, gold_answer):

        self.original_question = question

        obj = {
            "question": question,
            "gold_answer": gold_answer,
            "agent_questions": questions
        }

        # Simulating answer collection for each round
        for round_number in range(self.max_rounds):
            # Create a new list for this round's answers
            current_round_answers = []

            # Simulate getting an answer from each agent
            if round_number == 0:
                for agent_id in range(self.num_agents):
                    real_agent_id = agent_id + 1
                    # Simulate an answer (you might collect this from user input or another function)
                    print(f"Agent {real_agent_id} in Round {round_number}")
                    answer = self.first_round(real_agent_id, questions[agent_id])
                    current_round_answers.append(answer)

            elif round_number > 0:
                # All answers are the same, stop interaction
                self.chosen_interacted_agent_per_round = dict()
                for agent_id in range(self.num_agents):
                    # Simulate an answer (you might collect this from user input or another function)
                    agent_id = agent_id + 1
                    interaction_agent_id = self.get_interaction_agent(agent_id)
                    print(f"Agent {agent_id} Interact with Agent {interaction_agent_id} in Round {round_number}")
                    answer = self.interaction_round(round_number, agent_id, interaction_agent_id)
                    current_round_answers.append(answer)


            # Store the current round's answers in the main list
            self.all_rounds_answers.append(current_round_answers)
            # check_interaction_necessity and calculate_uncertainty_score
            if self.check_interaction_necessity(round_number):
                continue
            else:
                break

        uncertainty_score = self.calculate_uncertainty_score()

        obj['uncertainty_score'] = uncertainty_score
        obj['final_answer'] = self.final_response_consistency
        obj['all_rounds_answers'] = self.all_rounds_answers
        obj['all_rounds_consistency'] = self.all_rounds_consistency
        obj['agents'] = self.agents
        obj['answer_log'] = self.answer_log

        # Example: Print all data collected
        for round_index, round_data in enumerate(self.all_rounds_answers):
            print(f"Round {round_index+1}: {round_data}")


        return obj



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="select the model name")
    parser.add_argument("--dataset_name", type=str, default="truthfulQA", help="dataset name")
    parser.add_argument("--file_dic", type=str, default="/diverseagententropy", help="data file dictionary")
    parser.add_argument("--save_file", type=str, default="agent_interaction", help="save file name")
    parser.add_argument("--start", type=int, default=0, help="data start index")
    parser.add_argument("--end", type=int, default=10000, help="data end index")
    parser.add_argument("--num_agents", type=int, default=5, help="num_agents")
    parser.add_argument("--max_rounds", type=int, default=5, help="max_rounds")
    parser.add_argument("--max_retries", type=int, default=5, help="max_retries")
    parser.add_argument("--mode", type=str, default="origin", help="mode")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)

    out_objs = []

    with open(
                args.file_dic + "/result/question/" + args.dataset_name + "_question_selection_" + args.model_name.replace('/','-') + "_0.json") as f:
            df_all = json.load(f)


    for i, df in enumerate(df_all):

        if i >= args.end: break
        if i < args.start: continue

        print(i)
        question = df['question']
        questions = df['final_questions']


        if args.mode == 'origin':
        # enforce 5 agents
            if len(questions) > 5:
                questions_to_choose = questions[3:-1]
                random.shuffle(questions_to_choose)

                questions = questions[0:3] + [questions_to_choose[0]] + questions[-1:]


        elif args.mode == 'se':
            # semantically equivalent questions instead of agent questions
            if "semantic_question_generation" in df["original_questions"].keys():
                questions = df["original_questions"]["semantic_question_generation"]
            else:
                continue

        gold_answer = df['gold_answer']
        print(question)
        print(gold_answer)
        print(questions)
        print(df["final_question_category"])
        print("Number of agents: ", len(questions))

        args.num_agents = len(questions)
        pipe = Pipeline(args)
        obj = pipe.run(question, questions, gold_answer)
        out_objs.append(obj)

        print('\n\n')

        json.dump(out_objs, open(
            args.file_dic + "/result/agent_interaction/" + args.dataset_name + "_" + args.save_file +  "_"  + args.model_name.replace('/','-') +  "_"  + str(args.start) +  "_"  + args.mode + ".json",
            "w"), indent=4)


if __name__ == '__main__':
	main()
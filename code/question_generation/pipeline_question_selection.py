
import json
import argparse
from collections import defaultdict
import random
from sentence_transformers import SentenceTransformer, util
import time
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
from ..utils import ask_model


class Pipeline:

    def __init__(self, args):
        self.modelId = args.model_name
        self.num_agents = args.num_agents
        self.conceptualized_question = args.conceptualized_question

    def generate_semantically_equivalent_question(self, question):
        message = [
            {
                "role": "system",
                "content": """For the given question, provide 5 semantically equivalent questions. Do not answer the question.
                STRICTLY follow the structure that each generated question is a line."""
            },
            {
                "role": "user",
                "content": "What is the most spoken language in the world?"
            },
            {
                "role": "assistant",
                "content": """Which language has the highest number of speakers globally?\nWhat language is spoken by the most people worldwide?\nWhich language tops the list of the world's most widely spoken languages?\nWhat is the world's dominant language by number of speakers?\nGlobally, which language is spoken by the greatest number of people?"""
            },
        ]

        message.append({"role": "user", "content": question})
        response = ask_model(message, use_temp=0.7, modelId=self.modelId)
        if len(response.replace('\n\n', '\n').split('\n')) != 5:
            sentence = f"You should not answer the question but generate 5 semantically equivalent questions. Regenerate your answer with no other explanations"
            message.append({"role": "assistant", "content": response})
            message.append({"role": "user", "content": question})
            response = ask_model(message, use_temp=0.7, modelId=self.modelId)

        return response.replace('\n\n', '\n').split('\n'), response

    def extract_atomic_fact_answer(self, response, question):
        # other way based solely on your response method

        message = [
            {
                "role": "system",
                "content": """Can you extract the answer to the given question using ONLY the information from the response? Please identify the answer directly and do not use your parametric knowledge. If the response includes a negation to the question, uses those as the answer. If you cannot extract the answer, you must only respond with "The answer cannot be explicitly extracted"."""
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
                "content": "The answer cannot be explicitly extracted."
            },

        ]

        prompt = 'Response: ' + response + '\nBased solely on the response, ' + question

        message.append({"role": "user", "content": prompt})
        response = ask_model(message, use_temp=0.15, modelId=self.modelId)
        atomic_fact_answer = response.strip()  # .split('Answer: ')[1]
        return atomic_fact_answer, response


    def check_unknown(self,answer):


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

        message.append({"role": "user", "content": answer})
        response = ask_model(message, use_temp=0.15, modelId=self.modelId)

        return response




    def check_question_answerability(self, question, original_question):

        message = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people answer questions. Ensure your responses are concise and strictly relevant to the queries presented, avoiding any unrelated content to the question."
            }
        ]

        message.append({"role": "user", "content": question})
        response = ask_model(message, use_temp=0.7, modelId=self.modelId)
        message.append({"role": "assistant", "content": response})

        atomic_fact_answer, extraction_response = self.extract_atomic_fact_answer(response, original_question)

        predict_answer = self.check_unknown(atomic_fact_answer)
        if predict_answer != "I don't know" and predict_answer != '' and "cannot be explicitly extracted" not in predict_answer:
            return True
        else:
            return False


    def question_reorder(self, origin_question, questions):
        similarity_score_dict = dict()
        scores = []
        for q in questions:
            embedding_1 = model.encode(origin_question, convert_to_tensor=True)
            embedding_2 = model.encode(q, convert_to_tensor=True)
            score = util.pytorch_cos_sim(embedding_1, embedding_2).tolist()[0][0]
            scores.append(score)

        paired_list = list(zip(questions, scores))
        sorted_pairs = sorted(paired_list, key=lambda x: x[1], reverse=True)
        sorted_questions, sorted_scores = zip(*sorted_pairs)
        sorted_questions = list(sorted_questions)
        sorted_scores = list(sorted_scores)

        for i, q in enumerate(sorted_questions):
            similarity_score_dict[q] = sorted_scores[i]

        return sorted_questions, similarity_score_dict



    def run(self, original_question, questions_dict, gold_answer):

        print(original_question)
        print(gold_answer)
        obj = {
            "question": original_question,
            "original_questions": questions_dict,
            "gold_answer": gold_answer
        }
        final_question = []
        final_question_category = dict()
        final_question.append(original_question)
        final_question_category[original_question] = 'original question'

        semantically_equivalent_questions = questions_dict['semantic_question_generation']

        if "I apologize, but I cannot" not in semantically_equivalent_questions[0]:
            temp_semantically_equivalent_question = random.choice(semantically_equivalent_questions)
            final_question.append(temp_semantically_equivalent_question)
            final_question_category[temp_semantically_equivalent_question] = 'semantically equivalent question'


        for category, questions in questions_dict.items():
            if category != "semantic_question_generation":
                questions_to_check = questions[:]
                random.shuffle(questions_to_check)
                for question in questions_to_check:
                    if self.check_question_answerability(question, original_question):
                        final_question.append(question)
                        final_question_category[question] = category
                        break

        if len(final_question) < self.num_agents:
            remain_question_num = self.num_agents - len(final_question)
            count = 0
            possible_category = {v: k for k, v in final_question_category.items()}.keys()
            for category in possible_category:
                if category != 'semantically equivalent question' and category != 'original question':
                    questions = questions_dict[category]
                    questions_to_check = questions[:]
                    random.shuffle(questions_to_check)
                    for question in questions_to_check:
                        if question not in final_question:
                            if self.check_question_answerability(question, original_question):
                                final_question.append(question)
                                final_question_category[question] = category
                                count += 1
                        if count == remain_question_num:
                            break
                if count == remain_question_num:
                    break

        if len(final_question) < self.num_agents:
            remain_question_num = self.num_agents - len(final_question)
            count = 0
            if "I apologize, but I cannot" not in semantically_equivalent_questions[0]:
                questions_to_check = list(set(semantically_equivalent_questions) - set([temp_semantically_equivalent_question]))
                random.shuffle(questions_to_check)
                for question in questions_to_check:
                    final_question.append(question)
                    final_question_category[question] = 'semantically equivalent question'
                    count += 1

                    if count == remain_question_num:
                        break




        sorted_questions, similarity_score = self.question_reorder(original_question, final_question)

        obj['final_questions'] = sorted_questions
        obj['final_question_category'] = final_question_category
        obj['final_question_similarity_score'] = similarity_score

        return obj


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="select the model name")
    parser.add_argument("--dataset_name", type=str, default="truthfulQA", help="dataset name")
    parser.add_argument("--dataset_specify", type=str, default="", help="dataset name")
    parser.add_argument("--file_dic", type=str, default="/diverseagententropy", help="data file dictionary")
    parser.add_argument("--save_file", type=str, default="question_selection", help="save datafile name")
    parser.add_argument("--start", type=int, default=0, help="training epoch")
    parser.add_argument("--end", type=int, default=10000, help="training epoch")
    parser.add_argument("--num_agents", type=int, default=5, help="num_agents")
    parser.add_argument("--conceptualized_question", type=bool, default=True, help="conceptualized_question")


    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    pipe = Pipeline(args)

    with open(args.file_dic + "/result/question_generation/" + args.dataset_name + "_question_generation_"  + args.model_name.replace('/','-') +  "_"  + str(args.start) + ".json") as f:
        df_all = json.load(f)


    out_objs = []
    for i, df in enumerate(df_all):

        if i >= args.end: break
        if i < args.start: continue

        print(i)
        question = df['question']
        gold_answer = df['gold_answer']
        questions = df['aspect_questions']
        obj = pipe.run(question, questions, gold_answer)
        out_objs.append(obj)
        print(obj['final_questions'])
        print('\n')



        json.dump(out_objs, open(
            args.file_dic + "/result/question/" + args.dataset_name + "_" + args.save_file +  "_"  + args.model_name.replace('/','-') +  "_"  + str(args.start) + ".json",
            "w"), indent=4)



if __name__ == '__main__':
	main()

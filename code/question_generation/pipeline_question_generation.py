import json
import argparse
from collections import defaultdict
import time
from ..utils import ask_model

class Pipeline:

    def __init__(self, args):
        self.modelId = args.model_name
        self.num_agents = args.num_agents
        self.conceptualized_question = args.conceptualized_question


    def question_conceptualization(self,question):
        message = [
            {
                "role": "system",
                "content": """Can you identify the broader category of the specific entity referenced in the question? 
                If there is a specific entity, you MUST CHANGE it to a general category, e.g., a person, a item, a place, a object. If there is no specific entity, you MUST KEEP the original question."""
            },
            {
                "role": "user",
                "content": "What is the most spoken language in the world?"
            },
            {
                "role": "assistant",
                "content": "What is the most spoken language in the world?"
            },
            {
                "role": "user",
                "content": "What is Joe Biden's occupation?"
            },
            {
                "role": "assistant",
                "content": "What is a person's occupation?"
            },
        ]

        message.append({"role": "user", "content": question})
        question = ask_model(message, use_temp=0.7, modelId=self.modelId)
        return question


    def topic_sampling(self, question):
        message = [
            {
                "role": "system",
                "content": """Can you identify up to 5 key conceptual aspects that are as varied and diverse as possible, ensuring a comprehensive and multifaceted understanding of the question?
                Given ONLY the conceptual aspect name, no other words or explanation. The aspect SHOUlD NOT indicate the answer to the question.
                Each aspect is a line <as short as possible; not a complete sentence!>"""
            },
            {
                "role": "user",
                "content": "What is the most spoken language in the world?"
            },
            {
                "role": "assistant",
                "content": "demographic statistics\neducation policy\ncultural influence\ntechnology and media\nglobalization effects"
            },
        ]

        message.append({"role": "user", "content": question})
        response = ask_model(message, use_temp=0.7, modelId=self.modelId)
        topic = response.replace('\n\n','\n').replace('<','').replace('>','').split('\n')
        return topic, response

    def topic_ordering(self, question, topic):
        message = [
            {
                "role": "system",
                "content": """Please order the aspects listed below based on how likely people are to ask follow-up questions about each, starting with the most likely. Consider the direct relevance of each aspect to the original question, the complexity and scope of the information each might contain, and general public interest.
                        Given ONLY the ordered aspect name, no other words or explanation.
                        Each aspect is a line <as short as possible; not a complete sentence!>"""
            },
            {
                "role": "user",
                "content": """Question: What is the most spoken language in the world?
                Aspects to Consider: demographic statistics\neducation policy\ncultural influence\ntechnology and media\nglobalization effects"""
            },
            {
                "role": "assistant",
                "content": "demographic statistics\ncultural influence\nglobalization effects\ntechnology and media\neducation policy"
            },
        ]

        prompt = "Question: " + question + "\nAspects to Consider: "
        for t in topic:
            prompt += t + "\n"
        message.append({"role": "user", "content": prompt})
        response = ask_model(message, use_temp=0.7, modelId=self.modelId)
        topic = response.replace('\n\n','\n').replace('<','').replace('>','').split('\n')

        return topic, response


    def check_question_content_necessity(self, question, generated_questions):

        message = [
            {
                "role": "system",
                "content": """You are an extremely strict judge. Examine whether specific knowledge of the given question is strictly required to answer each subsequent question, ensuring that all contextual details MUST be considered. 
                Given ONLY the short explanation and the judge (Yes or No). You MUST follow the answer structure. 
                For example:
                Q1 Explanation: <ONLY say YES if the COMPLETE knowledge of the given question MUST BE EXPLICITLY REQUIRED to answer the subsequent question.>
                Q1 Judge: <judge, as short as possible; If words like potentially, possible, likely, could, would are in the Explanation, answer No>"""
            },
            {
                "role": "user",
                "content": "Given question: What is the most spoken language in the world?\nQ1: How does the prevalence of the most spoken language in the world influence global media and entertainment?\nQ2: In what ways has the most popular language in the world shaped international business and trade practices?\nQ3: How has the education system in non-native speaking countries adapted to teach the commonly spoken languages in the world?\nQ4: If I traveled to most countries in the world, which language would I be most likely to encounter?"
            },
            {
                "role": "assistant",
                "content": "Q1 Explanation: this question exactly requires the given question knowledge.\nQ1 Judge: Yes\nQ2 Explanation: \"the most popular language in the world\" in Q2 is different from \"the most spoken language in the world\" in the given question.\nQ2 Judge: No\nQ3 Explanation: t\"the commonly spoken languages in the world\" in Q3 is different from \"the most spoken language in the world\" in the given question.\nQ3 Judge: No\nQ4 Explanation: Although understanding which language is most spoken could inform this answer, the link is indirect and not explicitly required.\nQ4 Judge: No"
            },
        ]
        prompt = "Given Question: " + question + "\n" + generated_questions
        message.append({"role": "user", "content": prompt})
        response = ask_model(message, use_temp=0.15, modelId=self.modelId)

        response_list = response.replace('\n\n','\n').split('\n')
        generated_question_list = generated_questions.split('\n')
        count = 0
        final_question_list = []
        final_questions = []

        if len(response_list) == 2 * len(generated_question_list):
            for i in range(0, len(response_list), 2):
                final_question_list.append(
                    [generated_question_list[count].split(': ')[1], response_list[i + 1], response_list[i]])
                if 'Yes' in response_list[i + 1]:
                    final_questions.append(generated_question_list[count].split(': ')[1])
                count += 1
        elif len(response_list) == len(generated_question_list):
            for i in range(len(response_list)):
                final_question_list.append(
                    [generated_question_list[count].split(': ')[1], response_list[i]])
                if 'Yes' in response_list[i]:
                    final_questions.append(generated_question_list[count].split(': ')[1])
                count += 1

        return final_questions, final_question_list, response


    def check_question_content_answer(self, question, final_questions):


        message = [
            {
                "role": "system",
                "content": """You are an extremely strict judge. Using only the information from each subsequent question, can you directly answer the given question?
                        Given ONLY the short explanation and the judge (Yes or No). You MUST follow the answer structure. 
                        For example:
                        Q1 Explanation: <ONLY say YES if any possible answers to the given question is EXPLICITLY MENTIONED or INFERED in question Q1. >
                        Q1 Judge: <judge, as short as possible; not a complete sentence, just the judge!>"""
            },
            {
                "role": "user",
                "content": "Given question: What is the most spoken language in the world?\nQ1: How does the prevalence of the most spoken language Spanish influence global media and entertainment?\nQ2: How many children in China learn the most spoken language in the world, English?\nQ3: In what ways has the most popular language in the world shaped international business and trade practices?"#\nQ4: Among English, Chinese and Spanish, what is the most spoken language in the world?"
            },
            {
                "role": "assistant",
                "content": "Q1 Explanation: The answer to the most spoken language in the world is mentioned as Spanish.\nQ1 Judge: Yes\nQ2 xplanation: The answer to the most spoken language in the world is mentioned as English.\nQ2 Judge: Yes\nQ3 Explanation: The answer to the most spoken language in the world is not explicitly mentioned.\nQ3 Judge: No" #\nQ4 Explanation: Explicit hint to the answer of thr given question exists in the question, e.g., Among engligh, Chinese and Spanish.\nQ4 Judge: Yes"
            },

        ]



        prompt = "Given Question: " + question
        for i,question in enumerate(final_questions):
            prompt += '\nQ' + str(i+1) + ': ' + question

        message.append({"role": "user", "content": prompt})
        response = ask_model(message, use_temp=0.15, modelId=self.modelId)

        response_list = response.replace('\n\n','\n').split('\n')

        count = 0
        final_question_list = []
        final_questions_list = []
        if len(response_list) == 2 * len(final_questions):
            for i in range(0, len(response_list), 2):
                final_question_list.append(
                    [final_questions[count], response_list[i+1], response_list[i]])
                if 'No' in response_list[i+1]:
                    final_questions_list.append(final_questions[count])
                count += 1
        elif len(response_list) == len(final_questions):
            for i in range(len(response_list)):
                final_question_list.append(
                    [final_questions[count], response_list[i]])
                if 'No' in response_list[i]:
                    final_questions_list.append(final_questions[count])
                count += 1

        return final_questions_list, final_question_list, response



    def generic_topic_question(self, question, aspect):
        #Focus the questions on the broader concepts or categories related to the original question, rather than on specific details or objects mentioned within it.
        message = [
            {
                "role": "system",
                "content": """Generate 5 questions that build upon a given question based on a given aspect. Ensure that each question STRICTLY requires knowledge of this original question to answer but DO NOT INCLUDE the direct answer to the original question in your generated questions but MUST INCLUDE the exact content of the original question. These questions should encourage a deeper exploration of the underlying themes or concepts introduced in the original question.
                        Given ONLY the question, no other words or explanation.
                        For example:
                        Q1: <the generated question should be SIMPLE and DO NOT INCLUDE the direct answer to the original question>"""
            },
            {
                "role": "user",
                "content": "Question: What is the most spoken language in the world?\nAspect: cultural influence"
            },
            {
                "role": "assistant",
                "content": "Q1: How does the prevalence of the most spoken language in the world influence global media and entertainment?\nQ2: In what ways has the most spoken language in the world shaped international business and trade practices?\nQ3: How has the education system in non-native speaking countries adapted to teach the most spoken language in the world?\nQ4: What role does the most spoken language in the world play in diplomatic relations and international policy-making?\nQ5: How has the cultural heritage of the most spoken language in the world influenced global cuisine and fashion trends?"
            },
        ]

        prompt = "Question: " + question + "\nAspect: " + aspect

        message.append({"role": "user", "content": prompt})
        response = ask_model(message, use_temp=0.7, modelId=self.modelId)
        print(1, response)


        if "I apologize" in response or "I cannot generate questions" in response or "it's not possible to generate" in response or "confusion" in response or "it's not possible to ask" in response or "misunderstanding" in response or "there may be a mistake" in response or "there may be a problem" in response or "there was a mistake" in response or "This question doesn't make sense" in response or "I think there's a small issue here" in response or "try again with a different question" in response or "Let me try again" in response or "I think there might be some mistake!" in response or "the question is invalid" in response or "I cannot fulfill your request" in response or "I think there may be a bit of a problem here" in response or "I cannot provide a response" in response or "I assume the original question is incorrect" in response or "is not a meaningful or realistic question"  in response:
            return [], []

        final_questions, final_question_list, response_1 = self.check_question_content_necessity(question,response.replace('\n\n','\n'))
        final_question_list_1 = []
        response_2 = ''
        if len(final_questions) != 0:
            final_questions, final_question_list_1, response_2 = self.check_question_content_answer(question, final_questions)

        return final_questions, [question, aspect, final_question_list_1, final_question_list, response, response_1, response_2]

    def generate_semantically_equivalent_question(self,question):

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
        print(response)
        if len(response.replace('\n\n', '\n').split('\n')) != 5:
            sentence = f"You should not answer the question but generate 5 semantically equivalent questions. Regenerate your answer with no other explanations"
            message.append({"role": "assistant", "content": response})
            message.append({"role": "user", "content": question})
            response = ask_model(message, use_temp=0.7, modelId=self.modelId)

        return response.replace('\n\n', '\n').split('\n'), response


    def run(self, question, gold_answer):
        print(question)
        print(gold_answer)
        obj = {
            "question": question,
            "gold_answer": gold_answer
        }

        if self.conceptualized_question:
            concept_question = self.question_conceptualization(question)
            obj['conceptualized_question'] = concept_question
            aspects, aspects_response = self.topic_sampling(concept_question)
            aspects, ordered_response = self.topic_ordering(concept_question, aspects)
        else:
            aspects, aspects_response = self.topic_sampling(question)
            aspects, ordered_response = self.topic_ordering(question, aspects)

        final_questions = []
        question_generation_log = defaultdict(list)
        question_generation_log['aspect_sampling'] = [aspects_response]
        question_generation_log['aspect_ordering'] = [ordered_response]
        obj['aspect'] = aspects
        obj['aspect_questions'] = defaultdict(list)




        for aspect in aspects:
            questions, questions_log = self.generic_topic_question(question,aspect)
            question_generation_log[aspect + '_question_generation'] = [questions, questions_log]
            final_questions += questions
            obj['aspect_questions'][aspect] = questions

        semantically_equivalent_questions, response =  self.generate_semantically_equivalent_question(question)
        question_generation_log['semantic_question_generation'] =[semantically_equivalent_questions, question, response]
        final_questions += semantically_equivalent_questions
        obj['aspect_questions']['semantic_question_generation'] = semantically_equivalent_questions

        obj['question_generation_log'] = question_generation_log

        print(final_questions)
        obj['questions'] = final_questions

        return obj


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="select the model name")
    parser.add_argument("--dataset_name", type=str, default="truthfulQA", help="dataset name")
    parser.add_argument("--file_dic", type=str, default="/diverseagententropy", help="data file dictionary")
    parser.add_argument("--save_file", type=str, default="question_generation", help="save datafile name")
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


    with open( args.file_dic + "/data/" + args.dataset_name + ".json") as f:
        df_all = json.load(f)

    out_objs = []
    for i, df in enumerate(df_all):

        if i >= args.end: break
        if i < args.start: continue

        print(i)
        question = df['question']
        gold_answer = df['gold_answer']
        obj = pipe.run(question, gold_answer)
        out_objs.append(obj)
        print(obj['aspect_questions'])
        print('\n')

        json.dump(out_objs, open(
            args.file_dic + "/result/question_generation/" + args.dataset_name + "_" + args.save_file +  "_"  + args.model_name.replace('/','-') +  "_"  + str(args.start) + ".json",
            "w"), indent=4)

if __name__ == '__main__':
	main()






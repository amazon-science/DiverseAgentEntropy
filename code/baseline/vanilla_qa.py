import json
import argparse
from ..utils import ask_model

class Pipeline:

    def __init__(self, args):
        self.modelId = args.model_name
        self.num_self_consistency = args.num_self_consistency
        self.max_retries = args.max_retries

    def vanilla_qa(self, question):
        message = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people answer questions. Ensure your responses are concise and strictly relevant to the queries presented, avoiding any unrelated content to the question."
            }
        ]
        prompt = question
        response_list = []
        message.append({"role": "user", "content": prompt})
        for _ in range(self.num_self_consistency):
            response = ask_model(message, use_temp=0.7,modelId=self.modelId)
            print(response)
            response_list.append(response)
        return response_list



    def run(self, question, gold_answer, vanilla_answers=None):
        print(question)
        print(gold_answer)
        obj = {
            "question": question,
            "gold_answer": gold_answer
        }
        if vanilla_answers == None:
            vanilla_answers = self.vanilla_qa(question)
        obj['vanilla_answers'] = vanilla_answers
        return obj




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="select the model name")
    parser.add_argument("--dataset_name", type=str, default="truthfulQA", help="dataset name")
    parser.add_argument("--file_dic", type=str, default="/diverseagententropy", help="data file dictionary")
    parser.add_argument("--save_file", type=str, default="vanilla_qa", help="save datafile name")
    parser.add_argument("--use_temp", type=int, default=0.7, help="LLM's temperature")
    parser.add_argument("--start", type=int, default=0, help="data start index")
    parser.add_argument("--end", type=int, default=10000, help="data end indx")
    parser.add_argument("--num_self_consistency", type=int, default=5, help="num_self_consistency")
    parser.add_argument("--max_retries", type=int, default=5, help="max_retries")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    pipe = Pipeline(args)

    with open( args.file_dic + "/data/" + args.dataset_name + ".json") as f:
        df_all = json.load(f)

    count_acc = 0
    count_cons_acc = 0
    count_consist = 0
    out_objs = []
    for i, df in enumerate(df_all):

        if i >= args.end: break
        if i < args.start: continue

        print(i)
        question = df['question']
        gold_answer = df['gold_answer']
        vanilla_answers = None
        obj = pipe.run(question, gold_answer, vanilla_answers)
        out_objs.append(obj)
        print('\n')

        json.dump(out_objs, open(
            args.file_dic + "/result/baseline/" + args.dataset_name + "_" + args.save_file + "_" + args.model_name.replace('/','-') +  "_" + str(
                args.start) + ".json",
            "w"), indent=4)

if __name__ == '__main__':
	main()

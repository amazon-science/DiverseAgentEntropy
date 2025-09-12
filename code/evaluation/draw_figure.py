import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import json
import argparse
from collections import defaultdict
from ..utils import ask_model

def check_unknown(answer, modelId):

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
    response = ask_model(message, use_temp=0.15, modelId=modelId)
    return response


# You need to define the threshold logic and compute recall and precision for each threshold
def compute_precision_recall(scores, evaluations, unknown_labels, thresholds):
    precisions = []
    recalls = []
    total_queries = len(scores)


    for threshold in thresholds:

        # Identify queries where the model should abstain based on the uncertainty score
        score_abstain = scores > threshold
        # Identify queries where the model's prediction is 'I don't know'
        unknown_abstain = unknown_labels
        # Combine both conditions
        abstain = np.logical_or(score_abstain, unknown_abstain)
        answer_mask = ~abstain  # Queries where the model provides an answer

        if np.sum(answer_mask) == 0:
                continue

        correct_answers = evaluations[answer_mask]
        # print(len(correct_answers))
        precision = np.mean(correct_answers)
        recall = np.sum(answer_mask) / total_queries

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls

def get_data(df_all, modelId):
    uncertainty_score = []
    evaluation = []
    unknown_labels = []

    for i,df in enumerate(df_all):
        unknown_label = False

        if df['uncertainty_score'] <= 1e-10:
            score = 0
        else:
            score = df['uncertainty_score']
        uncertainty_score.append(score)
        if df['evaluation'] == "Yes" or df['evaluation'] == "YES":
            evaluation.append(1)
        elif df['evaluation'] == "No":
            evaluation.append(0)
        elif df['evaluation'] == "Unknown":
            evaluation.append(0)
            unknown_label = True
        else:
            evaluation.append(0)
            unknown_label = True

        if check_unknown(df['agent_final_answer'],modelId) == "I don't know":
            unknown_label = True

        if unknown_label:
            unknown_labels.append(True)
        else:
            unknown_labels.append(False)

    return uncertainty_score, evaluation, unknown_labels






def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testing_model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="select the model name")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="select the model name")
    parser.add_argument("--dataset_name", type=str, default="truthfulQA", help="dataset name")
    parser.add_argument("--file_dic", type=str, default="/diverseagententropy", help="data file dictionary")
    parser.add_argument("--save_file", type=str, default="agent_interaction", help="save datafile name")
    parser.add_argument("--use_temp", type=int, default=0.7, help="LLM's temperature")
    parser.add_argument("--start", type=int, default=0, help="data start index")
    parser.add_argument("--end", type=int, default=10000, help="data end index")
    parser.add_argument("--num_self_consistency", type=int, default=5, help="num_self_consistency")
    parser.add_argument("--max_retries", type=int, default=5, help="max_retries")
    parser.add_argument("--threshold", type=float, default=0.97, help="threshold")
    parser.add_argument("--mode", type=str, default="main", help="threshold")



    args = parser.parse_args()
    return args






def main():
    args = parse_args()
    print(args)

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})

    with open(args.file_dic + "/result/final_answer/baseline/" + args.dataset_name + "_vanilla_qa_"  + args.model_name.replace('/','-') + "_" + "0.json") as f:
        df_vanilla = json.load(f)

    uncertainty_score, evaluation, unknown_labels = get_data(df_vanilla, args.testing_model_name)

    # 0.4,0.6 majority vote threshold: 0.971
    precisions, recalls = compute_precision_recall(np.array(uncertainty_score), np.asarray(evaluation), unknown_labels,
                                                   np.array([0.971]))

    print('vanilla', precisions, recalls, 1 - float(recalls[0]))

    thresholds = np.linspace(0, 2, 100)
    precisions, recalls = compute_precision_recall(np.array(uncertainty_score), np.asarray(evaluation), unknown_labels,
                                                   np.array(thresholds))

    plt.plot(recalls, precisions, label='SemanticEntropy', color='green', linestyle='--', linewidth=3)


    with open(
            args.file_dic + "/result/final_answer/agent/" + args.dataset_name + "_" + args.save_file +  "_"  + args.model_name.replace('/','-') + "_" + str(
                args.start) + "_origin.json") as f:
        df_all = json.load(f)

    uncertainty_score, evaluation, unknown_labels = get_data(df_all, args.testing_model_name)
    precisions, recalls = compute_precision_recall(np.array(uncertainty_score), np.asarray(evaluation),
                                                   unknown_labels,
                                                   np.array([0.971]))

    print('agent', precisions, recalls, 1 - float(recalls[0]))

    thresholds = np.linspace(0, 2, 100)
    precisions, recalls = compute_precision_recall(np.array(uncertainty_score), np.asarray(evaluation),
                                                   unknown_labels,
                                                   np.array(thresholds))

    plt.plot(recalls, precisions, label='Agent with 5 questions', color='blue', linewidth=3)



    plt.xlabel('Recall',fontsize=20)
    plt.ylabel('Accuracy',fontsize=20)
    plt.xticks(fontsize=18)  # Set font size for x-axis tick labels
    plt.yticks(fontsize=18)  # Set font size for y-axis tick labels
    if "llama" in args.model_name:
        plt.title(args.dataset_name + ' ' +args.model_name[0:-5])
    if "claude" in args.model_name:
        plt.title(args.dataset_name + ' ' +args.model_name[0:-14])
    plt.title(args.dataset_name + ' ' + args.model_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(args.file_dic + "/result/figure/" + args.mode + "/" + args.dataset_name + "_" + args.save_file +  "_"  + args.model_name.replace('/','-') + ".png")
    plt.show()


if __name__ == '__main__':
	main()
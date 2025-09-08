# DiverseAgentEntropy

This is the repo to reproduce the paper [DiverseAgentEntropy](https://arxiv.org/abs/2412.09572), a novel method for quantifying the uncertainty in the factual parametric knowledge of Large Language Models (LLMs).

## Setup 

Please install the packages in ```requirements.txt```
 
### Data preparation
You should prepare QA datasets in the following format under ```../data```:
```
[
  {
    "question": "<string: the input question>",
    "gold_answer": "<string: the authoritative or ground-truth answer>",
  },
  {
    "question": "<string: the input question>",
    "gold_answer": "<string: the authoritative or ground-truth answer>",
  },
  // ... more entries ...
]
```

### API inference
We are currently using the AWS Bedrock platform, but you can switch to any other API call method by updating the ```../code/utils.py``` file accordingly.

### Diverse agent question generation
We first generate varied questions about the same underlying original query with different contexts.
```
python -m code.question_generation.pipeline_question_generation --dataset_name=dataset_name --model_name=model_name 
```
We then select n questions as the final questions for agent interaction. 
```
python -m code.question_generation.pipeline_question_selection --dataset_name=dataset_name --model_name=model_name 
```

### Agent Interaction
We use the selected diverse questions to encourage agent interactions to further reveal the model's knowledge of the original query. 

```
python -m code.agent_interaction.pipeline_agent_interaction  --dataset_name=dataset_name --model_name=model_name --mode=origin
```
We then implement the abstention policy to get the uncertainty score for each original query. 
```
python -m code.evaluation.agent_evaluation  --dataset_name=dataset_name --model_name=model_name --mode=origin
```


### Baseline
We use the following code to run the self-consistency-based SemanticEntropy baseline. 

```
python -m code.baseline.vanilla_qa  --dataset_name=dataset_name --model_name=model_name
```
We evaluate the baseline:
```
python -m code.evaluation.vanilla_evaluation  --dataset_name=dataset_name --model_name=model_name
```



### Evaluation
Finally, we evaluate different methods and draw the AR-curve.

```
python -m code.evaluation.draw_figure  --dataset_name=dataset_name --model_name=model_name
```


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


## 🐛 Issues

If you encounter any issues or have questions, please open an issue on GitHub or contact the authors.
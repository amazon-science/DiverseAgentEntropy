import boto3
import time

def ask_model(messages, max_token=256, use_temp=1, top_p=1, modelId='meta.llama3-70b-instruct-v1:0'):

    brt = boto3.client(service_name='bedrock-runtime')

    if 'llama' in modelId:
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>".format(message["content"])
            elif message["role"] == "user":
                prompt += "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>".format(message["content"])
            elif message["role"] == "assistant":
                prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>".format(message["content"])
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        body = json.dumps({
            "prompt": prompt,
            "max_gen_len": max_token,
            "temperature": use_temp,
            "top_p": top_p,
        })
    elif 'claude' in modelId:

        prompt_messages = []
        system_message = ''
        for message in messages:
            if message["role"] == "system":
                system_message = message["content"]
            elif message["role"] == "user":
                prompt_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": message["content"]
                        }]})
            elif message["role"] == "assistant":
                prompt_messages.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": message["content"]
                        }]})
        if system_message != '':
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_token,
                "temperature": use_temp,
                "top_p": top_p,
                "system": system_message,
                "messages": prompt_messages
            })
        else:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_token,
                "temperature": use_temp,
                "top_p": top_p,
                "messages": prompt_messages
            })

    modelId = modelId  # 'anthropic.claude-v2'
    accept = 'application/json'
    contentType = 'application/json'

    response = brt.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

    response_body = json.loads(response.get('body').read())

    if 'llama' in modelId:
        return response_body.get('generation')
    elif 'claude' in modelId:
        return response_body['content'][0]['text']


import os

import openai
from util.prompt_generator import *
import csv
import json


openai.api_key = "your-openai-api-key"
model_engine = "gpt-4"


def call_chatgpt_for_summary(
        part1_metadata,
        project_tree,
        prev_summary,
        call_graph,
        usage_context
):

    system_prompt = """
    You are an AI code summarization specialist with deep Java expertise. 
    Enhance the previous summary using code contexts (call graph and usage) from a project-level view.
    Follow the steps:
    1. Identify gaps in the previous summary based on contexts.
    2. Reflect on project-level aspects (architecture, tech stack, usage).
    3. Polish to max 100 words.
    """

    user_prompt = f"""
    part1:
    Project tree: {project_tree}
    Metadata: {part1_metadata}

    part2:
    Previous summary: "{prev_summary}"

    part3:
    Call graph: {call_graph}
    Usage context: {usage_context}
    """

    restrictions = """
    Now, think step-by-step:
    1. Identify what the previous summary lacks when considering the given code contexts.
    2. Reflect whether your proposed enhancement addresses those gaps from a broader, project-level view.
    3. Revise your enhancement accordingly.
    
    Only after this reflection, generate your final summary (max 100 words).
    """

    # 调用API
    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + restrictions}
        ],
        temperature=0.7,
        max_tokens=200
    )

    final_summary = response.choices[0].message["content"].strip()
    return final_summary


if __name__ == "__main__":
    data_path = './data/'
    for repo in os.listdir(data_path):
        for file in os.listdir(data_path + '/' + repo):
            enhanced_summary = call_chatgpt_for_summary(
                get_metadata(file),
                get_project_tree(repo),
                get_prev_summary("./model/sota/", file),
                get_call_graph(file),
                get_usage_context(file)
            )
            output_file = f'./result/{repo}.csv'
            with open(output_file, newline="", encoding="utf-8-sig") as csvfile:
                writer = csv.DictWriter(csvfile)
                writer.writerow(file.split("/")[-1] + enhanced_summary)


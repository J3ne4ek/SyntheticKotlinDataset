from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
import time

from typing import Callable, List

import openai
from datasets import load_dataset
from openai import OpenAIError

from rich.progress import (
    Progress,
    TimeElapsedColumn,
)

from tqdm import tqdm


class Generator:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
    ):
        self.model = model

    def generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": """ The resulted Kotlin code must be of the style: 

            ```
            /**
             * Detailed description of the function's purpose.
             *
             * @param paramName Description of each parameter.
             * @return Description of the return value, if applicable.
             */
            fun functionName(parameters): ReturnType {
                // Kotlin code implementation
            }
            ```""",
            },
            {"role": "user", "content": prompt},
        ]

        chat_completion = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=250,
            timeout=60,
        )

        return chat_completion.choices[0].message.content


def generation(
    prompt: str,
    generator: Generator,
    update_progress: Callable,
    retries: int,
) -> str:
    success = False
    time.sleep(random.random())
    for i in range(retries):
        try:
            result = generator.generate(prompt)
            success = True
        except OpenAIError as e:
            print(e)
            time.sleep(1)
        else:
            break

    if success:
        exercise = result
        update_progress()
        return exercise

    else:
        print(f"Generation failed for prompt {prompt}, skipping")
        return ""


def _generation_wrapper(
    prompt: str,
    update_progress: Callable,
    save_dir: str,
    retries: int,
):
    file_path = "results_draft"
    file_path = os.path.join(save_dir, file_path + ".jsonl")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    generator = Generator()

    result = generation(prompt, generator, update_progress, retries)
    write_results_to_jsonl(file_path, result)


def mass_generation(
    prompts: List[str],
    save_dir: str,
    pool_size: int = 10,
    retries: int = 10,
):
    """
    Generate from a list of prompts. Use a thread pool to parallelize the generation with catch and
    retry mechanism
    """
    with Progress(
        *Progress.get_default_columns(), "•", TimeElapsedColumn()
    ) as progress:
        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            progress_task = progress.add_task("[red]Generating...", total=len(prompts))

            def update_progress():
                progress.update(progress_task, advance=1)

            tasks = []

            for prompt in prompts:
                tasks.append(
                    executor.submit(
                        _generation_wrapper,
                        prompt,
                        update_progress,
                        save_dir,
                        retries,
                    )
                )

            for task in tasks:
                try:
                    task.result()
                except Exception as e:
                    print(e)


def write_results_to_jsonl(file_path: str, result: str):
    line = {"code": result}
    with open(file_path, "a") as file:
        json.dump(line, file)
        file.write("\n")


def create_prompt(problem, solution):
    prompt = f"""Translate the following Python code into Kotlin.

                {problem}
                {solution}
                
                Please provide the equivalent Kotlin code. Move comment with task description comment 
                before function definition. Respond only with a Kotlin code as a plane string, don't 
                use code block marks."""

    return prompt


def main():
    prompts = []
    dataset = load_dataset("jinaai/code_exercises", split="train[:10000]")

    for i, item in tqdm(enumerate(dataset)):
        prompts.append(create_prompt(item["problem"], item["solution"]))

    output_path = "../kotlin_dataset"
    mass_generation(
        prompts,
        save_dir=output_path,
        pool_size=10,
        retries=10,
    )


if __name__ == "__main__":
    main()

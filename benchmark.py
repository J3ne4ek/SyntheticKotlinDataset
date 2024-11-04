import argparse
import json
import re

from datasets import load_dataset
import jsonlines
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from tqdm import tqdm
from mxeval.evaluation import evaluate_functional_correctness


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops, tokenizer):
        (StoppingCriteria.__init__(self),)
        self.stops = rf"{stops}"
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        last_three_tokens = [int(x) for x in input_ids.data[0][-3:]]
        decoded_last_three_tokens = self.tokenizer.decode(last_three_tokens)

        return bool(re.search(self.stops, decoded_last_three_tokens))


def generate(problem, model, tokenizer):
    criterion = StoppingCriteriaSub(stops="\n}\n", tokenizer=tokenizer)
    stopping_criteria = StoppingCriteriaList([criterion])

    problem = tokenizer.encode(problem, return_tensors="pt").to("cuda")
    sample = model.generate(
        problem,
        max_new_tokens=256,
        min_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        num_beams=1,
        stopping_criteria=stopping_criteria,
    )

    answer = tokenizer.decode(sample[0], skip_special_tokens=True)
    return answer


def clean_answer(code):
    # Clean comments
    code_without_line_comments = re.sub(r"//.*", "", code)
    code_without_all_comments = re.sub(
        r"/\*.*?\*/", "", code_without_line_comments, flags=re.DOTALL
    )
    # Clean signatures
    lines = code.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("fun "):
            return "\n".join(lines[i + 1 :])

    return code


def create_answers(problem_dict, model, tokenizer, output_file):
    output = []
    for key in tqdm(list(problem_dict.keys()), leave=False):
        problem = problem_dict[key]["prompt"]
        answer = generate(problem, model, tokenizer)
        answer = clean_answer(answer)
        output.append({"task_id": key, "completion": answer, "language": "kotlin"})

    with jsonlines.open(output_file, mode="w") as writer:
        for line in output:
            writer.write(line)


def main(args):
    model_name = args.model_path
    dataset = load_dataset("jetbrains/Kotlin_HumanEval")["train"]
    problem_dict = {problem["task_id"]: problem for problem in dataset}

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    output_file = f"answers"

    create_answers(problem_dict, model, tokenizer, output_file)

    evaluate_functional_correctness(
        sample_file=output_file,
        k=[1],
        n_workers=16,
        timeout=15,
        problem_file=problem_dict,
    )

    with open(output_file + "_results.jsonl") as fp:
        total = 0
        correct = 0
        for line in fp:
            sample_res = json.loads(line)
            print(sample_res)
            total += 1
            correct += sample_res["passed"]

    print(f"Pass rate: {correct / total * 100}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="Salesforce/codegen-350M-mono",
        help="Provide path to model. "
        "It's local path with jsonl file for ExerciseDataset and name for hf kotlin_dataset ",
    )

    args = parser.parse_args()
    main(args)

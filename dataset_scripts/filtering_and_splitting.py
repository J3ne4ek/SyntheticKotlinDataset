import json


def split_kotlin_code(code_str):
    if "{" not in code_str:
        return None, None

    start_index = code_str.index("{")
    problem = code_str[: start_index + 1].strip()
    solution = code_str[start_index + 1 :].strip()

    return problem, solution


def main():
    items = []
    with open("kotlin_dataset/results_draft.jsonl") as file:
        for i, line in enumerate(file):
            code = json.loads(line)["code"]
            if code.startswith("/**\n"):
                problem, solution = split_kotlin_code(code)
                if problem is not None:
                    items.append({"problem": problem, "solution": solution})


if __name__ == "__main__":
    main()

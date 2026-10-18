"""Type a sentence, see the plan and the 13 numbers each policy would receive (no simulator).

    python -m pet.planner.cli [--start sitting]      needs OPENAI_API_KEY in the environment
"""

import argparse

from .llm import plan_text, plan_with_retry


def main(start: str) -> None:
    print(f"Duck starts {start}. Type a sentence (empty line to quit).\n")
    while True:
        sentence = input("you> ").strip()
        if not sentence:
            break
        res = plan_with_retry(sentence, start)
        print(f"   status: {res.status} ({res.llm_calls} LLM call(s))")
        if res.steps:
            print(f"   {plan_text(res.steps)}")
        for line in res.log:
            print(f"   log: {line}")
        for policy, cmd in res.commands:
            print(f"   -> {policy:<8} [{', '.join(f'{v:.2f}' for v in cmd)}]")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", choices=["standing", "sitting", "fallen"], default="standing")
    main(ap.parse_args().start)

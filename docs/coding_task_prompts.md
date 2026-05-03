# Coding Task Prompt Summary

This file summarizes the prompt families currently used for the main coding-task evaluations in this repository.

## Overview

| Task | Prompt family | Default prompt style in repo | CoT by default | Actual output requirement |
|---|---|---|---|---|
| MMLU-coding | OpenAI-style simple-evals multiple choice | `simple_evals` | Yes | Last line should be `Answer: $LETTER` |
| CodeMMLU | simple-evals-style multiple choice | `simple_evals` | No | Exactly one line: `Answer: $LETTER` with no reasoning or extra text |
| HumanEval / EvalPlus | Official code generation prompt | EvalPlus runner prompt | No explicit CoT prompt | Generate a self-contained Python solution/script |
| BigCodeBench-Hard | Official code generation prompt | BigCodeBench runner prompt | No explicit CoT prompt | Generate a self-contained Python script in a markdown code block |

## 1. MMLU-coding

Default path:

- task name: `mmlu_coding`
- default prompt style: `simple_evals`
- default resolution logic: [baseline.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/baseline.py)
- task-specific prompt builder: [mmlu_coding.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/tasks/mmlu_coding.py)

Default prompt template:

```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}
```

Notes:

- This is the current default for `MMLU-coding`.
- It is a multiple-choice prompt in an OpenAI simple-evals style.
- It explicitly asks the model to think step by step.
- The evaluator expects the final answer to be recoverable from `Answer: $LETTER`.

Supported alternate prompt styles:

- `simple_evals_nonthinking`

```text
{question}

Respond with only the single letter of the correct answer (A, B, C, or D).
```

- `reasoning_boxed`

```text
{question}

Think step by step, then give the final answer as a single letter in \boxed{}.
```

## 2. CodeMMLU

Default path:

- task name: `codemmlu`
- default prompt style: `simple_evals`
- default resolution logic: [baseline.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/baseline.py)
- task-specific prompt builder: [codemmlu.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/tasks/codemmlu.py)

Default prompt template:

```text
Answer the following multiple choice question. Reply with exactly one line in this format:
Answer: $LETTER

LETTER must be one of the listed option letters. Do not include reasoning, explanations, code fences, or any extra text.

{question}
```

Notes:

- This is also labeled `simple_evals`, but unlike `MMLU-coding`, it is non-CoT by default.
- It is much stricter about output format.
- The intended output is exactly one final answer line.

Supported alternate prompt styles:

- `simple_evals_cot`

```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of the listed option letters. Think step by step before answering.

{question}
```

- `simple_evals_nonthinking`

```text
{question}

Respond with only the single letter of the correct answer.
```

## 3. HumanEval / EvalPlus

Execution path:

- runner implementation: [modal_feasibility_app.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/modal_feasibility_app.py)
- dataset flow: EvalPlus direct HF runner

Prompt framing used by the runner:

```text
Please provide a self-contained Python script that solves the following problem in a markdown code block:
```

Response prefix used by the chat wrapper:

```text
Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:
```

Notes:

- This is not a multiple-choice prompt.
- This is a code generation prompt.
- The current setup does not inject an explicit chain-of-thought instruction.
- The model is expected to produce valid Python code that solves the task.

## 4. BigCodeBench-Hard

Execution path:

- runner implementation: [modal_feasibility_app.py](/Users/aaronyu/Desktop/TA-MPQ/src/ta_mpq/modal_feasibility_app.py)
- dataset flow: BigCodeBench direct HF runner

Prompt framing used by the runner:

```text
Please provide a self-contained Python script that solves the following problem in a markdown code block:
```

Response prefix used by the chat wrapper:

```text
Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:
```

Notes:

- This is a generation-style benchmark, not a multiple-choice benchmark.
- The model is expected to output a self-contained Python script.
- The current setup does not add an explicit CoT instruction.
- In practice, this prompt family is much closer to HumanEval than to MMLU-style evaluation.

## Key Difference Summary

- `MMLU-coding` uses a multiple-choice prompt with CoT by default.
- `CodeMMLU` uses a stricter multiple-choice prompt without CoT by default.
- `HumanEval` and `BigCodeBench-Hard` use code-generation prompts rather than multiple-choice prompts.
- So even though all four are “coding tasks,” they do not use one unified prompt family.

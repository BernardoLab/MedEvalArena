"""Shared prompt templates for quiz-taking flows."""

# Header prepended to each MCQ presented to models during evaluation.
EVAL_HEADER = (
    "NOT FOR CLINICAL USE.\n"
    "You are taking a multiple-choice exam. Choose the single best option.\n"
    "At the very end, you MUST write exactly: The answer is (X)\n"
)

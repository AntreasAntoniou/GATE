import pytest
from gate.metrics.vqa_eval import VQAEval, AnswerData, VQAItem
import numpy as np

# Sample data
sample_data = {
    1: VQAItem(
        question_type="is the",
        answers=[
            AnswerData(answer="yes", answer_confidence="yes", answer_id=1),
            AnswerData(answer="yes", answer_confidence="yes", answer_id=2),
            AnswerData(answer="yes", answer_confidence="yes", answer_id=3),
            AnswerData(answer="yes", answer_confidence="yes", answer_id=4),
            AnswerData(answer="yes", answer_confidence="yes", answer_id=5),
            AnswerData(answer="yes", answer_confidence="maybe", answer_id=6),
            AnswerData(answer="no", answer_confidence="maybe", answer_id=7),
            AnswerData(answer="yes", answer_confidence="yes", answer_id=8),
            AnswerData(answer="yes", answer_confidence="yes", answer_id=9),
            AnswerData(answer="yes", answer_confidence="yes", answer_id=10),
        ],
        image_id=1,
        answer_type="yes/no",
        question_id=1,
        question="Is the food eaten?",
    ),
    2: VQAItem(
        question_type="is the",
        answers=[
            AnswerData(answer="apple", answer_confidence="yes", answer_id=1),
            AnswerData(answer="orange", answer_confidence="yes", answer_id=2),
            AnswerData(answer="banana", answer_confidence="yes", answer_id=3),
            AnswerData(answer="apple", answer_confidence="yes", answer_id=1),
            AnswerData(answer="apple", answer_confidence="yes", answer_id=1),
            AnswerData(answer="apple", answer_confidence="yes", answer_id=1),
            AnswerData(answer="apple", answer_confidence="yes", answer_id=1),
            AnswerData(answer="apple", answer_confidence="yes", answer_id=1),
            AnswerData(answer="apple", answer_confidence="yes", answer_id=1),
        ],
        image_id=1,
        answer_type="what is",
        question_id=1,
        question="What is the item on the table?",
    ),
}

# sample_data_dict = {
#     1: {
#         "question_type": "is the",
#         "answer_type": "yes/no",
#         "answers": [
#             {"answer": "yes"},
#             {"answer": "yes"},
#             {"answer": "yes"},
#             {"answer": "yes"},
#             {"answer": "yes"},
#             {"answer": "yes"},
#             {"answer": "yes"},
#             {"answer": "no"},
#             {"answer": "yes"},
#             {"answer": "yes"},
#         ],
#     },
#     2: {
#         "question_type": "what is",
#         "answer_type": "other",
#         "answers": [
#             {"answer": "apple"},
#             {"answer": "orange"},
#             {"answer": "banana"},
#             {"answer": "apple"},
#             {"answer": "apple"},
#             {"answer": "apple"},
#             {"answer": "apple"},
#             {"answer": "apple"},
#             {"answer": "apple"},
#             {"answer": "apple"},
#         ],
#     },
# }

# Test cases for predictions
test_data = [
    (1, "yes", {"overall": [1.0], "is the": [1.0]}),
    (1, "no", {"overall": [1.0 / 3.0], "is the": [1.0 / 3.0]}),
    (2, "apple", {"overall": [1.0], "what is": [1.0]}),
    (2, "orange", {"overall": [1.0 / 3.0], "what is": [1.0 / 3.0]}),
    (2, "bike", {"overall": [0.0], "what is": [0.0]}),
]


@pytest.mark.parametrize(
    "question_id, predicted_answer, expected_result", test_data
)
def test_vqa_eval(question_id, predicted_answer, expected_result):
    # Create a VQAEval instance
    vqa_eval = VQAEval(
        vqa_data=sample_data,
        vqa_predictions={question_id: AnswerData(answer=predicted_answer)},
    )

    # Run the evaluation
    result = vqa_eval.evaluate(question_ids=[question_id])

    # Check if the evaluation results match the expected results
    for key in expected_result:
        for res, exp in zip(result[key], expected_result[key]):
            assert np.isclose(res, exp)

from ochat.evaluation.grading import math_grader


def default_compare_answer(answer, label):
    assert isinstance(label, list)

    return answer in label


COMPARE_ANSWER_FUNCTION = {
    "zs/agieval": default_compare_answer,
    "zs/bbh_mc_orca": default_compare_answer,
    "zs/truthfulqa_orca": default_compare_answer,

    "fs_cothub/bbh": default_compare_answer,
    "fs_cothub/gsm8k": default_compare_answer,

    "special/prm800k_math": math_grader.grade_answer
}

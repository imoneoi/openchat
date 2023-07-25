import re


def zs_agieval_match_answer(task_data, response):
    # AGIEval match first capital letter, following original paper implementation
    # https://github.com/microsoft/AGIEval/blob/main/src/post_process.py

    letter_set = {"A", "B", "C", "D", "E", "F"}
    for c in response:
        if c in letter_set:
            return True, c

    return False, ""


def zs_bbh_mc_orca_truthfulqa_orca_match_answer(task_data, response):
    # For BBH & TruthfulQA, match first option letter

    for c in response:
        if c in task_data["options"]:
            return True, c

    return False, ""


def fs_cothub_bbh_match_answer(task_data, response):
    # CoT hub match answer for BBH
    # https://github.com/FranxYao/chain-of-thought-hub/blob/main/BBH/run_bbh_gpt_3.5_turbo.py

    ans_line = response.split('answer is ')

    # Expect to see 'answer is'. If not return whole string
    if len(ans_line) == 1:
        return False, response
    else:
        ans = ans_line[-1].strip()

    if task_data["options"]:
        # Multiple choice, find appearing letter
        options = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)', '(O)', '(P)', '(Q)', '(R)', '(S)', '(T)', '(U)', '(V)', '(W)', '(X)', '(Y)', '(Z)']

        for option in options:
            if option in ans:
                return True, option

        return False, ans
    else:
        # Free form, direct return
        if ans[-1] == '.':
            ans = ans[:-1]

        return True, ans


def fs_cothub_gsm8k_match_answer(task_data, response):
    # CoT hub match answer for GSM8k, match last numeric value
    # https://github.com/FranxYao/chain-of-thought-hub/blob/main/gsm8k/gpt3.5turbo_gsm8k_complex.ipynb

    pattern = '\d*\.?\d+'
    pred = re.findall(pattern, response)
    if len(pred) >= 1:
        return True, pred[-1]

    return False, response


MATCH_ANSWER_FUNCTION = {
    "zs/agieval": zs_agieval_match_answer,
    "zs/bbh_mc_orca": zs_bbh_mc_orca_truthfulqa_orca_match_answer,
    "zs/truthfulqa_orca": zs_bbh_mc_orca_truthfulqa_orca_match_answer,

    "fs_cothub/bbh": fs_cothub_bbh_match_answer,
    "fs_cothub/gsm8k": fs_cothub_gsm8k_match_answer
}

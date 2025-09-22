from rllm.rewards.math_reward import rllm_reward_fn_math


def rllm_reward_fn_math_transformed(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
):
    """
    Transforms inputs expected by verl compute_score() to inputs expected by rllm_reward_fn_math()
    """
    reward_output = rllm_reward_fn_math(data_source=data_source, llm_solution=solution_str, ground_truth=ground_truth, extra_info=extra_info)
    
    return reward_output.reward


if __name__ == "__main__":
    action = "<think>...</think>\nThe answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}."
    ground_truth = "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"

    output = rllm_reward_fn_math_transformed(
        data_source="",
        solution_str=action,
        ground_truth=ground_truth,
    )
    print(output)

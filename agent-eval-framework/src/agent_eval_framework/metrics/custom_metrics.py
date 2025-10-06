def keyword_match(prompt: str, reference: str, actual_response: str, actual_trajectory: list) -> dict:
    """
    A sample custom metric that checks for a keyword in the response.

    Args:
        prompt (str): The input prompt given to the agent.
        reference (str): The expected or "golden" response.
        actual_response (str): The final text response from the agent.
        actual_trajectory (list): The sequence of tool calls made by the agent.

    Returns:
        A dictionary with a "score" key (float or int) and an optional "reason" key (str).
    """
    score = 0.0
    reason = "Initial score"

    # Example: Check if the response contains the word "capital"
    if "capital" in actual_response.lower():
        score = 1.0
        reason = "The response contained the keyword 'capital'."
    else:
        score = 0.0
        reason = "The response did not contain the keyword 'capital'."

    return {"score": score, "reason": reason}
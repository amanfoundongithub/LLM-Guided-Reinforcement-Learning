import ollama
from utils.helper import convert_minigrid_to_text_S3R1
import re
import numpy as np 

# Initialize the Ollama client
client = ollama.Client()

# Define the model
model = 'llama3.2'

def suggest_llm_action(env, env_name, action_dim, state):
    # Convert the state to an LLM-readable format
    llm_readable_format = convert_minigrid_to_text_S3R1(env, state)

    # Prepare the prompt
    prompt = f"""
    You are given an RL environment '{env_name}'.
    The current state is: 
    '{llm_readable_format}'
    
    You are an RL expert. Using your knowledge of the environment, suggest a possible action.
    Respond with a single integer in the range 0 to {action_dim - 1}, do not tell any reasoning.
    
    Description of actions:
    - 0: turn left
    - 1: turn right
    - 2: move forward
    - 3: pickup
    - 4: drop
    - 5: toggle (to open/close doors)
    - 6: done/noop
    """

    # Call the model and get the response 
    response = client.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
    response_text = response['message']['content']

    
    # Attempt to extract the first integer from the response
    match = re.search(r'\b([0-6])\b', response_text)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"LLM response: {response_text}")


def suggest_llm_distribution(env, env_name, action_dim, state):
    # Convert the state to an LLM-readable format
    llm_readable_format = convert_minigrid_to_text_S3R1(env, state)

    # Prepare the prompt
    prompt = f"""
    You are given an RL environment '{env_name}'.
    The current state is: 
    '{llm_readable_format}'

    You are an RL expert. Based on your understanding of the environment and the current state,
    estimate the probabilities of taking each of the {action_dim} actions.

    Respond with a list of {action_dim} floats representing the probabilities, in the following order:
    - 0: turn left
    - 1: turn right
    - 2: move forward
    - 3: pickup
    - 4: drop
    - 5: toggle (to open/close doors)
    - 6: done/noop

    Normalize all values to make the sum 1.
    Do not include any explanation or text, just return the list.
    """

    # Call the model and get the response
    response = client.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
    response_text = response['message']['content'] 

    # Extract the list of floats from the response using regex
    match = re.findall(r"\[([^\]]+)\]", response_text)
    if match:
        try:
            probs = [float(x.strip()) for x in match[0].split(',')]
            probs = np.array(probs)
            if len(probs) != action_dim or not np.isclose(probs.sum(), 1.0):
                raise ValueError("Output is not a valid probability distribution.")
            return probs
        except Exception as e:
            raise ValueError(f"Error: {response_text}") from e
    else:
        raise ValueError(f"Error: {response_text}")
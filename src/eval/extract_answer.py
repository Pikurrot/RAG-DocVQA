import os
from openai import OpenAI
from typing import Optional

def extract_answer(
    question: str,
    output: str,
    prompt: Optional[str] = None,
    model_name: str = "gpt-4o"
) -> str:
    """
    Extract structured answer from model output using GPT-4o.
    
    Args:
        question: The question asked
        output: Model's raw output
        prompt: Optional prompt template
        model_name: OpenAI model to use
    
    Returns:
        Extracted answer or original output if extraction fails
    """
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        return output  # Return original output if no API key
        
    try:
        client = OpenAI()
        
        # Construct the prompt
        if prompt:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Question: {question}\nModel Output: {output}"}
            ]
        else:
            messages = [
                {"role": "system", "content": "Extract the answer from the model output. Return only the answer, nothing else."},
                {"role": "user", "content": f"Question: {question}\nModel Output: {output}"}
            ]
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error in answer extraction: {str(e)}")
        return output  # Return original output if extraction fails 
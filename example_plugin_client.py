import requests

def audit_llm_response(question: str, answer: str, topic: str = "general"):
    """
    Plug-in function to check an LLM response for hallucinations.
    Point this to your running AuditTrail API.
    """
    url = "http://127.0.0.1:8000/api/qa-audit"
    
    payload = {
        "question": question,
        "answer": answer,
        "topic": topic
        # Context is missing, so the API will automatically search Google!
    }
    
    try:
        print(f"Checking answer for: '{question}'...")
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        print("\n=== AUDIT RESULT ===")
        print(f"Verdict: {result['verdict'].upper()}")
        print(f"Hallucination Score: {result['ensemble_score']:.2f} / 1.00")
        
        # Show what sources the API found
        print("\nGround-Truth Sources Found:")
        for source in result.get("context_sources", []):
            print(f"- {source['title']} ({source['url']})")
            
        return result
        
    except Exception as e:
        print(f"Error calling Audit API: {e}")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    # Imagine this is another project (like a chatbot)
    user_question = "Who is the current President of France?"
    
    # Let's say your chatbot generated this wrong answer:
    chatbot_answer = "Nicolas Sarkozy is the current President of France."
    
    # Plug in the audit check!
    audit_llm_response(user_question, chatbot_answer)

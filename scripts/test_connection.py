"""
This script tests two capabilities: Embeddings 
and Chat Completions (send instructions and questions to an AI model and receive generated answers)
"""

from pharma_research_agent.config import (
    OPENAI_EMBEDDING_MODEL,
    OPENAI_MODEL,
    create_client,
)


def main():
    print("Testing Syntax API connection...")

    try:
        client = create_client()

        # Test 1: Embeddings
        print("\n[1/2] Testing embeddings...")

        embedding_response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=["Warfarin is an anticoagulant medication."],
        )

        vector = embedding_response.data[0].embedding

        print("Embeddings: SUCCESS")
        print(f"Vector dimensions: {len(vector)}")

        # Test 2: Chat completion
        print("\n[2/2] Testing chat completion...")

        chat_response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly: API connection successful.",
                }
            ],
            max_tokens=30,
        )

        answer = chat_response.choices[0].message.content

        print("Chat completion: SUCCESS")
        print(f"Model response: {answer}")

        print("\nAll API tests passed!")

    except Exception as exc:
        print(f"\nAPI test failed: {type(exc).__name__}")
        print("Check your local configuration, endpoint, model access, and NETWORK QUALITY.")


if __name__ == "__main__":
    main()

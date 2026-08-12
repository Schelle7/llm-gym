import json

from config import MAX_ITERATIONS, MODEL, client, create_messages
from tools import tool_definitions, tool_registry


def run_agent(user_prompt):
    messages = create_messages(user_prompt)

    for iteration in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tool_definitions,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message
        print(f"Model response {iteration + 1}: {assistant_message}")
        messages.append(assistant_message.model_dump(exclude_none=True))

        if not assistant_message.tool_calls:
            print(assistant_message.content)
            return

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_arguments = json.loads(tool_call.function.arguments)
            tool_function = tool_registry[tool_name]
            tool_result = tool_function(**tool_arguments)
            print(f"Tool call: {tool_name}({tool_arguments})")
            print(f"Tool result: {tool_result}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result),
                }
            )

    raise RuntimeError("The model exceeded the maximum number of tool iterations")


def main():
    while True:
        user_prompt = input("Write prompt: ")
        run_agent(user_prompt)


if __name__ == "__main__":
    main()
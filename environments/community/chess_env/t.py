import asyncio

from rich import print as rprint
from transformers import AutoTokenizer

from atroposlib.envs.base import APIServerConfig, ServerManager


async def test():

    server = ServerManager(
        configs=[
            APIServerConfig(
                tokenizer_name="./base_model/checkpoint-168",
                base_url="http://localhost:9000/v1",
                api_key="x",
                num_requests_for_eval=256,
                server_type="vllm",
            )
        ]
    )

    tokenizer = AutoTokenizer.from_pretrained("./base_model/checkpoint-168")

    messages = [
        {"role": "system", "content": "You are a helpful assistant that plays chess"},
        {
            "role": "user",
            "content": "What is the best move for white in the following position? [FEN string here]",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    async with server.managed_server(tokenizer=tokenizer) as managed:
        completions = await managed.completion(
            prompt=prompt,
            n=1,
            max_tokens=1024 * 4,  # Chess reasoning can be long, but keep bounded
            temperature=1.0,
        )

        # state = managed.get_state()
        # nodes = state["nodes"]

    rprint(completions.model_dump())


if __name__ == "__main__":
    asyncio.run(test())

# test_executor.py
import asyncio
import json
from app.services.executors.executor import _execute_agent_async

async def test_run():
    execution_id = 106  # Replace with actual execution ID
    agent_id = 2      # Replace with actual agent ID
    platform = "reddit"  # Replace with actual platform
    
    result = await _execute_agent_async(execution_id, agent_id, platform)

    # save result to file
    with open("result.json", "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    asyncio.run(test_run())
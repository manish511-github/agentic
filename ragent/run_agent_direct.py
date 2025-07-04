#!/usr/bin/env python3
"""
Direct command-line script to run agent execution without Celery

Usage:
    python run_agent_direct.py <execution_id> <agent_id>

Example:
    python run_agent_direct.py 123 456
"""

import sys
import os
import asyncio
from typing import Dict, Any

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

async def run_agent_direct(execution_id: int, agent_id: int) -> Dict[str, Any]:
    """
    Run the agent directly using the async execution function.
    
    Args:
        execution_id: The execution ID to process
        agent_id: The agent ID to execute
        
    Returns:
        Dict[str, Any]: Execution results
    """
    try:
        # Import the async execution function
        from app.services.executors.executor import _execute_agent_async
        
        # Get the agent platform first
        from app.services.executors.executor import _get_execution_and_agent
        
        execution, agent = _get_execution_and_agent(execution_id, agent_id)
        
        if not execution or not agent:
            return {"error": "Failed to retrieve execution or agent data"}
        
        if not agent.agent_platform:
            return {"error": f"Agent platform not specified for agent {agent_id}"}
        
        # Run the async execution
        result = await _execute_agent_async(execution_id, agent_id, agent.agent_platform)
        
        return result
        
    except Exception as e:
        return {"error": f"Failed to run agent: {str(e)}"}

async def main():
    """Main async function to handle command-line arguments and run the agent"""
    
    # Check command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python run_agent_direct.py <execution_id> <agent_id>")
        print("Example: python run_agent_direct.py 123 456")
        sys.exit(1)
    
    try:
        execution_id = int(sys.argv[1])
        agent_id = int(sys.argv[2])
    except ValueError:
        print("Error: execution_id and agent_id must be integers")
        sys.exit(1)
    
    print(f"🚀 Running agent directly execution_id={execution_id}, agent_id={agent_id}")
    print("=" * 60)
    
    # Run the agent
    result = await run_agent_direct(execution_id, agent_id)
    
    # Display results
    print("\n📊 Execution Results:")
    print("=" * 60)
    
    if result.get("error"):
        print(f"❌ Error: {result['error']}")
        sys.exit(1)
    else:
        print("✅ Agent execution completed successfully")
        
        # Print result details
        for key, value in result.items():
            if isinstance(value, (list, dict)):
                print(f"📋 {key}: {type(value).__name__} with {len(value)} items")
            else:
                print(f"📋 {key}: {value}")
    
    print("\n🎉 Done!")

if __name__ == "__main__":
    asyncio.run(main()) 
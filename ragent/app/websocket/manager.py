from fastapi import WebSocket
from typing import Dict, List
import json
import structlog
import asyncio
from datetime import datetime

logger = structlog.get_logger()

class ConnectionManager:
    def __init__(self):
        # Store active connections
        self.active_connections: Dict[int, List[WebSocket]] = {}  # user_id -> [websockets]
        self.connection_times: Dict[int, datetime] = {}  # Track connection times
        self.ping_interval = 30  # seconds
        self.ping_task = None
    
    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
        self.connection_times[id(websocket)] = datetime.now()
        logger.info("WebSocket connected", user_id=user_id)
        
        # Start ping task if not running
        if not self.ping_task:
            self.ping_task = asyncio.create_task(self._ping_connections())
    
    def disconnect(self, websocket: WebSocket, user_id: int):
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
                if id(websocket) in self.connection_times:
                    del self.connection_times[id(websocket)]
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        logger.info("WebSocket disconnected", user_id=user_id)
    
    async def _ping_connections(self):
        """Send periodic pings to keep connections alive"""
        while True:
            try:
                for user_id, connections in self.active_connections.items():
                    for connection in connections[:]:  # Copy list to avoid modification during iteration
                        try:
                            await connection.send_json({"type": "ping", "timestamp": datetime.now().isoformat()})
                        except Exception as e:
                            logger.error("Error sending ping", error=str(e), user_id=user_id)
                            self.disconnect(connection, user_id)
                
                await asyncio.sleep(self.ping_interval)
            except Exception as e:
                logger.error("Error in ping task", error=str(e))
                await asyncio.sleep(self.ping_interval)
    
    async def broadcast_to_user(self, user_id: int, message: dict):
        """Broadcast message to all connections of a specific user"""
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id][:]:  # Copy list to avoid modification during iteration
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error("Error sending WebSocket message", error=str(e), user_id=user_id)
                    self.disconnect(connection, user_id)
    
    async def broadcast_agent_update(self, user_id: int, agent_id: int, data: dict):
        """Broadcast agent update to user's connections"""
        message = {
            "type": "agent_update",
            "agent_id": agent_id,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast_to_user(user_id, message)
    
    async def broadcast_agent_status(self, user_id: int, agent_id: int, status: dict):
        """Broadcast agent status update to user's connections"""
        message = {
            "type": "agent_status",
            "agent_id": agent_id,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast_to_user(user_id, message)

# Create global connection manager instance
manager = ConnectionManager() 
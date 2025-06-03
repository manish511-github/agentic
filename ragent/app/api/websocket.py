from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from app.websocket.manager import manager
from app.auth import get_current_user_ws
from app.models import UserModel
import structlog
import json

router = APIRouter()
logger = structlog.get_logger()

@router.websocket("/ws/agents")
async def websocket_endpoint(
    websocket: WebSocket,
    current_user: UserModel = Depends(get_current_user_ws)
):
    if not current_user:
        logger.error("WebSocket connection rejected: Invalid authentication")
        return

    try:
        await manager.connect(websocket, current_user.id)
        logger.info("WebSocket connected", user_id=current_user.id)
        
        # Send initial connection success message
        await websocket.send_json({
            "type": "connection_status",
            "status": "connected",
            "user_id": current_user.id
        })
        
        while True:
            try:
                # Keep connection alive and handle any incoming messages
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    # Handle any incoming messages here if needed
                    logger.info("Received WebSocket message", message=message)
                except json.JSONDecodeError:
                    logger.error("Invalid JSON message received")
                    
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected normally", user_id=current_user.id)
                break
            except Exception as e:
                logger.error("WebSocket error", error=str(e), user_id=current_user.id)
                # Try to send error to client
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Internal server error"
                    })
                except:
                    break
                
    except Exception as e:
        logger.error("WebSocket connection error", error=str(e), user_id=current_user.id)
    finally:
        manager.disconnect(websocket, current_user.id)
        logger.info("WebSocket connection cleaned up", user_id=current_user.id) 
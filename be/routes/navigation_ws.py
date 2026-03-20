from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.navigation_service import NavigationService
from models.navigation_models import ErrorResponse

router = APIRouter(prefix="/api/navigation", tags=["navigation"])

navigation_service = NavigationService()


@router.websocket("/ws")
async def websocket_navigation(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")

            if message_type == "session_start":
                required = ["user_id", "map_id", "start", "goal"]
                for key in required:
                    if key not in message:
                        await websocket.send_json(ErrorResponse(message=f"Missing field {key}").dict())
                        break
                else:
                    start = message["start"]
                    goal = message["goal"]
                    map_id = message["map_id"]
                    session = navigation_service.start_session(map_id=map_id, start=start, goal=goal)
                    await websocket.send_json({
                        "type": "session_started",
                        "session_id": session.session_id,
                        "status": session.status,
                        "initial_path": session.path,
                    })

            elif message_type == "position_frame":
                try:
                    session_id = message.get("session_id")
                    position = message.get("position")
                    if not session_id or not position:
                        await websocket.send_json(ErrorResponse(message="session_id and position required").dict())
                        continue

                    result = navigation_service.update_position(session_id, position)
                    await websocket.send_json({"type": "position_update", **result})

                    if result.get("replan"):
                        await websocket.send_json({
                            "type": "replan",
                            "session_id": session_id,
                            "new_path": result.get("new_path"),
                            "reason": "deviation",
                        })

                    if result.get("arrival"):
                        await websocket.send_json({"type": "session_completed", "session_id": session_id})

                except KeyError as e:
                    await websocket.send_json(ErrorResponse(message=str(e)).dict())
                except Exception as e:
                    await websocket.send_json(ErrorResponse(message=f"Internal error: {e}").dict())

            elif message_type == "session_end":
                session_id = message.get("session_id")
                if session_id:
                    navigation_service.close_session(session_id)
                    await websocket.send_json({"type": "session_ended", "session_id": session_id})
                else:
                    await websocket.send_json(ErrorResponse(message="session_id required").dict())

            else:
                await websocket.send_json(ErrorResponse(message=f"Unknown message type: {message_type}").dict())

    except WebSocketDisconnect:
        return
    except Exception as e:
        await websocket.send_json(ErrorResponse(message=f"Connection error: {e}").dict())
        await websocket.close()

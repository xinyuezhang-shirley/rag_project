from fastapi import APIRouter, Request

from app.schemas.common import EchoRequest
from app.services.echo import echo_message

router = APIRouter(tags=["echo"])

@router.post("/echo")
async def echo(request_body: EchoRequest, request: Request):
    result = echo_message(request_body.message)
    request_id = getattr(request.state, "request_id", "")

    return {
        "data": result,
        "request_id": request_id,
    }
# to include websockets in the future service level does not need to change because all it is doing it to 
# validate message, so http and websocket are not the same entrance
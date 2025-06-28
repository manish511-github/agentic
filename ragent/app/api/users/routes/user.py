from fastapi import APIRouter, Depends,status,BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import JSONResponse

from app.api.users.responses.user import UserResponse
from app.api.users.schemas.users import RegisterUserRequest,VerifyUserRequest
from app.api.users.services import user
from app.database import get_sync_db

user_router = APIRouter(
    prefix="/users",
    tags=["Users"],
    responses={404: {"description": "Not found"}},
)
@user_router.post("", status_code=status.HTTP_201_CREATED, response_model=UserResponse)
async def register_user(data: RegisterUserRequest, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_sync_db)):
    # This function registers a new user and sends a verification email in the background
    return await user.create_user_account(data, db, background_tasks)

@user_router.post("/verify", status_code=status.HTTP_200_OK)
async def verify_user_account(data: VerifyUserRequest, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_sync_db)):
    await user.activate_user_account(data, db, background_tasks)
    return JSONResponse({"message": "Account is activated successfully."})
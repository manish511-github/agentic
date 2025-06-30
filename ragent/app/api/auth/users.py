from fastapi import APIRouter, Depends,status,BackgroundTasks, Header
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm


from app.auth.schemas.users import EmailRequest, ResetRequest, UserResponse, RegisterUserRequest, VerifyUserRequest, LoginResponse
from app.auth.services import user
from app.database import get_sync_db
from app.auth.security import oauth2_scheme, get_current_user
guest_router = APIRouter(
    prefix="/auth",
    tags=["Auth"],
    responses={404: {"description": "Not found"}},
)
user_router = APIRouter(
    prefix="/users",
    tags=["Users"],
    responses={404: {"description": "Not found"}},

)
auth_router = APIRouter(
    prefix="/users",
    tags=["Users"],
    responses={404: {"description": "Not found"}},
    dependencies=[Depends(oauth2_scheme), Depends(get_current_user)]
)
@user_router.post("", status_code=status.HTTP_201_CREATED, response_model=UserResponse)
async def register_user(data: RegisterUserRequest, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_sync_db)):
    # This function registers a new user and sends a verification email in the background
    return await user.create_user_account(data, db, background_tasks)

@user_router.post("/verify", status_code=status.HTTP_200_OK)
async def verify_user_account(data: VerifyUserRequest, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_sync_db)):
    await user.activate_user_account(data, db, background_tasks)
    return JSONResponse({"message": "Account is activated successfully."})

@guest_router.post("/login", status_code=status.HTTP_200_OK, response_model=LoginResponse)
async def user_login(data: OAuth2PasswordRequestForm = Depends(),  db: AsyncSession = Depends(get_sync_db)):
    return await user.get_login_token(data, db)

@guest_router.post("/refresh", status_code=status.HTTP_200_OK, response_model=LoginResponse)
async def refresh_token(refresh_token = Header(), db: AsyncSession = Depends(get_sync_db)):
    return await user.get_refresh_token(refresh_token, db)

@guest_router.post("/forgot-password", status_code=status.HTTP_200_OK)
async def forgot_password(data: EmailRequest, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_sync_db)):
    await user.email_forgot_password_link(data, background_tasks, db)
    return JSONResponse({"message": "A email with password reset link has been sent to you."})

@guest_router.put("/reset-password", status_code=status.HTTP_200_OK)
async def reset_password(data: ResetRequest,background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_sync_db)):
    await user.reset_user_password(data,background_tasks, db)
    return JSONResponse({"message": "Your password has been updated."})

@auth_router.get("/me", status_code=status.HTTP_200_OK, response_model=UserResponse)
async def fetch_user(user = Depends(get_current_user)):
    return user

@auth_router.get("/{pk}", status_code=status.HTTP_200_OK, response_model=UserResponse)
async def get_user_info(pk, db: AsyncSession = Depends(get_sync_db)):
    return await user.fetch_user_detail(pk, db)
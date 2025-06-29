from fastapi import BackgroundTasks
from app.settings import get_settings
from app.models import UserModel
from app.auth.security import hash_password
from app.core.email.config import send_email
from app.core.email.context.auth.auth_context import USER_VERIFY_ACCOUNT, FORGOT_PASSWORD

settings = get_settings()


async def send_account_verification_email(user: UserModel, background_tasks: BackgroundTasks):
    """
    Sends an email to the user for account verification.

    This function generates a unique token based on the user's context string and sends an email to the user with a verification link.
    The verification link includes the token and the user's email. The email is sent in the background using BackgroundTasks.

    Args:
        user (UserModel): The user model instance.
        background_tasks (BackgroundTasks): An instance of BackgroundTasks to handle the email sending in the background.
    """
    
    # Generate a unique context string for the user verification process
    string_context = user.get_context_string(context=USER_VERIFY_ACCOUNT)
    # Hash the context string to create a unique token for verification
    token = hash_password(string_context)
    # Construct the activation URL with the token and user's email
    activate_url = f"{settings.FRONTEND_HOST}/auth/account-verify?token={token}&email={user.email}"
    # Prepare the email data with app name, user name, and activation URL
    data = {
        'app_name': settings.APP_NAME,
        "name": user.username,
        'activate_url': activate_url
    }
    # Set the email subject with the app name
    subject = f"Account Verification - {settings.APP_NAME}"
    # Send the verification email in the background
    await send_email(
        recipients=[user.email],
        subject=subject,
        template_name="user/account-verification.html",
        context=data,
        background_tasks=background_tasks
    )
    
    
async def send_account_activation_confirmation_email(user: UserModel, background_tasks: BackgroundTasks):
    data = {
        'app_name': settings.APP_NAME,
        "name": user.username,
        'login_url': f'{settings.FRONTEND_HOST}'
    }
    subject = f"Welcome - {settings.APP_NAME}"
    await send_email(
        recipients=[user.email],
        subject=subject,
        template_name="user/account-verification-confirmation.html",
        context=data,
        background_tasks=background_tasks
    )
    
async def send_password_reset_email(user: UserModel, background_tasks: BackgroundTasks):
    from app.auth.security import hash_password
    string_context = user.get_context_string(context=FORGOT_PASSWORD)
    token = hash_password(string_context)
    reset_url = f"{settings.FRONTEND_HOST}/reset-password?token={token}&email={user.email}"
    data = {
        'app_name': settings.APP_NAME,
        "name": user.username,
        'activate_url': reset_url,
    }
    subject = f"Reset Password - {settings.APP_NAME}"
    await send_email(
        recipients=[user.email],
        subject=subject,
        template_name="user/password-reset.html",
        context=data,
        background_tasks=background_tasks
    )

async def send_password_updated_email(user: UserModel, background_tasks: BackgroundTasks):
    data = {
        'app_name': settings.APP_NAME,
        'name': user.username,
    }
    subject = f"Password Updated - {settings.APP_NAME}"
    await send_email(
        recipients=[user.email],
        subject=subject,
        template_name="user/password-updated.html",
        context=data,
        background_tasks=background_tasks
    )
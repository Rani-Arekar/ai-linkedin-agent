"""FastAPI OAuth endpoints that keep tokens out of responses."""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from database.database import SessionLocal
from services.linkedin_api_service import LinkedInAPIError, LinkedInAPIService
from services.linkedin_oauth_service import LinkedInOAuthService, OAuthError

router = APIRouter(prefix="/auth/linkedin", tags=["linkedin-auth"])
oauth_service = LinkedInOAuthService()
api_service = LinkedInAPIService()


@router.get("/connect")
def connect_linkedin() -> RedirectResponse:
    """Redirect the member to LinkedIn's official authorization endpoint."""

    try:
        return RedirectResponse(oauth_service.get_authorization_url())
    except OAuthError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.get("/callback")
def linkedin_callback(
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
) -> dict[str, str]:
    """Validate callback state, exchange code, retrieve identity, and store locally."""

    if error:
        raise HTTPException(status_code=400, detail="LinkedIn authorization was denied")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")
    try:
        token = oauth_service.exchange_code_for_token(code, state)
        user = api_service.get_current_user(token.access_token)
        with SessionLocal() as session:
            oauth_service.store_token(session, token, account_id=user.id)
        return {"status": "connected"}
    except (OAuthError, LinkedInAPIError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
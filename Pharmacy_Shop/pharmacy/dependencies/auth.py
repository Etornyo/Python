from typing import Annotated



from fastapi import Depends, status
from fastapi.exceptions import HTTPException
from jose import jwt, JWTError

from pharmacy.database.models.users import User
from pharmacy.database.models.Admin import Admin

from pharmacy.dependencies.database import Database
from pharmacy.dependencies.oauth_scheme import user_scheme, admin_scheme

# Token exception
token_exception =HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials"
)
def get_authenticated_user(db: Database, token: str = Depends(user_scheme))->User:

    try:
        data: dict[str,str] = jwt.decode(
            token=token, Key="December", algorithms=["HS256"]
        )

    except JWTError:
        raise token_exception

    user_id = int(data["sub"])

    user:User | None = db.get(User,user_id)

    if user is None:
        raise token_exception

    return user



def get_authenticated_admin(db: Database, token: str = Depends(admin_scheme))->Admin:

    try:
        data: dict[str,str] = jwt.decode(
            token=token, Key="Zekhz", algorithms=["HS256"]
        )

    except JWTError:
        raise token_exception

    admin_id = int(data["sub"])

    admin:Admin| None = db.get(User,admin_id)

    if admin is None:
        raise token_exception


    return admin


AuthenticatedUser = Annotated[User,Depends(get_authenticated_user)]
AuthenticatedAdmin = Annotated[Admin,Depends(get_authenticated_admin)]


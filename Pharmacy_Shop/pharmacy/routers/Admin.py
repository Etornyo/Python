# Admin will enter and edit using this file
# contains routers

import sqlalchemy.exc
from fastapi import APIRouter, status, Depends
from fastapi.exceptions import HTTPException
from sqlalchemy import select

# Local library call
from pharmacy.schemas.Admin import AdminCreate, AdminSchema
from pharmacy.database.models.Admin import Admin
from pharmacy.dependencies.database import AnnotatedAdmin, Database
from pharmacy.dependencies.auth import AuthenticatedAdmin, get_authenticated_admin
# Security
from pharmacy.security import get_hash, password_match_hash
from pharmacy.dependencies.jwt import create_token
from fastapi.security import OAuth2PasswordRequestForm
# from jose import jwt
from pharmacy.schemas.token import Token

# Enabling us to separate the tabs
router = APIRouter(prefix="/admins", tags=["Admin"])


@router.post("/", response_model=AdminSchema)
def create_admin(
        admin_data: AdminCreate,
        db: Database
) -> Admin:
    admin_data.password = get_hash(admin_data.password)
    admin = Admin(**admin_data.model_dump())

    try:
        db.add(admin)
        db.commit()
        db.refresh(admin)

        return admin

    except sqlalchemy.exc.IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="already exists"
        )


@router.get("/", response_model=list[AdminSchema], dependencies=[Depends(get_authenticated_admin)])
def get_admin_list(db: Database) -> list[Admin]:
    return db.scalars(select(Admin)).all()


# Admin Authentication
@router.post("/{authenticate}", response_model=Token, dependencies=[Depends(get_authenticated_admin)])
def login_for_admin_authentication_token(
        db: Database,
        credentials: OAuth2PasswordRequestForm = Depends()):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect Username or password"
    )
    admin: Admin | None = db.scalar(
        select(Admin).where(Admin.username == credentials.username)
    )

    if admin is None:
        raise credentials_exception

    if not password_match_hash(plain=credentials.password, hashed=admin.password):
        raise credentials_exception
    data = {"sub": str(admin.id)}

    token = create_token(data=data)

    return {"token_type": "bearer", "access_token": token}


@router.get("/current_admin", response_model=AdminSchema, dependencies=[Depends(get_authenticated_admin)])
def get_current_admin(admin: AuthenticatedAdmin) -> Admin:
    return admin


@router.get("/{admin_id}", response_model=AdminSchema, dependencies=[Depends(get_authenticated_admin)])
def get_admin(admin: AnnotatedAdmin) -> Admin:
    return admin


'''@router.get("/{admin_id}", response_model=AdminSchema, dependencies=[Depends(get_authenticated_admin)])
def get_admin(admin: AnnotatedAdmin) -> Admin:
    return admin'''


# Delete an Admin
@router.delete("/{admin_id}", dependencies=[Depends(get_authenticated_admin)])
def delete_admin(admin: AnnotatedAdmin, db: Database) -> None:  # Doesn't return anything
    db.delete(admin)
    db.commit()

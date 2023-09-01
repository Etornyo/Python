import fastapi
import sqlalchemy.exc
from fastapi import APIRouter,status
# import route
from fastapi.exceptions import HTTPException
from sqlalchemy import select

# Local library call
from pharmacy.database.core import SessionMaker
from pharmacy.schemas.users import UserCreate, UserSchema
from pharmacy.database.models.users import User

router = fastapi.APIRouter()
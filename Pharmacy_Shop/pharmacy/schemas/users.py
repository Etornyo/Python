from datetime import date
from pydantic import BaseModel,EmailStr

# Has database that doesn't change
class UserBase(BaseModel):
    username: str
    contact: str
    address: str | None
    email: EmailStr | None
    date_of_birth: date | None
    
# We won't send them the password. We inherit those that don't change from the mirror    
class UserCreate(UserBase):
    password: str
    
class UserSchema(UserBase):
    id: int    

#User that mirrors your database. Del this
class User(BaseModel):
    id: int
    username: str
    password: str
    contact: str
    address: str | None
    email: EmailStr | None
    date_of_birth: date | None
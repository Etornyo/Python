from pydantic import BaseModel,EmailStr


class AdminBase (BaseModel):
    username: str
    email: EmailStr


# We won't send them the password. We inherit those that don't change from the mirror
class AdminCreate (AdminBase):
    password: str

# id always in schema
class AdminSchema (AdminBase):
    id: int


# Add and comment the mirror table
class Admin(BaseModel):
    id: int
    username: str
    email: EmailStr
    password: str
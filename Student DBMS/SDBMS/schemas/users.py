from datetime import date
from pydantic import BaseModel,EmailStr

# We can't give the user access to everthing so we will restrict them with the classes we use
class UserBase(BaseModel):

class UserCreate(UserBase):

class UserSchema(UserBase):



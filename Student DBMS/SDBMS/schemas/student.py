from datetime import date
from pydantic import BaseModel, EmailStr

# Contains database that doesn't change
class StudentBase (BaseModel):
    Username: str
    Email: EmailStr | None
    First_name: str
    Middle_name: str
    Surname: str
    Date_of_birth: date | None
    Program_otionID: int


class StudentCreate(StudentBase):
    Password: str

class StudentSchema(StudentBase):
    id:int




class student(BaseModel):
    Id: int
    Username: str
    Email: EmailStr | None
    Password: str
    First_name: str
    Middle_name: str
    Surname: str
    Date_of_birth: date | None
    HouseID: int
    Program_otionID: int

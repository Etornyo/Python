from sqlalchemy import String,Date
from sqlalchemy.orm import Mapped, mapped_column
from SDBMS.Database.core import Base

class User(Base):
    __table__= "users"
    # python rep for data type    =       one for sqlite
    id:Mapped[int]= mapped_column()
    username:Mapped[str]= mapped_column(string)
    email:Mapped[str]= mapped_column(string)
    First_name:Mapped[str]= mapped_column(string)
    Middle_name:Mapped[str]= mapped_column(string)
    Surname_name:Mapped[str]= mapped_column(string)
    password: Mapped[str] = mapped_column(string)
    date_of_birth:Mapped[date]= mapped_column(Date)
    
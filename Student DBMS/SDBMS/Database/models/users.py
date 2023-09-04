from datetime import date
from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from SDBMS.Database.core import Base

class User(Base):
    __table__= "users"
    # python rep for data type    =       one for sqlite
    id:Mapped[int]= mapped_column(primary_key=True)
    username:Mapped[str]= mapped_column(String,nullable=False)
    email:Mapped[str]= mapped_column(String,unique=True,nullable=False)
    First_name:Mapped[str]= mapped_column(String, nullable=False)
    Middle_name:Mapped[str]= mapped_column(String)
    Surname_name:Mapped[str]= mapped_column(String, nullable=False)
    password: Mapped[str] = mapped_column(String, nullable=False)
    date_of_birth:Mapped[date]= mapped_column(nullable=False)
    
from sqlalchemy import String
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column


from SDBMS.Database.core import Base


def Teacher(Base):
    __tablename__ = "teacher"


    id: Mapped[int] = mapped_column(primary_key=True)
    Username: Mapped[str] = mapped_column(String,nullable=False)
    email: Mapped[str] = mapped_column(String, unique=True,nullable=False)
    password: Mapped[str] = mapped_column(String, nullable=False)
    First_Name: Mapped[str] = mapped_column(String, nullable=False)
    Middle_Name: Mapped[str] = mapped_column(String)
    Last_Name: Mapped[str] = mapped_column(String, nullable=False)
    ProgramID: Mapped[int] = mapped_column(ForeignKey(), nullable=False)
    DepartmentID: Mapped[int] = mapped_column(ForeignKey(), nullable=False)
    FormID: Mapped[int] = mapped_column(ForeignKey(), nullable=False)

from sqlalchemy import String,ForeignKey
from sqlalchemy.orm import Mapped, mapped_column



def program_option_has_courseID():
    __tablename__ = "program_option_has_courseID"

    id: Mapped[int] = mapped_column(primary_key=True)
    courseID: Mapped[int] = mapped_column(ForeignKey(), nullable=False)
    programID: Mapped[int] = mapped_column(ForeignKey(), nullable=False)
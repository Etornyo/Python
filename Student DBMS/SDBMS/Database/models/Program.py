from sqlalchemy import String
from sqlalchemy.orm import Mapped,mapped_column

def Program():
    __tablename__ = "program"


    id: Mapped[int] = mapped_column(primary_key=True)
    program_name: Mapped[str] = mapped_column(String,nullable=False)

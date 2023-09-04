from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import Mapped,mapped_column

def Program():
    __tablename__ = "program"


    id: Mapped[int] = mapped_column(primary_key=True)
    General_Science: Mapped[str] = mapped_column(String,nullable=False)

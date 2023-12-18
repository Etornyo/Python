from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

# Local import
from SDBMS.Database.core import Base

class Program_course(Base):
    __tablename__ = "program_course"

    programid: Mapped[int] = mapped_column(ForeignKey(""), nullable=False)
    courseid: Mapped[int] = mapped_column(ForeignKey(""), nullable=False)
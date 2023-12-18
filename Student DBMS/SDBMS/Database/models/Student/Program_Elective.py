from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

# Local import
from SDBMS.Database.core import Base

class Program_course(Base):
    __tablename__ = "program_elective"

    programid: Mapped[int] = mapped_column(ForeignKey(""), nullable=False)
    electiveid: Mapped[int] = mapped_column(ForeignKey(""), nullable=False)
    allowed_count: Mapped[int] = mapped_column()
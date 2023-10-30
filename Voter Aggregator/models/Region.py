
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column


from  import


# Creating an object relational mapper to connect database
class Region (Base):
    __tablename__ = "Region"

    RegionID: Mapped[uuid] = mapped_column(primary_key=True)
    Name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    Code: Mapped[int | None] = mapped_column(String, unique=True,nullable=False)

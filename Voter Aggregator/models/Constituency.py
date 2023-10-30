import uuid

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column


from pharmacy.database.core import Base


# Creating an object relational mapper to connect database
class Constituency (Base):
    __tablename__ = "Constituency"

    ConstituencyID: Mapped[uuid] = mapped_column(primary_key=True)
    Name: Mapped[str] = mapped_column(nullable=False)
    Code: Mapped[int | None] = mapped_column(nullable=False)
    RegionID: Mapped[int] = mapped_column(ForeignKey("Region.RegionID"), nullable=False)


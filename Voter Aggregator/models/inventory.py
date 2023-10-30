from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from pharmacy.database.core import Base
class Inventory(Base):
    __tablename__ = "inventories"

    id: Mapped[int] = mapped_column (primary_key=True)  # In order to identify using Mapped we used the syntax so when changing in postgres it will be okay
    name: Mapped[str] = mapped_column (String, unique=True, nullable=False)
    quantity: Mapped[int] = mapped_column ( nullable=False)  # Added password
    price: Mapped[float] = mapped_column ( nullable=False)
    address: Mapped[str | None] = mapped_column (String)
    email: Mapped[str | None] = mapped_column (String, unique=True)



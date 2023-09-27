
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column


from pharmacy.database.core import Base


# Creating an object relational mapper to connect database
class Admin (Base):
    __tablename__ = "admin"  # Note format for creating the table and how it is connected to Base

    id: Mapped[int] = mapped_column(primary_key=True)  # In order to identify using Mapped we used the syntax so when changing in postgres it will be okay
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    email: Mapped[str | None] = mapped_column(String, unique=True,nullable=False)
    password: Mapped[str] = mapped_column(String, nullable=False)  # Added password



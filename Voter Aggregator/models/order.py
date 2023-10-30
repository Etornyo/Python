from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime,date,time


from pharmacy.database.core import Base



# Creating an object relational mapper to connect database
class Order (Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column (primary_key=True)
    checkout_id: Mapped[int] = mapped_column (ForeignKey("checkout.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"),nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    Date: Mapped[date] = mapped_column(nullable=False)
    time_of_order: Mapped[time] = mapped_column ( nullable=False )

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column


from pharmacy.database.core import Base


# Creating an object relational mapper to connect database
class CartItem (Base):
    __tablename__ = "cart_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    quantity: Mapped[int | None] = mapped_column(nullable=False)
    inventory_id: Mapped[int] = mapped_column(ForeignKey("inventories.id"), nullable=False)


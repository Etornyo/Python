from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column


from pharmacy.database.core import Base


# Creating an object relational mapper to connect database
class Checkout(Base):
    __tablename__ = "checkout"

    id: Mapped[int] = mapped_column(primary_key=True)
    cart_item_id: Mapped[int] = mapped_column(ForeignKey("cart_items.id"), nullable=False)
    sub_total: Mapped[float | None] = mapped_column(nullable=False)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), nullable=False)  # Added password


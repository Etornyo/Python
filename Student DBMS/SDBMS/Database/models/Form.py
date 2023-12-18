from sqlalchemy import String,ForeignKey
from sqlalchemy.orm import Mapped,mapped_column



def House():
    __tablename__ = "house"

    id: Mapped[int] = mapped_column ( primary_key=True )
    class_name: Mapped[str] = mapped_column(String, nullable=False)
    Form_master: Mapped[int] = mapped_column(ForeignKey(""), nullable=False)

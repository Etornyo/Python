from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped,mapped_column



def House():
    __tablename__ = "house"


    id: Mapped[int] = mapped_column(primary_key=True)
    House_master: Mapped[int] = mapped_column(ForeignKey(), nullable=False)
    Assit_master: Mapped[int] = mapped_column(ForeignKey(), nullable=False)
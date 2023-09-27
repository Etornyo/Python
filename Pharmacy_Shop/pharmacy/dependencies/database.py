from typing import Annotated

from fastapi import status,Depends
from fastapi.exceptions import HTTPException
from sqlalchemy.orm import Session


from pharmacy.database.core import SessionMaker
from pharmacy.database.models.users import User
from pharmacy.database.models.Admin import Admin
from pharmacy.database.models.inventory import Inventory
from pharmacy.database.models.cart_items import CartItem


def database_connect():
    with SessionMaker() as connect:
        yield connect

# Database connection used as database
Database=Annotated[Session,Depends(database_connect)]

# For Users
def get_user_or_404(db: Database, user_id:int ) -> User:
    user: User | None = db.get(User, user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            details="User Not Found"
        )

    return user

# For Admins
def get_admin_or_404(db: Database, admin_id:int ) -> Admin:
    admin: Admin | None = db.get(Admin, admin_id)

    if admin is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            details="User Not Found"
        )

    return admin


# For Inventory
def get_inventory_or_400(db: Database, inventory_id:int ) -> Inventory:
    inventory: Inventory | None = db.get(Inventory, inventory_id)

    if inventory is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            details="Inventory Already Exist"
        )
    return inventory

# For Cart_items
def get_cart_item_or_404(db:Database, cart_item_id:int)-> CartItem:
    cart_item: CartItem | None = db.get(CartItem, cart_item_id)

    if cart_item is None:
        raise HTTPException(
            status= status.HTTP_404_NOT_FOUND,
            detail="Item Not Found"
        )


AnnotatedUser = Annotated[User, Depends(get_user_or_404)]
AnnotatedAdmin = Annotated[Admin, Depends(get_admin_or_404)]
AnnotatedInventory = Annotated[Inventory, Depends(get_inventory_or_400)]
AnnotatedCartItem = Annotated[CartItem, Depends(get_cart_item_or_404)]
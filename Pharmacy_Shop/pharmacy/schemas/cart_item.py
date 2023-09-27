from pydantic import BaseModel


class CartItemBAse (BaseModel):
    quantity: int
    inventory_id: int


class CartItemCreate (CartItemBAse):
    pass


class CartItemSchema (CartItemBAse):
    id: int
    user_id:int


# Moirror
class CartItem ( BaseModel ):
    id: int
    users_id: int
    quantity: int
    inventory_id: int

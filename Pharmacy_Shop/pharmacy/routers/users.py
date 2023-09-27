# User will enter and edit using this file
# contains routers

import sqlalchemy.exc
from fastapi import APIRouter, status, Depends
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.exceptions import HTTPException
from sqlalchemy import select


# Local library call
from pharmacy.database.models.checkout import Checkout
from pharmacy.Enums import OrderStatus
from pharmacy.schemas.order import OrderSchema
from pharmacy.schemas.users import UserCreate, UserSchema
from  pharmacy.schemas.Admin import AdminSchema
from pharmacy.schemas.cart_item import CartItemSchema, CartItemCreate
from pharmacy.schemas.order import OrderItem, OrderSchema
from pharmacy.database.models.users import User
from pharmacy.database.models.order import Order
from pharmacy.database.models.cart_items import CartItem
from pharmacy.database.models.inventory import Inventory
from pharmacy.dependencies.database import AnnotatedUser, AnnotatedCartItem, Database, get_inventory_or_400
from pharmacy.dependencies.auth import AuthenticatedUser, get_authenticated_admin
# from pharmacy.dependencies.auth import AuthenticatedAdmin
#  Security
from pharmacy.security import get_hash, password_match_hash
from pharmacy.dependencies.jwt import create_token
# from jose import jwt
from pharmacy.schemas.token import Token


# Enabling us to seperate the tabs
router = APIRouter(prefix="/users", tags=["User"])


@router.post("/", response_model=UserSchema)  # in bracete specify the path to use
# We aren't doing anything asynchronous here
def create_user(
        user_data: UserCreate,
        db: Database
)-> User:
    user_data.password = get_hash(user_data.password)
    user = User(**user_data.model_dump())
        # username=user_data.username,
        # password=user_data.password,
        # contact=user_data.contact,
        # address=user_data.address,
        # email=user_data.email,
        # date_of_birth=user_data.date_of_birth)

    # username: str = Body(embed=True),
    # password: str = Body(embed=True),
    # contact: str = Body(embed=True),
    # email: str = Body(embed=True),
    # date_of_birth: str = Body(embed=True)

    # Database connection 
    # with SessionMaker() as db:
    try:
        db.add(user)
        db.commit()
        db.refresh(user)

        return user

    except sqlalchemy.exc.IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="already exists"
        )



@router.get("/", response_model=list[UserSchema], dependencies=[Depends(get_authenticated_admin)])
def get_user_list(db:Database) -> list[User]:
    return db.scalars(select(User)).all()

# Authentication
@router.post("/{authenticate}", response_model=Token)
def login_for_authentication_token(
        db: Database,
        credentials: OAuth2PasswordRequestForm = Depends()
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect Username or password"
    )
    user: User | None = db.scalar(
        select(User).where(User.username == credentials.username)
    )

    if user is None:
        raise credentials_exception

    if not password_match_hash(plain=credentials.password,hashed=user.password):
        raise credentials_exception

    data = {"sub": str (user.id)}

    token = create_token(data=data)

    return {"token_type": "bearer","access_token":token}

@router.get("/current", response_model=UserSchema)
def get_current_user(user:AuthenticatedUser)->User:
    return user



# Cart
@router.post("/current/cart-item", response_model=CartItemSchema)
def  add_to_cartitem(
        user: AnnotatedUser,
        cart_item_data: CartItemCreate,
        db: Database,
)-> CartItem: #check the 404
    inventory = get_inventory_or_400(db=db,inventory_id=cart_item_data.inventory_id)


    if cart_item_data.quantity > inventory.quantity:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Quantiy is more than stock"
        )
    cart_item = CartItem(**cart_item_data.model_dump(),user_id=user.id)

    db.add(cart_item)
    db.commit()
    db.refresh(cart_item)

    return cart_item



@router.get("/current/cart_item", response_model=list[CartItemSchema])
def  get_cart_item_list(
        user: AnnotatedUser,
        cart_item_data: CartItemCreate,
        db: Database,
)->list[CartItemSchema]:
    return db.scalars ( select ( CartItem ).where(CartItem.user_id==user.id)).all ()


# Order
@router.post("/current/orders")
def place_order(user: AnnotatedUser, db: Database)->None:
    cart_items = db.scalars(select(CartItem).where(CartItem.user_id ==user.id.all()))

    if not cart_items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail= "Cannot Place an Order if cart is empty"
        )

    order =  Order(status=OrderStatus.PENDING.value,user_id=user.id)
    db.add(order)
    db.commit()
    db.refresh(order)

    checkout: list[Checkout]=[]

    for cart_item in cart_items:
        inventory: Inventory | None = db.get(Inventory, cart_item.inventory_id)

        if inventory is None:
            continue

        if cart_item.quantity> inventory.quantity:
            db.delete(order)
            db.commit()

            raise HTTPException(
                status_code= status.HTTP_400_BAD_REQUEST,
                detail=f"Not enough items in stock for {inventory.name}"
                        f"Tried to order: {cart_item.quantity}"
                        f"In stock: {inventory.name}"
            )

        checkouts = Checkout (
            order_id=order.id,
            cart_item_id = cart_item.id,
            sub_total=cart_item.quantity * inventory.price
        )

        checkouts.append(checkout)
        inventory.quantity -= cart_item.quantity

    db.add_all(checkouts)
    db.commit()



# Order
@router.get("/current/orders", response_model=list[OrderSchema])
def get_order_list(db:Database, user: AuthenticatedUser):
    db_orders = db.scalars(select(Order).where(Order.user_id == user.id)).all()

    orders: list[OrderSchema] = []

    for db_order in db_orders:
        order = OrderSchema(id=db_order.id, status=db_order.status)

        checkouts = db.scalars(select(Checkout).where(Checkout.order_id == db_order.id)).all()

        for checkout in checkouts:
            order.total += checkout.sub_total

            cart_item = db.scalar(
                select(CartItem).where(CartItem.id == checkout.cart_item_id)
            )

            inventory = db.scalar(
                select(Inventory).where(Inventory.id == cart_item.inventory_id)
            )

            order_item = OrderItem(
                quantity=cart_item.quantity,
                name=inventory.name,
                sub_total=checkout.sub_total
            )

            order.order_items.append(order_item)

        order.user = user
        orders.append(order)

    return orders


# Cart
@router.delete("/current/cart_item/{cart_item_id}")
def delete_cart_item(user: AnnotatedCartItem, db: Database, cart_item: AnnotatedCartItem)->None:
    if cart_item.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorised"
        )
    db.delete(cart_item)
    db.commit()



            
@router.get("/{user_id}", response_model = UserSchema, dependencies=[Depends(get_authenticated_admin)])
def get_user(user: AnnotatedUser)->User:
    return user

# Delete a user
@router.delete("/{user_id}", response_model=list[AdminSchema], dependencies=[Depends(get_authenticated_admin)])
def delete_user(user:AnnotatedUser,db: Database)->None: # Doesn't return anything
        db.delete(user)
        db.commit()

# User will enter and edit using this file
# contains routers

import sqlalchemy.exc
from fastapi import APIRouter, status
from fastapi.exceptions import HTTPException
from sqlalchemy import select

# Local library call
from pharmacy.schemas.inventory import InventoryCreate, InventorySchema
from pharmacy.database.models.inventory import Inventory
from pharmacy.dependencies.database import AnnotatedInventory, Database

# Enabling us to separate the tabs
router = APIRouter(prefix="/inventory", tags=["Inventory"])


@router.post("/", response_model=InventorySchema)  # in bracket specify the path to use
# We aren't doing anything asynchronous here
def create_inventory(inventory_data: InventoryCreate, db: Database,):
    inventory = Inventory(**inventory_data.model_dump())

    try:
        db.add(inventory)
        db.commit()
        db.refresh(inventory)

        return inventory

    except sqlalchemy.exc.IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="already exists"
        )


@router.get("/", response_model=list[InventorySchema])
def get_inventor_list(db: Database) -> list[Inventory]:
    return db.scalars(select(Inventory)).all()


@router.get("/inventories/{inventory_id}", response_model=InventorySchema)
def get_inventory(inventory: AnnotatedInventory):
    return inventory


# Delete a user
@router.delete("/inventories/{inventory_id}")
def delete_inventory(inventory: AnnotatedInventory, db: Database):
    db.delete(inventory)
    db.commit()

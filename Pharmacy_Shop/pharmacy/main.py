from contextlib import asynccontextmanager

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware

# importing from files in directory
from pharmacy.database.core import Base, SessionMaker, engine
from pharmacy.routers import users,inventory,Admin


# This block automatically close the session. It's a damn manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Base.metadata.drop_all(bind=engine)# Comment after creating a user
    Base.metadata.create_all (bind=engine)

    yield


# Using rules on function arguments the first in the fast function is the argument and the second after the '=' is the value
app = FastAPI (lifespan=lifespan)
app.include_router ( users.router )
app.include_router ( inventory.router )
app.include_router ( Admin.router )

origins = {
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "http://localhost:63342",

}


app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

"""@app.get ("/ping")
def ping_pong() -> dict[str, str]:
    return {"message": "pong"}


@app.get ("/name/{first_name}")  # curly braces around making it a variable
def get_first_name(first_name: str) -> dict[str, str]:
    return {"name": "first_name"}


@app.post ("/name")
def get_surname(surname: str = Body ()) -> dict[str, str]:  # will appear in the body in the page
    return {"name": "surname"}

# @app.post("/Fullname")
# def get_full_name(Full_name: str = Body()) -> dict[str, str]:
#     return{"Fullname": Full_name.split(' ')}
# app.post("/join_name")
# def get_full_name(name: str = Body()) -> dict[str, str]:
#    return{"join_name": name.replace('-')}'''
"""
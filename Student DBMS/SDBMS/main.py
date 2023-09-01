# For instance
from fastapi import FastAPI, Body
from contextlib import  asynccontextmanager

app = FastAPI()
# app.include_router(router)

from SDBMS.Database.core import Base, SessionMaker, engine
from SDBMS.routers.users import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    with SessionMaker as Session:
        Base.metadata.create_all(bind=engine)

# Sending to the API using '.get'. Receive from API using '.post'.
@app.get("/ping")
def ping_pong()->dict[str,str]:
    return {"message" : "Hello there"}

@app.post("/Fullname")
def get_name(Full_name: str = Body) -> dict[str,str]:
    print("What is you name in full?")
    return {"Fullname": Full_name.split}
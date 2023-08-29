from sqlalchemy import create_engine
from sqlalchemy import sessionmaker

engine = create_engine("sqlite://:memory:",echo = True,connect_args = {})


SessionMaker = sessionmaker(bind = engine)


with SessionMaker.begin() as Session:
    ...


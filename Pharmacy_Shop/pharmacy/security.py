from passlib.context import CryptContext


password_contex = CryptContext(schemes=["bcrypt"])


def get_hash(password: str)->str:
    return password_contex.hash(secret=password)


def password_match_hash(plain: str,hashed: str)->bool:
    return password_contex.verify(secret=plain, hash=hashed)
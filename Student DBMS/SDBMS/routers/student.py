import sqlalchemy.exc
from fastapi import APIRouter, status, Depends
from fastapi.exceptions import HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select

# Local Library
from SDBMS.schemas.student import StudentCreate, StudentSchema


router = APIRouter(prefix="/student", tags=["Student"])

@router.post("/", response_model=StudentSchema)
def create_student(
        student_data: StudentCreate,
        db: Database
) -> Student:
    student_data.password = get_hash ( student_data.password )
    student = Student ( **student_data.model_dump () )

    try:
        db.add ( student )
        db.commit ()
        db.refresh ( student)

        return student

    except sqlalchemy.exc.IntegrityError:
        db.rollback ()
        raise HTTPException (
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="already exists"
        )



@router.get()
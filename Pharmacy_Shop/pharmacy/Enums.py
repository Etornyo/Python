from enum import Enum


class OrderStatus(str, Enum):
    # User interface = Database Interface
    PENDING = "pending"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
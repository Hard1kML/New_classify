from pydantic import BaseModel

# Схема данных запроса

class TextRequest(BaseModel):
    text: str
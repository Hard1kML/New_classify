import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Используем переменную окружения database_url или дефолтное значение
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://classifier_user:DataML12@localhost:5432/comclassify"
)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    comment = Column(String, nullable=False)
    predicted_class = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# Создать таблицы(вызвать один раз при старте)
def init_db():
    Base.metadata.create_all(bind=engine)

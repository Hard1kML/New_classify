import os
from pathlib import Path
# Константы для путей к моделям и меткам классов
LR_MODEL_PATH = "src/backend/models/lr_adv.pkl"  # получаем значение от 0 до 3
TFIDF_PATH = "src/backend/models/tf_idf_adv.pkl"  # получаем фичи(вектор)



# Константы для путей к моделям и меткам классов NN
MODEL_PATH = Path(r"C:\Users\ser9e\New_classify\src\backend\models\bert")
MODEL_PATH_POSIX = MODEL_PATH.resolve().as_posix()  # корректный путь для Transformers 

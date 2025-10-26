import os 
import streamlit as st 
import requests
import logging

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #e0f7fa, #ffffff);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# настройка логирования 
logging.basicConfig(level=logging.INFO)
logging.info('запуск Streamlit приложения')

#Получаем url бэкенда из переменной окружения 
BACKEND_URL = os.environ.get('BACKEND_URL', 'http://localhost:8000')
logging.info(f'Используем url бэкенда: {BACKEND_URL}')
st.markdown("<h1 style='color: black;'>Классификация текста</h1>", unsafe_allow_html=True)
text=st.text_area('Введите текст для классификации:')
if st.button('Классифицировать'):
    if text.strip(): 
        logging.info(f'Получен текст для классификации: {text}') 
         # Отправка POST запроса к FastAPI серверу
        logging.info(f'Отправка запроса на сервер...')
        response=requests.post(f'{BACKEND_URL}/predict/', json={'text': text})
        if response.ok:
            logging.info('получен ответ от сервера')
            result = response.json() 

            predicted_class = result.get('predicted_class', 'не определено')

            # Словарь цветов для эмоций — подкорректируй под свои классы
            colors = {
                "позитивный": "#FFD700",
                "негативный": "#FF4500",
                "не определено": "#32CD32",
                "мусор": "#808080",
            } 
            color = colors  .get(predicted_class.lower(), "#000000")

            st.markdown(f"### Результат: <span style='color:{color};'>{predicted_class.capitalize()}</span>",unsafe_allow_html=True) 
        else:
            logging.error(f'Ошибка при обращении к серверу: {response.status_code}')
            st.error('Ошибка при обращении к серверу.')
    else:
        st.markdown(
    "<p style='color: black; background-color: #fff3cd; padding: 10px; border-radius: 5px;'>Пожалуйста, введите текст.</p>",
    unsafe_allow_html=True
)        

import streamlit as st
from print_func import *
from ts_model import *

def intro():
    select = st.sidebar.selectbox("Вибір розділу", ["Про задачу", "Про команду"])
    if select == "Про задачу":
        task_info()
        
    elif select == "Про команду":
        print_command_review()


def about_models(select_action):

    if select_action == "На економічних параметрах":
        print_about_economic_features_model()

    elif select_action == "Часові ряди":
        print_about_time_series_model()

def models():
    select = st.sidebar.selectbox("Вибір розділу", ["Отримати прогноз", "Інтерпретація моделі"])
    if select == "Отримати прогноз":
        select_predict()
        
    elif select == "Інтерпретація моделі":
        select_interp()

def select_info_model():
    about_col()
    steps_work()
        
def select_predict():
    intro_model()
    predict_size = select_predict_size()
    select, sеlect_step = select_type_learning(predict_size)
    value = st.button('Отримати прогноз')
    if value:
            y_test, y_train, preds, mae, X_test, X_train, model = work_model(predict_size, select, sеlect_step)
            print_res(y_test, preds, mae, X_test, predict_size)
            predict_plot(X_train[-4*predict_size:], X_test, preds, y_test)
            
def select_interp():
    intro_shap()
    predict_size = select_predict_size()
    if predict_size>=3:
        with st.spinner('Навчаємо та тренуємо модель...'):
            y_test, y_train, preds, mae, X_test, X_train, model = work_model(predict_size, "Звичайне навчання", 0)
        shap_plots(model, X_train, y_train)    

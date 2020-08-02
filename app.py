import streamlit as st
import numpy as np
import pandas as pd
from print_func import *
from utils import *

def main():
    st.sidebar.title("Що бажаєте переглянути:")
    app_mode = st.sidebar.radio("Перейти до", ["Вступ", "Підходи до вирішення задачі",
                                          "Модель та результати"])
    
    if app_mode == "Вступ":
        print_intro()
        intro()
            
    elif app_mode == "Підходи до вирішення задачі":
        st.title("Наші підходи")
        select_solution = st.sidebar.selectbox("Оберіть розділ", ["Вибрати", "На економічних параметрах", 
                                                                "Часові ряди"])
        about_models(select_solution)
        
    elif app_mode == "Модель та результати":
        st.title("Результати роботи")
        models()
    

main()
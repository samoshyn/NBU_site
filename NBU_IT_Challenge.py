import streamlit as st
import numpy as np
import pandas as pd
from print_func import *
from utils import *

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}    
    """
st.markdown(hide_footer_style, unsafe_allow_html=True) 

def main():
    st.sidebar.title("Що бажаєте переглянути:")
    app_mode = st.sidebar.radio("Перейти до", ["Вступ", "Підходи до вирішення задачі",
                                          "Модель та результати"])
    
    if app_mode == "Вступ":
        print_intro()
        intro()
            
    elif app_mode == "Підходи до вирішення задачі":
        st.title("Наші підходи")
        select_solution = st.sidebar.selectbox("Оберіть розділ", ["На економічних параметрах", 
                                                                "Часові ряди"])
        about_models(select_solution)
        
    elif app_mode == "Модель та результати":
        st.title("Результати роботи")
        models()
    

main()

import streamlit as st
import pickle

# Load the model and data
model = pickle.load(open("PriceCar-Predict.sav", "rb"))

st.title('Car Price Prediction')
st.write('Enter the following features to predict the price of the car :')

CarName= st.number_input('Car Company')
wheelbase= st.number_input('Wheelbase')
curbweight = st.number_input('Curb Weigh')
enginesize = st.number_input('Engine Size')
boreratio = st.number_input('Bore Ratio')
horsepower = st.number_input('Horse Power')
carlength = st.number_input('Car Lenght')
carwidth = st.number_input('Car Widht')
price = st.number_input('Price')
 
 
 
 

predict = ''

if st.button('Price Predict'):
    predict = model.predict(
        [[ wheelbase, curbweight, enginesize, boreratio, horsepower, carlength, carwidth, price]]
    )
    st.write (' harga mobil dalam USD : ', predict)
    st.write (' harga mobil dalam IDR (Juta) :', predict*19000)

import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st

def load_model2():
    with open('dtc.pkl', 'rb') as file:
      modelcar2 = pk.load(file)
    return modelcar2

# Call the function to load the model
model = load_model2()

st.header('Car Price Prediction ML Model')

cars_data = pd.read_excel(r"E:\Cardheko\Carsdata1.xlsx")


Fuel = st.selectbox('Fuel', cars_data['Fuel'].unique())
KmDriven = st.selectbox('No of kms Driven', cars_data['KmDriven'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
Seats = st.selectbox('No of Seats', cars_data['Seats'].unique())
Owner = st.selectbox('owner', cars_data['Owner'].unique())
modelYear = st.selectbox('Car Manufactured Year', cars_data['modelYear'].unique())
Engine = st.selectbox('Engine CC', cars_data['Engine'].unique())
Mileage = st.selectbox('Car Mileage', cars_data['Mileage'].unique())
Bhp = st.selectbox('Max Power', cars_data['Bhp'].unique())
Color = st.selectbox('Color', cars_data['Color'].unique())
Brand = st.selectbox('Select Car Brand', cars_data['Brand'].unique())
# Body_Type = st.selectbox('Body type', cars_data['Body Type'].unique())
# model = st.selectbox('Model', cars_data['model'].unique())
# variantname = st.selectbox('Variant Name', cars_data['variantName'].unique)
# Insurance_Validity = st.selectbox('Insurance Validity', cars_data['Insurance Validity'].unique())





if st.button("Predict"):
    cars = pd.DataFrame(
    [[Fuel,KmDriven,transmission,Seats,Owner,modelYear,Engine,Mileage,Bhp,Color,Brand]],
    columns=["Fuel","KmDriven","transmission","Seats","Owner","modelYear","Engine","Mileage","Bhp","Color","Brand"])
    
    cars['Fuel'].replace(['Petrol', 'Diesel', 'Lpg', 'Cng', 'Electric'],[1,2,3,4,5],inplace=True)
    cars['transmission'].replace(['Manual', 'Automatic'],[1,2],inplace=True)
    # cars['Body Type'].replace(['Hatchback', 'SUV', 'Sedan', 'MUV', 'Coupe', 'Minivans',
    #     'Pickup Trucks', 'Convertibles', 'Hybrids', 'Wagon'],[1,2,3,4,5,6,7,8,9,10],inplace=True)
    cars['Brand'].replace(['Maruti', 'Ford', 'Tata', 'Hyundai', 'Jeep', 'Datsun', 'Honda',
       'Mahindra', 'Mercedes-Benz', 'BMW', 'Renault', 'Audi', 'Toyota',
       'Mini', 'Kia', 'Skoda', 'Volkswagen', 'Volvo', 'MG', 'Nissan',
       'Fiat', 'Mitsubishi', 'Jaguar', 'Land', 'Chevrolet', 'Citroen',
       'OpelCorsa', 'Isuzu', 'Lexus', 'Porsche', 'Hindustan',
       'Ambassador'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],inplace=True)
    cars['Color'].replace(['White', 'Red', 'Grey', 'Maroon', 'Orange', 'Silver', 'Blue',
       'Brown', 'Yellow', 'Black', 'Golden', 'Green', 'Purple', 'Gold',
       'TITANIUM GREY', 'Violet', 'MODERN STEEL METALLIC',
       'PLATINUM WHITE', 'Golden Brown', 'Aurora Black Pearl', 'Beige',
       'Star Dust', 'Flash Red', 'PLATINUM WHITE PEARL', 'Wine Red',
       'Taffeta White', 'Minimal Grey', 'Fiery Red', 'T Wine',
       'Prime Star Gaze'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],inplace=True)
    # cars['Insurance Validity'].replace(['Third Party insurance', 'Comprehensive', 'Third Party',
    #    'Zero Dep', '2', '1', 'Not Available'],
    #                       [3,8,3,9,2,1,0]
    #                       ,inplace=True)

    car_price = model.predict(cars)

    st.markdown('Car Price is going to be '+ str(car_price[0]))

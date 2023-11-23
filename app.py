import pandas as pd
import streamlit as st
import joblib
import pickle as pk

def preprocess_data(df):

    df['cylinders'] = df['cylinders'].apply(lambda x : x.replace(' cylinders',''))
    df['manufacturer'] = df['manufacturer'].apply(lambda x: str(x).lower())
    
    tf = pd.read_csv('nef.csv',index_col=0)
    df = pd.get_dummies(df)

    ef = df.reindex(tf.columns, fill_value=0,axis=1)

    return ef

st.header('Car Prediction dataset')

st.text('Want to find out how much your used vehicle will cost but cant decide on a price ? \n\nFill in the details of your vehicle and find out how much will it sell')

manufacturer = ['GMC', 'Chevrolet', 'Toyota', 'Ford', 'Jeep', 'Nissan', 'Ram', 'Mazda', 
'Cadillac', 'Honda', 'Dodge', 'Lexus', 'Jaguar', 'Buick', 'Chrysler', 'Volvo', 'Audi', 
'Infiniti', 'Lincoln', 'Alfa-Romeo', 'Subaru', 'Acura', 'Hyundai', 'Mercedes-benz', 'BMW', 
'Mitsubishi', 'Volkswagen', 'Porsche', 'KIA', 'Rover', 'Ferrari', 'Mini', 'Pontiac', 'Fiat', 
'Tesla', 'Saturn', 'Mercury', 'Harley-Davidson', 'Datsun', 'Aston-martin', 'Land rover', 'Morgan']

# manufacturer = [comp.lower() for comp in manufacturer]

quality = [x for x in range(1,6)]

cylinders = ['8 cylinders', '6 cylinders', '4 cylinders', '5 cylinders',
       '3 cylinders', '10 cylinders', '12 cylinders']

fuel = ['Gas', 'Other', 'Diesel', 'Hybrid', 'Electric']

drive = ['RWD', '4WD', 'FWD']

transmission = ['Other','Automatic','Manual']

size = ['Full-Size','Mid-Size','Compact','Sub-Compact']

type = ['Pickup', 'Truck', 'Other', 'Coupe', 'SUV', 'Hatchback',
       'Mini-van', 'Sedan', 'Offroad', 'Bus', 'Van', 'Convertible',
       'Wagon']

paint_color = ['White', 'Blue', 'Red', 'Black', 'Silver', 'Grey', 'Brown',
       'Yellow', 'Orange', 'Green', 'Custom', 'Purple']

title_status = ['Clean', 'Rebuilt', 'Lien', 'Salvage', 'Missing',
       'Parts Only']

year = [x for x in range(1900,2023)]

manufacturer_data = st.selectbox('Pick manufacturer',manufacturer,)
quality_data = st.select_slider('On a scale of 1 to 5, how good is your car ?',quality)
cylinders_data = st.selectbox('How many cylinders does your car have ?',cylinders)
fuel_data = st.selectbox('What fuel type does your car use ?', fuel)
drive_data = st.radio('What gear does your car use ?',drive)
transmission_data = st.radio('Auto or manual ?',transmission)
size_data = st.radio('What size is your car ?',size)
type_data = st.selectbox('What type of vehicle is it ?',type)
paint = st.selectbox('What is the color of your car ?',paint_color)
titl = st.selectbox('What is the condition of your car ?',title_status)
year_data = st.selectbox('What year is your car made in ?',year)
odometer = st.number_input('Enter odometer values between 10000 and 3000000',min_value=10000, max_value=3000000)

cars_dict = {

        'manufacturer':[manufacturer_data],
        'condition':[quality_data],
        'cylinders':[cylinders_data],
        'fuel':[fuel_data],
        'drive':[drive_data],
        'transmission':[transmission_data],
        'size':[size_data],
        'type':[type_data],
        'paint':[paint],
        'title_status':[titl],
        'year':[year_data],
        'odometer':[odometer]
}


get_price = st.button('Click to get your price')

if get_price:
              cars_items = cars_dict.items()
              cars_list = list(cars_items)
              cars_df = pd.DataFrame.from_dict(cars_dict)

              for columns in cars_df.columns:
                     if cars_df[columns].dtypes == 'O':
                            cars_df[columns] = cars_df[columns].apply(lambda x : str(x).lower())

              cars_df = preprocess_data(cars_df)


              with open('carstd.sav','rb') as f:
                     standard_scaler = pk.load(f)

              cars_df = standard_scaler.transform(cars_df)

              with open('carpred.pkl','rb') as f:
                     load_model = joblib.load(f)

              ypr = load_model.predict(cars_df)
              cost = 'Congratulations, your ' + type_data + ' from ' +  manufacturer_data + ' would have a price tag of ${:0.2f}.'.format(ypr[0])
              # print(cost)
              st.success(cost)
              # print(ypr)
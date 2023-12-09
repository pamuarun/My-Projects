from tkinter import *
from tkinter import messagebox as mb
import requests
from PIL import Image
from datetime import datetime

def get_weather():
    global city
    city = city_input.get()
    api_key = 'YOUR_API_KEY'  # Replace with your OpenWeatherMap API key
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid=6b8d0772165f4ff918bdcb38b54a2da3&units=metric'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        pressure = data['main']['pressure']
        wind = data['wind']['speed'] * 3.6  # Convert from m/s to km/h
        epoch_time = data['dt']
        date_time = datetime.fromtimestamp(epoch_time)
        desc = data['weather'][0]['description']
        cloudy = data['clouds']['all']

        # Update the timelabel with the current date and time
        timelabel.config(text=f"Time: {date_time.strftime('%H:%M:%S')}\nDate: {date_time.strftime('%Y-%m-%d')}")

        temp_field.insert(0, '{:.1f} °C'.format(temp))
        pressure_field.insert(0, str(pressure) + "hPa")
        humid_field.insert(0, str(humidity) + "%")
        wind_field.insert(0, '{:.2f} km/h'.format(wind))
        cloud_field.insert(0, str(cloudy) + " % ")
        desc_field.insert(0, str(desc))
    else:
        mb.showerror("Error", "City Not Found. Enter a valid City")
        city_input.delete(0, END)

def reset_fields():
    temp_field.delete(0, END)
    pressure_field.delete(0, END)
    humid_field.delete(0, END)
    wind_field.delete(0, END)
    cloud_field.delete(0, END)
    desc_field.delete(0, END)
    forecast_field.delete(1.0, END)
    city_input.delete(0, END)
    timelabel.config(text='')

def get_forecast():
    global city

    # Replace 'YOUR_API_KEY' with your actual OpenWeatherMap API key
    api_key = 'YOUR_API_KEY'

    # Fetch the weather forecast using OpenWeatherMap API
    url = f'http://api.openweathermap.org/data/2.5/forecast?q={city}&appid=6b8d0772165f4ff918bdcb38b54a2da3&units=metric'
    response = requests.get(url)

    if response.status_code == 200:
        forecast_data = response.json()

        # Clear any previous data in the forecast field
        forecast_field.delete(1.0, END)

        # Extract relevant forecast information
        for forecast in forecast_data['list']:
            timestamp = forecast['dt']
            date_time = datetime.fromtimestamp(timestamp)
            temp = forecast['main']['temp']
            desc = forecast['weather'][0]['description']

            # Display forecast information in the forecast_field
            forecast_field.insert(END,
                                  f"{date_time.strftime('%Y-%m-%d %H:%M:%S')} - Temperature: {temp}°C, Description: {desc}\n")
    else:
        mb.showerror("Error", "Failed to fetch weather forecast")

root = Tk()
root.title('Weather Application')
root.configure(bg='royal blue1')
root.geometry("900x600")

title = Label(root, text='Weather Detection and Forecast', fg='yellow', bg='royal blue1', font=('bold', 15))
label1 = Label(root, text='Enter the city name : ', font=('bold', 12), bg='royal blue1')
city_input = Entry(root, width=24, fg='red2', font=12, relief=GROOVE)
timelabel = Label(root, text='', bg='royal blue1', font=('bold', 14), fg='yellow')

btn_submit = Button(root, text='Get Weather', width=10, font=12, bg='lime green', command=get_weather)
btn_forecast = Button(root, text='Weather Forecast', width=14, font=12, bg='lime green', command=get_forecast)
btn_reset = Button(root, text='Reset', font=12, bg='lime green', command=reset_fields)

label2 = Label(root, text="Temperature :", font=('bold', 12), bg='royal blue1')
label3 = Label(root, text="Pressure : ", font=('bold', 12), bg='royal blue1')
label4 = Label(root, text="Humidity :", font=('bold', 12), bg='royal blue1')
label5 = Label(root, text="Wind :", font=('bold', 12), bg='royal blue1')
label6 = Label(root, text="Cloudiness:", font=('bold', 12), bg='royal blue1')
label7 = Label(root, text="Description :", font=('bold', 12), bg='royal blue1')

temp_field = Entry(root, width=24, font=11)
pressure_field = Entry(root, width=24, font=11)
humid_field = Entry(root, width=24, font=11)
wind_field = Entry(root, width=24, font=11)
cloud_field = Entry(root, width=24, font=11)
desc_field = Entry(root, width=24, font=11)

title.grid(row=0, column=0, columnspan=3, pady=10)
label1.grid(row=1, column=0, padx=5, pady=5)
city_input.grid(row=1, column=1, padx=5, pady=5)
btn_submit.grid(row=2, column=1, pady=5)  # Placing Get Weather button under the Enter City textbox
btn_forecast.grid(row=2, column=2, pady=5)  # Placing Forecast button to the right of Get Weather button

label2.grid(row=3, column=0, padx=5, pady=5, sticky='w')
temp_field.grid(row=3, column=1, padx=5, pady=5)

label3.grid(row=4, column=0, padx=5, pady=5, sticky='w')
pressure_field.grid(row=4, column=1, padx=5, pady=5)

label4.grid(row=5, column=0, padx=5, pady=5, sticky='w')
humid_field.grid(row=5, column=1, padx=5, pady=5)

label5.grid(row=6, column=0, padx=5, pady=5, sticky='w')
wind_field.grid(row=6, column=1, padx=5, pady=5)

label6.grid(row=7, column=0, padx=5, pady=5, sticky='w')
cloud_field.grid(row=7, column=1, padx=5, pady=5)

label7.grid(row=8, column=0, padx=5, pady=5, sticky='w')
desc_field.grid(row=8, column=1, padx=5, pady=5)

btn_reset.grid(row=9, column=1, pady=5)

# Add a Text widget for displaying the weather forecast
forecast_field = Text(root, width=40, height=10, font=11, wrap=WORD)
forecast_field.grid(row=3, column=2, rowspan=7, padx=10, pady=10, sticky='n')

root.mainloop()

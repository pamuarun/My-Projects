from googletrans import Translator
from gtts import gTTS
from playsound import playsound
import time
import os
import speech_recognition as sr
import pyttsx3
import datetime as dt
import random
import requests
import pywhatkit as pk
import wikipedia as wiki
from plyer import notification
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

# Function to get the latest headlines
def get_latest_headlines(api_key, country="us"):
    base_url = "https://newsapi.org/v2/top-headlines"
    params = {"country": country, "apiKey": api_key}

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        news_data = response.json()
        articles = news_data.get("articles", [])

        if not articles:
            print("No headlines found.")
            return

        print("Latest Headlines:")
        for index, article in enumerate(articles, start=1):
            title = article.get("title", "No Title")
            print(f"{index}. {title}")

    except requests.exceptions.RequestException as e:
        print(f"Error getting headlines: {e}")

listener = sr.Recognizer()
speaker = pyttsx3.init()
translator = Translator()

# RATE
rate = speaker.getProperty('rate')
speaker.setProperty('rate', 160)

# VOICE
voices = speaker.getProperty('voices')
speaker.setProperty('voice', voices[1].id)

# Replace 'your_email@gmail.com' and 'your_app_password' with your actual Gmail credentials
sender_email = 'arunteja962@gmail.com'
password = 'nrdc brix tztk jtau'  # Use the app password generated from your Google Account

# Function to speak text
def speak(text):
    text_without_semicolon = text.replace(':', '')
    speaker.say('Yes Boss,' + text_without_semicolon)
    speaker.runAndWait()

# Function to speak extended text
def speak_ex(text):
    speaker.say(text)
    speaker.runAndWait()

# Function to take voice command
def take_command():
    command = ''
    try:
        with sr.Microphone() as source:
            print('Listening...')
            voice = listener.listen(source, timeout=5)
            command = listener.recognize_google(voice)
            command = command.lower()
            print('User said:', command)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    return command

# Function to send email
def send_email(subject, body):
    recipient_email = input("Enter the recipient's email address: ")

    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = subject

    message.attach(MIMEText(body, 'plain'))

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, recipient_email, message.as_string())

    print(f"Email sent to {recipient_email}")

# Function to get weather details
def get_weather(city):
    try:
        units = 'metric'
        api_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&units={units}&appid=6b8d0772165f4ff918bdcb38b54a2da3"
        response = requests.get(api_url)
        weather_data = response.json()

        temperature = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        pressure = weather_data['main']['pressure']
        wind_speed = weather_data['wind']['speed']
        min_temp = weather_data['main']['temp_min']
        max_temp = weather_data['main']['temp_max']
        weather_condition = weather_data['weather'][0]['description']

        weather_report = (
            f"The current weather in {city} is {temperature} degrees Celsius, "
            f"with {humidity}% humidity, {pressure} hPa pressure, and {wind_speed} m/s wind speed. "
            f"The minimum temperature is {min_temp} degrees Celsius, and the maximum temperature is {max_temp} degrees Celsius. "
            f"The weather condition is {weather_condition}."
        )

        print(weather_report)
        speak(weather_report)

    except Exception as e:
        print(e)
        speak("Sorry, I couldn't fetch the weather information at the moment.")

# Function to set reminder
def set_reminder(reminder_text, time_in_hours, minute_input, am_pm_input, phone_number):
    now = dt.datetime.now()
    reminder_time = now.replace(hour=int(time_in_hours), minute=int(minute_input), second=0, microsecond=0)

    if am_pm_input.lower() == 'pm' and reminder_time.hour < 12:
        reminder_time = reminder_time.replace(hour=reminder_time.hour + 12)

    time_difference = (reminder_time - now).seconds
    notification_title = 'Reminder'
    notification_text = reminder_text

    notification.notify(
        title=notification_title,
        message=notification_text,
        timeout=time_difference,
        toast=True
    )

    try:
        pk.sendwhatmsg(phone_number, reminder_text, reminder_time.hour, reminder_time.minute + 1)
        print(f"Reminder set for {reminder_time.hour}:{reminder_time.minute} and message sent.")
    except Exception as e:
        print(e)
        print("Reminder set, but there was an issue sending the message. Please check the message length.")

# Change va_name to "Alexa"
va_name = 'Alexa'
speak(f'I am your {va_name}, Tell me boss.')

# Main loop
while True:
    user_command = take_command()
    print(user_command)

    if 'close' in user_command:
        print('Yes boss see you again. I will be there whenever you call me.')
        speak('See you again . I will be there whenever you call me.')
        break

    elif 'time' in user_command:
        cur_time = dt.datetime.now().strftime('%I:%M %p')  # Correct time format
        print(cur_time)
        speak(cur_time)

    elif 'play' in user_command:
        user_command = user_command.replace('play ', '')
        print('Playing ' + user_command)
        speak('Playing ' + user_command + ', enjoy.')
        pk.playonyt(user_command)

    elif 'search for' in user_command or 'google' in user_command:
        speak('Searching for ' + user_command)
        pk.search(user_command)

    elif 'who is' in user_command:
        user_command = user_command.replace('who is', '')
        info = wiki.summary(user_command, 2)
        print(info)
        speak(info)

    elif 'who are you' in user_command:
        speak_ex(f'I am your {va_name}, Tell me boss.')

    elif 'set reminder' in user_command:
        try:
            speak("What should I remind you about?")
            reminder_text = take_command()

            if not reminder_text:
                speak("Please provide a reminder text.")
                continue

            speak("In how many hours?")
            duration_input = input("Enter the number of hours: ")

            try:
                time_in_hours = float(duration_input)
                if time_in_hours > 0:
                    speak("At what minute?")
                    minute_input = input("Enter the minute: ")

                    speak("AM or PM?")
                    am_pm_input = input("Enter AM or PM: ")

                    speak("Please provide your phone number with country code.")
                    phone_number = input("Enter your phone number: ")

                    set_reminder(reminder_text, time_in_hours, float(minute_input), am_pm_input, phone_number)

                else:
                    speak("Please provide a positive duration.")
            except ValueError:
                speak("Invalid input. Please provide a numeric duration in hours.")

        except Exception as e:
            print(e)
            speak("Sorry, I am not able to set this reminder.")

    elif 'weather' in user_command:
        speak("Sure, please enter the city name.")
        city_name = input("Enter the city: ")
        get_weather(city_name)

    elif 'send email' in user_command:
        speak("What should be the subject of the email?")
        email_subject = take_command()

        speak("Please type the body of the email.")
        email_body = input("Type your email body: ")

        send_email(email_subject, email_body)

    elif 'general knowledge' in user_command:
        questions_and_answers = {
            "Who is the president of the United States?": "Joe Biden",
            "What is the capital of France?": "Paris",
        }

        speak("Sure, I will ask you a general knowledge question. Are you ready?")
        user_response = take_command()

        if 'yes' in user_response:
            question = random.choice(list(questions_and_answers.keys()))
            speak(question)
            user_answer = take_command()

            correct_answer = questions_and_answers.get(question, "I don't know the answer.")
            if user_answer.lower() == correct_answer.lower():
                speak("Correct! Well done.")
            else:
                speak(f"Sorry, the correct answer is {correct_answer}.")
        else:
            speak("Alright, let me know when you are ready.")

    elif 'get news' in user_command:
        news_api_key = '955e1fa659d240e6932b41c286a7c746'  # Replace with your actual News API key
        if not news_api_key:
            speak("Please set the NEWS_API_KEY environment variable.")
        else:
            get_latest_headlines(news_api_key)

    else:
        speak_ex('Please say it again boss')

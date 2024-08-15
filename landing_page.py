#This is the main page (so far) for the Gradio interface. 
#It will be the first page that the user sees when they run the program.
###############
import gradio as gr
from gradio_calendar import Calendar
import datetime
import yfinance as yf
import pandas as pd


async def update(ticker1, ticker2, inp_start_date, inp_end_date):
    tickers = [ticker1, ticker2]
    data = yf.download(tickers, inp_start_date, inp_end_date, auto_adjust=True)['Close']
    #await data
    return data

demo = gr.Interface(
    fn=update,
    inputs=[gr.Textbox(placeholder="Enter the first stock ticker")
            ,gr.Textbox(placeholder="Enter the second stock ticker")
            ,Calendar(type="datetime", label="Select a start date", info="Click the calendar icon to bring up the calendar.", render=False)
            ,Calendar(type="datetime", label="Select an end date", info="Click the calendar icon to bring up the calendar.", render=False)
            ],
    outputs=["dataframe"])

demo.launch()
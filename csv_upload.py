import gradio as gr
import pandas as pd

def csv_upload(file):
    df = pd.read_csv(file)
    return df

iface = gr.Interface(fn=csv_upload, inputs="file", outputs="dataframe")
iface.launch()
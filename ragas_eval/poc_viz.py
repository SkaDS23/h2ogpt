import gradio as gr
from pymongo import MongoClient
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


client = MongoClient('mongodb://localhost:27017/')
db = client['h2ogpt']
collection = db['PostEval']

def fetch_model_names_eval():
    documents = collection.find({})
    
    first_keys = []
    
    for doc in documents:
        keys = [key for key in doc.keys() if key != '_id']
        if keys:
            first_keys.append(keys[0])  
        
    return first_keys
def fetch_data_dashboard(model:str):
    document = collection.find({model: {"$exists": True}})
    
    metrics_to_include=['faithfulness', 'context_recall', 'context_precision', 'answer_similarity', 'answer_correctness', 'answer_relevancy', 'harmfulness']
    formatted_data = []
    
    for doc in document:
        for first_key, nested_keys in doc.items():
            if first_key != '_id':
                for line_key, metrics in nested_keys.items():
                    filtered_metrics = {key: metrics[key] for key in metrics_to_include if key in metrics}
                    formatted_data.append(filtered_metrics)

    df = pd.DataFrame(formatted_data)
    pre_processed_df = fill_nan_with_mean(df)
    return pre_processed_df

def fetch_data_dashboard2(model:str):
    document = collection.find({model: {"$exists": True}})
    
    metrics_to_include=['faithfulness', 'context_recall', 'context_precision', 'answer_similarity', 'answer_correctness', 'answer_relevancy', 'harmfulness']
    formatted_data = []
    
    for doc in document:
        for first_key, nested_keys in doc.items():
            if first_key != '_id':
                for line_key, metrics in nested_keys.items():
                    filtered_metrics = {key: metrics[key] for key in metrics_to_include if key in metrics}
                    formatted_data.append(filtered_metrics)

    df = pd.DataFrame(list(formatted_data))
    pre_processed_df = fill_nan_with_mean(df)
    return pre_processed_df

def clear_data():
    # Returns an empty DataFrame with the same headers
    return pd.DataFrame(columns=['faithfulness', 'context_recall', 'context_precision', 'answer_similarity', 'answer_correctness', 'answer_relevancy', 'harmfulness'])

def fill_nan_with_mean(df):
    means = df.mean().round(2)
    df_filled = df.fillna(means)
    
    return df_filled
def visualize_metrics_linechart2(df):
    if df.empty:
        print("The DataFrame is empty. No visualizations can be made.")
        fig = go.Figure()
        return fig

    fig = go.Figure()

    for i, column in enumerate(df.columns):
        visible = True if i == 0 else 'legendonly'
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df[column], 
            mode='markers+lines', 
            name=column,
            visible=visible
        ))

    fig.update_layout(
        title={
            'text': "Evaluation Metrics Trend",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Row Index",
        yaxis_title="Value",
        legend=dict(x=1.1, y=0.5, title="Metrics", font=dict(size=10))
    )

    fig.update_xaxes(tickvals=list(range(len(df))))

    return fig

def visualize_metrics_boxplot(df):
    if df.empty:
        print("The DataFrame is empty. No visualizations can be made.")
        fig = go.Figure()
        return fig

    fig = go.Figure()

    for column in df.columns:
        fig.add_trace(go.Box(y=df[column], name=column, boxmean='sd'))

    y_min = df.min().min() - 1 
    y_max = df.max().max() + 1 
    
    fig.update_layout(
        title={
            'text': "Metrics Box Plot",
            'x': 0.5,
            'xanchor': 'center'
        },
        yaxis_title="Value",
        xaxis_title="Metrics",
        showlegend=False,
        xaxis=dict(
            range=[-1, len(df.columns)],  
            tickvals=list(range(len(df.columns))),
            ticktext=df.columns
        ),
        yaxis=dict(
            range=[y_min, y_max]  
        )
    )

    return fig

def visualize_metrics_violin(df):
    fig = px.violin(df, box=True, points="all", title="Violin Plot of Metrics")
    
    fig.update_layout(
        title={
            'text': "Violin Plot of Metrics",
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def visualize_metrics_bar_chart(df):
    fig = go.Figure()
    
    means = df.mean()
    fig.add_trace(go.Bar(x=means.index, y=means.values, name="Mean"))
    
    fig.update_layout(
        title={
            'text': "Bar Chart of Mean Metrics",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Metrics",
        yaxis_title="Mean Value"
    )
    
    return fig

def visualize_metrics_radar(df):
    categories = df.columns.tolist()
    mean_values = df.mean().tolist()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=mean_values,
        theta=categories,
        fill='toself',
        name='Metrics'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=False
            )),
        showlegend=False,
        title={
            'text': "Radar Chart of Mean Metrics",
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def visualize_metrics_density(df):
    if df.empty:
        print("The DataFrame is empty. No visualizations can be made.")
        fig = go.Figure()
        return fig
    
    fig = go.Figure()

    for column in df.columns:
        fig.add_trace(go.Violin(x=df[column], name=column, box_visible=True, meanline_visible=True))

    fig.update_layout(
        title={
            'text': "Density Plot of Metrics",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Metrics",
        yaxis_title="Density",
        legend=dict(
            x=1.02,  
            y=0.5,   
            xanchor='left',  
            yanchor='middle',  
            font=dict(size=10)
        )
    )

    return fig

def update_plot(selected_model):
    data_to_visualize = fetch_data_dashboard(selected_model)  
    
    LinePlot = visualize_metrics_linechart2(data_to_visualize)
    BoxPlot = visualize_metrics_boxplot(data_to_visualize)
    RadarChart = visualize_metrics_radar(data_to_visualize)
    BarChart = visualize_metrics_density(data_to_visualize)
    return LinePlot, BoxPlot, RadarChart, BarChart


def reset_dashboard():
    return gr.update(value=None, visible=True), gr.update(value=None, visible=True), gr.update(value=None, visible=True), gr.update(value=None, visible=True)
    
    
custom_html = """
<div id="chatbot-container">
    <div id="chat-window">
        <div id="output"></div>
    </div>
    <input id="input" type="text" placeholder="Type a message..." />
    <button id="send-button">Send</button>
</div>

<style>
#chatbot-container {
    width: 100%;
    height: 300px;
    display: flex;
    flex-direction: column;
    align-items: stretch;
}
#chat-window {
    flex: 1;
    border: 1px solid #ccc;
    padding: 10px;
    overflow-y: auto;
}
#output {
    white-space: pre-wrap;
}
#input {
    border: 1px solid #ccc;
    padding: 5px;
    margin-top: 10px;
    flex: 0 0 auto;
}
#send-button {
    margin-top: 10px;
    padding: 5px 10px;
}
</style>
"""

custom_js = """
function initializeChatbot() {
    document.getElementById('send-button').onclick = function() {
        var input = document.getElementById('input').value;
        if (input.trim() === '') return;

        var outputDiv = document.getElementById('output');
        var newMessage = document.createElement('div');
        newMessage.textContent = 'You: ' + input;
        outputDiv.appendChild(newMessage);

        var botMessage = document.createElement('div');
        botMessage.textContent = 'Bot: ' + 'You said: ' + input;
        outputDiv.appendChild(botMessage);

        document.getElementById('input').value = '';
        outputDiv.scrollTop = outputDiv.scrollHeight;
    };
}
globalThis.initializeChatbot = initializeChatbot;
"""
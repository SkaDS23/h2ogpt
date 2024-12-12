import numpy as np
import matplotlib.pyplot as plt

def radar_chart(df, metrics=None, title="Radar Chart", labels=None):
    """
    Generate a radar chart from a DataFrame with metrics ranging from 0 to 1.

    Parameters:
    - df (DataFrame): DataFrame containing metrics (each metric ranging from 0 to 1).
    - metrics (list): List of metrics to include in the radar chart. If None, all columns of the DataFrame are used.
    - title (str): Title of the radar chart.
    - labels (list): Optional list of labels for the metrics. If None, column names of the DataFrame are used.

    Returns:
    - None
    """
    # If metrics list is not provided, use all columns of the DataFrame
    if metrics is None:
        metrics = df.columns.tolist()

    # If labels list is not provided, use column names of the DataFrame
    if labels is None:
        labels = metrics

    # Number of metrics
    num_metrics = len(metrics)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Define angles for each axis
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()

    # Plot each metric on the radar chart
    for i, metric in enumerate(metrics):
        # Values for this metric
        values = df[metric].tolist()

        # Complete the loop
        values.append(values[0])

        # Plot the radar chart
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=labels[i])

    # Fill the area inside the radar chart
    ax.fill(angles, values, 'b', alpha=0.1)

    # Add labels and title
    ax.set_title(title, size=20, pad=20)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Show the plot
    plt.show()

# Example usage:
if __name__ == "__main__":
    import pandas as pd

    # Sample DataFrame with metrics ranging from 0 to 1
    data = {
        "Metric 1": [0.8, 0.6, 0.7, 0.9],
        "Metric 2": [0.4, 0.5, 0.6, 0.3],
        "Metric 3": [0.7, 0.8, 0.6, 0.5],
        "Metric 4": [0.9, 0.7, 0.8, 0.6],
        "Metric 5": [0.6, 0.5, 0.4, 0.3]
    }

    df = pd.DataFrame(data)

    # Plot radar chart
    radar_chart(df, title="Radar Chart Example", labels=["A", "B", "C", "D", "E"])

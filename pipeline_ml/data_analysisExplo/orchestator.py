import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd

class ExploratoryDataAnalysis:
    def __init__(self, data):
        self.data = data
        # Configuration des couleurs et du style
        self.background_color = 'rgba(50, 50, 50, 1)'  # Dark gray
        self.text_color = 'white'
        self.curve_colors = [
            'rgba(65, 105, 225, 0.8)',  # Blue
            'rgba(244, 164, 96, 0.8)',  # Salmon
            'rgba(205, 92, 92, 0.8)',   # Red
            'rgba(20, 92, 92, 0.8)',    # Green
            'rgba(205, 9, 92, 0.8)'     # Magenta
        ]

    def plot_variables(self, x, y, title_graph) -> None:
        """
        Plots variables over time.
        
        args:
            x : column time,
            y : column varibale,
        returns: show figure
        """
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=x, y=y, fill='tozeroy', name=title_graph,
                       line=dict(color=self.curve_colors[0]))
        )
        fig.update_layout(
            title=title_graph,
            plot_bgcolor=self.background_color,
            paper_bgcolor=self.background_color,
            font=dict(color=self.text_color)
        )
        fig.show()

    def plot_dual_variable_curve(self, x, y1, y2, title_graph, y1_name, y2_name):
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Scatter(x=x, y=y1, name=y1_name,
                       line=dict(color=self.curve_colors[0], width=3)),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=x, y=y2, name=y2_name,
                       line=dict(color=self.curve_colors[1], width=3)),
            secondary_y=True,
        )

        # Add figure title
        fig.update_layout(
            title_text=title_graph,
            plot_bgcolor=self.background_color,
            paper_bgcolor=self.background_color,
            font=dict(color=self.text_color)
        )

        # Set x-axis title
        fig.update_xaxes(title_text="Timestamp")

        # Set y-axes titles
        fig.update_yaxes(title_text=f"<b>Primary</b> {y1_name}", secondary_y=False)
        fig.update_yaxes(title_text=f"<b>Secondary</b> {y2_name}", secondary_y=True)

        fig.show()

    def plot_dual_variable_curve_dynamique(self, x, y1, y2, title_graph, y1_name, y2_name, time_step='5 m', frame_duration=100):
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Scatter(x=[], y=[], name=y1_name,
                    line=dict(color=self.curve_colors[0], width=3)),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=[], y=[], name=y2_name,
                    line=dict(color=self.curve_colors[1], width=3)),
            secondary_y=True,
        )

        # Add figure title and layout updates
        fig.update_layout(
            title_text=title_graph,
            plot_bgcolor=self.background_color,
            paper_bgcolor=self.background_color,
            font=dict(color=self.text_color),
            xaxis=dict(range=[np.min(x), np.min(x)]),  # Initial x-axis range covers no data
            updatemenus=[dict(type="buttons", showactive=True,
                            buttons=[dict(label="Play",
                                            method="animate",
                                            args=[None, {"frame": {"duration": frame_duration, "redraw": True},
                                                        "fromcurrent": True,
                                                        "transition": {"duration": 300, "easing": "linear"}}]),
                                        dict(label="Pause",
                                            method="animate",
                                            args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                            "mode": "immediate",
                                                            "transition": {"duration": 0}}])])]
        )

        # Convert time_step to appropriate timedelta
        split_time = time_step.split(' ')
        if len(split_time) == 2:
            time_value, time_unit = split_time
            time_delta = np.timedelta64(int(time_value), time_unit)
        else:
            raise ValueError("time_step must be in the format '<integer><space><unit>' like '5 m'")

        # Create frames for animation
        frames = [go.Frame(data=[go.Scatter(x=x[:i], y=y1[:i]), go.Scatter(x=x[:i], y=y2[:i])],
                        layout=dict(xaxis=dict(range=[x[max(0, i - int(time_value))], x[i]])))
                for i in range(1, len(x))]

        fig.frames = frames

        # Set x-axis and y-axes titles
        fig.update_xaxes(title_text="Timestamp")
        fig.update_yaxes(title_text=f"<b>Primary</b> {y1_name}", secondary_y=False)
        fig.update_yaxes(title_text=f"<b>Secondary</b> {y2_name}", secondary_y=True)

        fig.show()
    
    def visualize_data(self):
        sns.pairplot(self.data)
        plt.show()

    # def feature_engineering(self):
    #     # Ajout de nouvelles features si nécessaire
    #     self.data['new_feature'] = self.data['Value Bets'] * self.data['Value Players']
        
    def add_cumulative_difference(self):
        self.data = self.data.copy()
        self.data = self.end_round_feature_filtering()
        # Calculer la différence entre 'Value Bets' et 'Value Prize'
        self.data['diff_bets_prize'] = self.data['Value Bets'] - self.data['Value Prize']
        # Cumuler les différences calculées précédemment
        self.data['cumulative_diff'] = self.data['diff_bets_prize'].cumsum()
        return self.data.reset_index(drop=True)
        
    def end_round_feature_filtering(self):
        return self.data[self.data['End of Round'] == True].drop('End of Round',axis=1)
        
        
    
        

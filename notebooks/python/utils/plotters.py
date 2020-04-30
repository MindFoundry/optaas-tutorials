from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display import display
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from plotly import offline as plotly, graph_objs as go, tools

PLOT_TYPES = {'random', 'grid', 'exploration', 'exploitation', 'initialisation', 'default', 'user-defined'}


class ChartPlotter:
    def __init__(self, plot_types_and_colours: Dict, minimise: bool = False, use_3d_plot: bool = False,
                 plot_best_scores_on_left: bool = True):
        assert PLOT_TYPES <= set(plot_types_and_colours.keys())
        self.plot_types_and_colours = plot_types_and_colours
        self.minimise = minimise
        self.use_3d_plot = use_3d_plot
        self.plot_best_scores_on_left = plot_best_scores_on_left
        self._2d_plot_column = 1 if plot_best_scores_on_left else 2
        self._3d_plot_column = 2 if plot_best_scores_on_left else 1

    def start_update(self):
        raise NotImplementedError()

    def update_display(self):
        raise NotImplementedError()

    def plot_best_scores(self, iterations, best_scores):
        raise NotImplementedError()

    def plot_scores_by_type(self, configuration_type: str, colour: str, iterations, scores):
        raise NotImplementedError()

    def plot_3d_scores_by_type(self, configuration_type: str, colour: str, x_values, y_values, z_values):
        raise NotImplementedError()

    def plot_3d_surface(self, X, Y, Z):
        raise NotImplementedError()

    def plot_3d_target(self, target_x, target_y, target_z):
        raise NotImplementedError()


class PlotlyChartPlotter(ChartPlotter):
    def __init__(self, minimise: bool = None, use_3d_plot: bool = None, plot_best_scores_on_left: bool = None):
        plot_types_and_colours = {
            'random': 'red',
            'grid': 'black',
            'exploration': 'orange',
            'initialisation': 'orange',
            'exploitation': 'blue',
            'default': 'cyan',
            'user-defined': 'black'
        }
        super().__init__(plot_types_and_colours, minimise, use_3d_plot, plot_best_scores_on_left)
        plotly.init_notebook_mode(connected=True)
        self.camera = dict(
            # up=dict(x=0, y=0, z=1),
            # center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=0.8)
        )

    def start_update(self):
        is_3d_left = self.use_3d_plot and not self.plot_best_scores_on_left
        is_3d_right = self.use_3d_plot and self.plot_best_scores_on_left
        self.fig = tools.make_subplots(rows=1, cols=2, specs=[[{'is_3d': is_3d_left}, {'is_3d': is_3d_right}]],
                                       print_grid=False)

    def update_display(self):
        plotly.iplot(self.fig, filename='plotly/graphs')

    def plot_best_scores(self, iterations, best_scores):
        best_score_plot = go.Scatter(
            x=iterations,
            y=best_scores,
            name='Best so far',
            mode='lines',
            line=dict(
                color='green'
            )
        )
        self.add_2d_plot(best_score_plot)

        yaxis_config = {'title': 'Score'}
        if self.minimise:
            yaxis_config['autorange'] = 'reversed'
            if all(score >= 0 for score in best_scores):
                yaxis_config['type'] = 'log'

        self.fig['layout']['yaxis1'].update(yaxis_config)
        self.fig['layout']['xaxis1'].update({'title': 'Iterations'})
        self.fig['layout']['legend'].update({'x': 0.45, 'y': 1})

    def plot_scores_by_type(self, configuration_type: str, colour: str, iterations, scores):
        configuration_type_plot = go.Scatter(
            x=iterations,
            y=scores,
            mode='markers',
            name=configuration_type,
            marker=dict(
                color=colour
            )
        )
        self.add_2d_plot(configuration_type_plot)

    def add_2d_plot(self, plot):
        self.fig.append_trace(plot, 1, self._2d_plot_column)

    def add_3d_plot(self, plot):
        self.fig.append_trace(plot, 1, self._3d_plot_column)

    def plot_3d_scores_by_type(self, configuration_type: str, colour: str, x_values, y_values, z_values):
        surface_plot = go.Scatter3d(
            x=x_values,
            y=y_values,
            z=z_values,
            mode='markers',
            name=configuration_type.capitalize(),
            showlegend=False,
            marker=dict(
                color=colour,
                size=4
            )
        )
        flat_plot = go.Scatter3d(
            x=x_values,
            y=y_values,
            z=[0] * len(z_values),
            mode='markers',
            name=configuration_type.capitalize(),
            showlegend=False,
            marker=dict(
                color=colour,
                size=4
            )
        )
        self.add_3d_plot(surface_plot)
        self.add_3d_plot(flat_plot)

    def plot_3d_surface(self, X, Y, Z):
        surface_plot = dict(type='surface', x=X, y=Y, z=Z, colorscale='Jet', opacity=0.5,
                            showscale=False)
        contour_plot = dict(type='surface', x=X, y=Y, z=np.zeros(Z.shape), colorscale='Jet',
                            surfacecolor=Z, opacity=0.75, showscale=False)
        self.add_3d_plot(surface_plot)
        self.add_3d_plot(contour_plot)
        self.fig['layout']['scene1'].update({'camera': self.camera})

    def plot_3d_target(self, target_x, target_y, target_z):
        target_score_plot = go.Scatter3d(
            x=[target_x],
            y=[target_y],
            z=[target_z],
            mode='markers',
            name='Target',
            marker=dict(
                color='green',
                size=6
            )
        )
        self.add_3d_plot(target_score_plot)


class MatPlotLibPlotter(ChartPlotter):
    def __init__(self, minimise: bool = None, use_3d_plot: bool = None, plot_best_scores_on_left: bool = None):
        plot_types_and_colours = {
            'random': 'k',
            'grid': 'k',
            'exploration': 'm',
            'initialisation': 'm',
            'exploitation': 'b',
            'default': 'c',
            'user-defined': 'k'
        }
        super().__init__(plot_types_and_colours, minimise, use_3d_plot, plot_best_scores_on_left)

    def start_update(self):
        plt.clf()
        self.fig = plt.figure(figsize=(20, 10))
        self.ax = self.fig.add_subplot(1, 2, self._2d_plot_column)

    def update_display(self):
        if self.use_3d_plot:
            self.ax3d.legend(loc='upper right', bbox_to_anchor=(0.11, 1))
        else:
            self.ax.legend()

        display(plt.gcf())
        plt.close('all')

    def plot_best_scores(self, iterations, best_scores):
        if self.minimise:
            self.ax.invert_yaxis()
            self.ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            if all(score >= 0 for score in best_scores):
                self.ax.set_yscale('log')

        self.ax.set_ylabel('Score')

        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax.set_xlabel('Iterations')

        self.ax.plot(best_scores, 'g', label='Best so far')

    def plot_scores_by_type(self, configuration_type: str, colour: str, iterations, scores):
        self.ax.plot(iterations, scores, 'o' + colour, label=configuration_type)

    def plot_3d_scores_by_type(self, configuration_type: str, colour: str, x_values, y_values, z_values):
        self.ax3d.scatter(x_values, y_values, z_values, c=colour, marker='o', zorder=10, label=configuration_type)
        self.ax3d.scatter(x_values, y_values, 0, c=colour, marker='o', zorder=10)

    def plot_3d_surface(self, X, Y, Z):
        self.ax3d = self.make_3d_axis()
        self.ax3d.view_init(15, 45)
        surface_plot = self.ax3d.plot_surface(X, Y, Z, cmap=plt.get_cmap('coolwarm'), zorder=2, rstride=1, cstride=1)
        surface_plot.set_alpha(0.25)
        self.ax3d.contourf(X, Y, Z, 50, zdir='z', offset=0, cmap=plt.get_cmap('coolwarm'), zorder=1)

    def plot_3d_target(self, target_x, target_y, target_z):
        self.ax3d.scatter([target_x], [target_y], [target_z], c='g', zorder=5, marker='o', label='Best')

    def make_3d_axis(self) -> Axes3D:
        return self.fig.add_subplot(1, 2, self._3d_plot_column, projection='3d')

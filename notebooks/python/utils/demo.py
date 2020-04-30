import math
import os
import random
from typing import List, Dict, Callable, Any, Union

import numpy as np
import pandas as pd
from IPython.display import display, clear_output

from mindfoundry.optaas.client.task import Configuration
from mindfoundry.optaas.client.utils import get_choice_value, get_choice_name
from utils.plotters import PlotlyChartPlotter, MatPlotLibPlotter

DEFAULT_DISPLAY_FREQUENCY = 1


class Demo:
    def __init__(self, configuration_keys: List[str],
                 use_plotly: bool = False, getters: Dict[str, Union[str, Callable]] = None,
                 minimise: bool = None, use_3d_plot: bool = False, display_frequency: int = None, plot_table: bool = True,
                 plot_best_scores_on_left: bool = True, display_table_best_first: bool = True, clear_output: bool = True):
        self.use_plotly = use_plotly
        self.plot_table = plot_table
        chart_plotter_type = PlotlyChartPlotter if use_plotly else MatPlotLibPlotter
        self.chart_plotter = chart_plotter_type(minimise=minimise, use_3d_plot=use_3d_plot,
                                                plot_best_scores_on_left=plot_best_scores_on_left)
        self.chart_plotter.use_3d_plot = use_3d_plot
        self.chart_plotter.minimise = minimise

        self.display_table_best_first = display_table_best_first

        self.current_iteration = 0
        self.display_frequency = DEFAULT_DISPLAY_FREQUENCY if display_frequency is None else display_frequency

        self.better_of = min if minimise else max
        self.best_score = math.inf if minimise else -math.inf
        self.best_scores = []

        self.configuration_keys = configuration_keys
        self.getters = getters or {}

        self.df = pd.DataFrame({key: [] for key in configuration_keys + ['Score', 'Configuration Type']})
        self.all_scores = {type: {} for type in self.chart_plotter.plot_types_and_colours}

        self.clear_output = clear_output

    def display(self, configuration: Configuration, score: float, counter: int):
        self.update_data(configuration, score)
        if counter % self.display_frequency == 0:
            self.update_display()

    def update_data(self, configuration: Configuration, score: float):
        self.log_configuration(configuration)
        self.log_score(configuration, score)

    def log_configuration(self, configuration: Configuration):
        data_values = {key: self.get_value(configuration.values, key) for key in self.configuration_keys}
        data_values['Score'] = None
        data_values['Configuration Type'] = configuration.type
        self.all_scores[configuration.type][self.current_iteration] = data_values
        self.df.loc[self.current_iteration] = data_values

    def log_score(self, configuration: Configuration, score: float):
        self.all_scores[configuration.type][self.current_iteration]['Score'] = score
        self.df.at[self.current_iteration, 'Score'] = score

        if self.display_table_best_first:
            self.df = self.df.sort_values(by=['Score'], ascending=self.chart_plotter.minimise)

        self.best_score = self.better_of(self.best_score, score)
        self.best_scores.append(self.best_score)

        self.current_iteration += 1

    def get_value(self, config: Dict, key: str) -> Any:
        value = config.get(key)
        getter = self.getters.get(key)
        if getter == 'choice_name':
            value = get_choice_name(value)
        elif getter == 'choice_value':
            value = get_choice_value(value)
        elif getter == 'is_present':
            value = (value is not None)
        elif getter is None:
            if isinstance(value, Dict):
                value = list(value.values())
                if len(value) == 1:
                    value = value[0]
        else:
            value = getter(config)
        return value

    def update_display(self):
        if self.clear_output:
            clear_output(wait=True)

        self.chart_plotter.start_update()

        self.add_overall_plots()

        for type, colour in self.chart_plotter.plot_types_and_colours.items():
            iterations = self.all_scores[type].keys()
            if iterations:
                dicts = self.all_scores[type].values()
                values = {key: [d[key] for d in dicts] for key in (self.configuration_keys + ['Score'])}
                self.add_per_configuration_type_plots(type.capitalize(), colour, iterations, values)

        self.chart_plotter.update_display()
        self.display_table()

    def display_table(self):
        if self.plot_table:
            display(self.df)

    def add_overall_plots(self):
        iterations = list(range(self.current_iteration))
        self.chart_plotter.plot_best_scores(iterations, self.best_scores)

    def add_per_configuration_type_plots(self, configuration_type: str, colour: str, iterations: List[int],
                                         values: Dict):
        self.chart_plotter.plot_scores_by_type(configuration_type, colour, list(iterations),
                                               values['Score'])


class DemoWith3dPlot(Demo):
    def __init__(self, minimum: float, maximum: float, score_function,
                 use_plotly: bool = None, minimise: bool = None, display_frequency: int = None,
                 use_3d_log_scale: bool = None, detail: int = 100):
        super().__init__(configuration_keys=['x', 'y'], use_plotly=use_plotly, minimise=minimise, use_3d_plot=True,
                         display_frequency=display_frequency)
        self.minimum = minimum
        self.maximum = maximum
        self.score_function = score_function

        self.detail = detail
        x_space = np.linspace(minimum, maximum, num=detail)
        y_space = np.linspace(minimum, maximum, num=detail)
        self.X, self.Y = np.meshgrid(x_space, y_space)

        if use_3d_log_scale is None:
            use_3d_log_scale = self.chart_plotter.minimise
        self.use_3d_log_scale = use_3d_log_scale

        if use_3d_log_scale:
            self.Z = np.log(score_function(self.X, self.Y) + 1)
        else:
            self.Z = np.vectorize(score_function)(self.X, self.Y)

    def add_overall_plots(self):
        super().add_overall_plots()
        self.chart_plotter.plot_3d_surface(self.X, self.Y, self.Z)

    def add_per_configuration_type_plots(self, configuration_type: str, colour: str, iterations: List[int],
                                         values: Dict):
        super().add_per_configuration_type_plots(configuration_type, colour, iterations, values)

        scores = values['Score']
        if self.use_3d_log_scale:
            scores = np.log([score + 1 for score in scores])
        self.chart_plotter.plot_3d_scores_by_type(configuration_type, colour, values['x'], values['y'], scores)

    def display_random_search(self):
        RandomSearchDemo(self).display_all()

    def display_grid_search(self):
        GridSearchDemo(self).display_all()


class ComparisonWith3dDemo(DemoWith3dPlot):
    def __init__(self, parent: DemoWith3dPlot, comparison_type: str):
        super().__init__(minimum=parent.minimum, maximum=parent.maximum, score_function=parent.score_function,
                         use_plotly=parent.use_plotly, minimise=parent.chart_plotter.minimise, detail=parent.detail,
                         use_3d_log_scale=parent.use_3d_log_scale, display_frequency=parent.display_frequency)
        self.best_score = parent.best_score
        self.number_of_iterations = len(parent.best_scores)
        self.comparison_type = comparison_type

    def display_all(self):
        for i in range(self.number_of_iterations):
            x, y = self.get_parameter_values(i)
            score = self.score_function(x, y)
            configuration = make_configuration(x, y, self.comparison_type)
            self.update_data(configuration, score)
        self.update_display()

    def get_parameter_values(self, counter: int):
        raise NotImplementedError()


class RandomSearchDemo(ComparisonWith3dDemo):
    def __init__(self, parent: DemoWith3dPlot):
        super().__init__(parent, 'random')
        random.seed(101)

    def display_all(self):
        super().display_all()

    def get_parameter_values(self, counter: int):
        x = random.uniform(self.minimum, self.maximum)
        y = random.uniform(self.minimum, self.maximum)
        return x, y


class GridSearchDemo(ComparisonWith3dDemo):
    def __init__(self, parent: DemoWith3dPlot):
        super().__init__(parent, 'grid')
        self.grid_size = (self.number_of_iterations // 4) * 3
        self.x_grid = np.linspace(self.minimum, self.maximum, num=self.grid_size)
        self.y_grid = np.linspace(self.maximum, self.minimum, num=self.grid_size)

    def get_parameter_values(self, counter: int):
        x = self.x_grid[counter % self.grid_size]
        y = self.y_grid[counter // self.grid_size]
        return x, y


def make_configuration(x: float, y: float, type: str) -> Configuration:
    return Configuration({
        'type': type,
        'values': {
            'x': x,
            'y': y
        },
        'id': random.randint(111, 999),
        '_links': {
            'results': {
                'href': ''
            }
        }
    })

import numpy as np
import matplotlib.pyplot as plt


########################################################################################################################
########################################################################################################################
# ------------Define toy problems:
class PriorMeansSimpleDemo:
    def __init__(self, conditional_offset=20):
        self.conditional_offset = conditional_offset

    ####################################################################################################################
    # The functions
    @staticmethod
    def base_function(x: float) -> float:
        """
        A simple polynomial with multiple local maxima. This will be passed as a prior to the PriorFromExpression profile.
        """
        return (x + 3) * (2 - x) * (x - 1) * (x + 2) * (x - .5) * (x + 1.5)

    def target(self, input_dict: dict) -> float:
        """
        Defines a slight modification of the base_function, changing slightly the position of the maximum
        """
        x = input_dict['id_x']
        return self.base_function(x) + 5 * np.sin(10 * x - 10)

    def conditional_base_function(self, input_dict) -> float:
        """
        Defines a conditional base function for a conditional space with one selection parameter, and one dimension in each
        leaf.
        """
        y = input_dict['id_y']

        if y == 0:
            return self.base_function(input_dict['id_x_1'])
        if y == 1:
            return self.base_function(input_dict['id_x_2']) + self.conditional_offset
        else:
            raise Exception("input y should have value of 0 or 1")


    def conditional_target(self, input_dict) -> float:
        """
        Defines a conditional target where the slight modification of the base_function depend on the selection parameter y,
        changing slightly the position of the maximum
        """
        y = input_dict['id_y']
        if y == 0:
            x = input_dict['id_x_1']
        elif y == 1:
            x = input_dict['id_x_2']
        else:
            raise Exception("input y should have value of 0 or 1")

        return self.conditional_base_function(input_dict) + 5 * np.sin(10 * x - 10)

    ####################################################################################################################
    # Plotting:

    def plot_target_against_mean(self, start, end, figure=None, show=True):
        # Plot the target function and means
        if figure is None:
            fig0 = plt.figure()
        else:
            fig0 = figure

        # Plot on the full input domain
        ax1 = plt.subplot(121)
        input_data = np.linspace(num=1000, start=start, stop=end)
        target_data = [self.target({'id_x': x}) for x in input_data]
        mean_data = [self.base_function(x) for x in input_data]
        ax1.plot(input_data, target_data, color='blue', label='target_func')
        ax1.plot(input_data, mean_data, '--', color='blue', label='prior_mean_func')

        # Plot focusing on close to where the maximum is
        ax2 = plt.subplot(122)
        input_data = np.linspace(num=1000, start=-3.1, stop=2.2)
        target_data = [self.target({'id_x': x}) for x in input_data]
        mean_data = [self.base_function(x) for x in input_data]
        ax2.plot(input_data, target_data, color='blue', label='target_func')
        ax2.plot(input_data, mean_data, '--', color='blue', label='prior_mean_func')
        ax2.set_xlim(-3.1, 2.2)
        ax2.set_ylim(min(target_data), max(target_data))

        # Add formatting: title and legend
        plt.legend()
        plt.suptitle('Targets and Means')

        if show:
            plt.show()

        return fig0, ax1, ax2

    def plot_conditional_target_against_prior_means(self, start, end, figure=None, show=True):
        # Plot the target function and means
        if figure is None:
            fig0 = plt.figure()
        else:
            fig0 = figure

        # Plot on the full input domain
        ax1 = plt.subplot(121)
        input_data = np.linspace(num=1000, start=start, stop=end)
        target_data_y_0 = [self.conditional_target({'id_x_1': x, 'id_y': 0}) for x in input_data]
        mean_data_y_0 = [self.conditional_base_function({'id_x_1': x, 'id_y': 0}) for x in input_data]
        target_data_y_1 = [self.conditional_target({'id_x_2': x, 'id_y': 1}) for x in input_data]
        mean_data_y_1 = [self.conditional_base_function({'id_x_2': x, 'id_y': 1}) for x in input_data]
        ax1.plot(input_data, target_data_y_0, color='black', label='target')
        ax1.plot(input_data, mean_data_y_0, '--', color='black', label='mean_func y=0')
        ax1.plot(input_data, target_data_y_1, color='blue', label='target')
        ax1.plot(input_data, mean_data_y_1, '--', color='blue', label='mean_func y=1')

        # Plot focusing on close to where the maximum is
        ax2 = plt.subplot(122)
        input_data = np.linspace(num=1000, start=-3.1, stop=2.2)
        target_data_y_0 = [self.conditional_target({'id_x_1': x, 'id_y': 0}) for x in input_data]
        mean_data_y_0 = [self.conditional_base_function({'id_x_1': x, 'id_y': 0}) for x in input_data]
        target_data_y_1 = [self.conditional_target({'id_x_2': x, 'id_y': 1}) for x in input_data]
        mean_data_y_1 = [self.conditional_base_function({'id_x_2': x, 'id_y': 1}) for x in input_data]
        ax2.plot(input_data, target_data_y_0, color='black', label='target')
        ax2.plot(input_data, mean_data_y_0, '--', color='black', label='mean_func y=0')
        ax2.plot(input_data, target_data_y_1, color='blue', label='target')
        ax2.plot(input_data, mean_data_y_1, '--', color='blue', label='mean_func y=1')
        ax2.set_xlim(-3.1, 2.2)
        ax2.set_ylim(min(min(target_data_y_0), min(target_data_y_1)),
                     max(max(target_data_y_0), max(target_data_y_1))
                     )

        # Add formatting: title and legend
        plt.legend()
        plt.title('Targets and Means')

        if show:
            plt.show()

        return fig0, ax1, ax2

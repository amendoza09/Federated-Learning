import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datagen import visualize_data


class SimpleLinRegModel:
    """
    A class modeling the parameters of simple linear regression model: f(x) = wx + b.
    This class stores two parameters (w, b) of regression model into a 1D numpy array. The
    first and second elements in `_parameters` member variable correspond to `w` and `b`, respectively.
    """
    def __init__(self):
        self._parameters = np.random.randn(2)

    # properties defined for convenience
    @property
    def slope(self):
        return self._parameters[0]

    @slope.setter
    def slope(self, value):
        self._parameters[0] = value

    @property
    def bias(self):
        return self._parameters[1]

    @bias.setter
    def bias(self, value):
        self._parameters[1] = value

    # member functions
    def compute_gradient(self, dataset):
        """
        This function computes the gradient of simple linear regression model. In other words,
        it returns a vector containing (dL/dw, dL/db) where L denotes the objective function.

        Parameters:
        -----------------------------------
        dataset: 2D numpy array of size n x 2, where n is the number of datapoints in the dataset.
                 The first column is the x-coordinate (i.e., a feature) and the second column is the 
                 response variable.

        Returns:
        ------------------------------------
        grad: 1D numpy array containing (dL/dw, dL/db), its shape is (2,).
        """
        # --------------------------------#
        #      Your code goes here.       #
        # --------------------------------#
        n = dataset.shape[0]
        x = dataset[:, 0]
        y = dataset[:, 0]
        
        grad = np.zeros(2)
        
        y_pred = x * n + y
        
        grad[0] = (-2/n) * np.sum(x * (y - y_pred))
        grad[1] = (-2/n) * np.sum( y - y_pred)

        return grad

    def visualize_model(self, datasets, save=False, filename=None):
        """
        Visualize the current model parameters (i.e., slope and bias) along
        with input dataset

        Parameters:
        -------------------------------------
        datasets: list of datasets held by each client, i.e., list of 2D numpy arrays
        save: boolean, If set True, the plot will be saved as a png file.
        filename: a string corresponding to the file name to save
        """
        fig = visualize_data(self.slope, self.bias, datasets)
        if save:
            if filename is None:
                raise ValueError("`save` is set true but filename is not provided.")
            fig.savefig(filename)
        else:
            plt.show()
        plt.close(fig)

    def __repr__(self):
        return f"slope: {self.slope:8.5f} bias: {self.bias:8.5f}"


class FLServer:
    def __init__(self):
        self.params = SimpleLinRegModel()

        # The server model parameters are initialized to (-0.1, 0).
        self.params.slope = -0.1
        self.params.bias = 0

    def on_round_start(self, simulator, num_clients):
        """
        This event is triggered when the simulator starts a new round.
        Parameters:
        ------------------
        simulator: an instance of UGAFL class
        num_clients: integer, number of clients being simulated
        """

        simulator.send_to_clients(self.params)

    def on_parameters_received(self, simulator, param_list):
        """
        This event is triggered when the server receives the updated parameters
        from all the clients. It needs to aggregate the received parameters, and the resulting
        parameter will be used in the next round.

        Parameters:
        -----------------------------
        simulator: an instance of UGAFL class
        param_list: built-in list object containing `num_clients` instances of model parmaeters
        """

        # --------------------------------#
        #      Your code goes here.       #
        # --------------------------------#
        
        total_slope = self.params.slope
        total_bias = self.params.bias
        
        for params in param_list:
            total_slope += params.slope
            total_bias += params.bias
        
        avg_slope = total_slope / simulator.num_clients
        avg_bias = total_bias / simulator.num_clients
        
        self.params.slope = avg_slope
        self.params.bias = avg_bias
        
        simulator.round_completed()
        



class FLClient:
    def __init__(self, rank, num_local_updates, local_step_size):
        self.rank = rank
        self.num_local_updates = num_local_updates
        self.local_step_size = local_step_size
        self.params = SimpleLinRegModel()

        # laod the dataset
        self.dataset = self.load_dataset()

        print(f"\tclient {rank}, dataset size={self.dataset.shape[0]}",
              "initial solution=", self.params)

    def load_dataset(self):
        """
        Load data from a csv file. The input file for client i is 'dataset_client{i}.csv'.

        Returns:
        --------------------------
        dataset: This is a a two-dimensional numpy array containing data loaded
                 from the csv file.
        """

        # --------------------------------#
        #      Your code goes here.       #
        # --------------------------------#

        filename = f'dataset_client{self.rank}.csv'
        
        try:
            data = pd.read_csv(filename)
            
            dataset = data.to_numpy()
            
            return dataset
        except FileNotFoundError:
            
            print(f"File '{filename}' not found")
            return None

    def on_parameters_received(self, simulator, server_params):
        """
        This function is called when a client received the model parameters from the server
        at the begining of current round. Unpon receiving the model parameters, a
        client needs needs to perform `self.num_local_updates` steps of gradient descent update.

        Parameters:
        ---------------------
        server_params: SimpleLinRegModel object received from the server
        """

        # --------------------------------#
        #      Your code goes here.       #
        # --------------------------------#

        for _ in range(self.num_local_updates):
            gradient = self.params.compute_gradient(self.dataset)
            
            learning_rate = self.local_step_size
            

        # local gradient descent steps are done!
        print(f"[{simulator.round_num}] client {self.rank} ({self.params})")

        # send the locally updated parameters to the server
        simulator.send_to_server(self.params)


class UGAFL:
    """
    A simple Federated Learning (FL) simulator. The overall control flow is as follows.

    run_single_round() --->  server.on_round_start() ---> send_to_clients() ---->
    client.on_parameters_received() ---> send_to_server() --->
    server.on_parameters_received() ---> round_completed()
    """
    def __init__(
        self,
        n_clients: int,
        num_local_updates: int = 3,
        step_size: float = 0.01
    ):
        """
        Parameters:
        --------------------
        n_clients: integer, number of clients to simulate
        """
        self.cts_queue = list()  # queue for client to server messages

        self.round_num = 0
        self.server = FLServer()
        print(f"UGAFL: creating {n_clients} clients...")
        self.clients = [FLClient(rank, num_local_updates, step_size)
                        for rank in range(n_clients)]

        # only for visualizeation purpose
        self.datasets = [client.dataset for client in self.clients]

    @property
    def num_clients(self):
        return len(self.clients)

    def visualize_model(self, save=False, filename=None):
        """
        This function visualizes the server's current model parameters along
        with each client's dataset.

        Parameters:
        -------------------------------------
        datasets: list of datasets held by each client, i.e., list of 2D numpy arrays
        save: boolean, If set True, the plot will be saved as a png file.
        filename: a string corresponding to the file name to save

        """
        self.server.params.visualize_model(self.datasets,
                                           save=save,
                                           filename=filename)

    def run_single_round(self, round_num):
        # start a round
        self.round_num = round_num
        print(f"\nUGAFL: Round {round_num} started")

        # notify the server, so that it can start distributing the parameters to clients.
        self._create_event(self.server, 'on_round_start', self, self.num_clients)

        # server received the update parameters from all clients.
        if len(self.cts_queue) == self.num_clients:
            self._create_event(self.server, 'on_parameters_received', self, self.cts_queue)
        else:
            print("UGAFL: something went wrong!")

    def round_completed(self):
        self.cts_queue.clear()
        print(f"UGAFL: Round {self.round_num} completed, server ({self.server.params})")

    def send_to_clients(self, params):
        for client in self.clients:
            self._create_event(client, 'on_parameters_received', self, params)

    def send_to_server(self, params):
        # client-to-server communications are done by putting the message (i.e., parameters) into
        # the queue.
        self.cts_queue.append(copy.deepcopy(params))

    def _create_event(self, event_source, event_name, *args, **kwargs):
        # call event handlers corresponding to the events
        if event_source is None:
            raise ValueError('Event source cannot be a null object.')

        func = getattr(event_source, event_name)

        if not callable(func):
            raise RuntimeError(f'Cannot create {event_name} for {type(event_source)} object.')

        output = func(*args, **kwargs)

        return output
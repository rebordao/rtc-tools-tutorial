import logging

from rtctools.simulation.csv_mixin import CSVMixin
from rtctools.simulation.simulation_problem import SimulationProblem
from rtctools.util import run_simulation_problem

logger = logging.getLogger("rtctools")


class Example(CSVMixin, SimulationProblem):
    """
    A basic example for introducing users to RTC-Tools 2 Simulation
    """

    def initialize(self):
        self.set_var("P_control", 0.0)
        super().initialize()

    # Min and Max flow rate that the storage is capable of releasing
    min_release, max_release = 0.0, 8.0  # m^3/s

    # Here is an example of overriding the update() method to show how control
    # can be build into the python script
    def update(self, dt):
        # Get the time step
        if dt < 0:
            dt = self.get_time_step()

        # Get relevant model variables
        volume = self.get_var("storage.V")
        target = self.get_var("storage_V_target")

        # Calucate error in storage.V
        error = target - volume

        # Calculate the desired control
        control = -error / dt

        # Get the closest feasible setting.
        bounded_control = min(max(control, self.min_release), self.max_release)

        # Set the control variable as the control for the next step of the simulation
        self.set_var("P_control", bounded_control)

        # Call the super class so that everything else continues as normal
        super().update(dt)


# Run
run_simulation_problem(Example, log_level=logging.DEBUG)

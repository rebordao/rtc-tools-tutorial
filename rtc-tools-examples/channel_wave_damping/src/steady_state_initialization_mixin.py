from rtctools.optimization.optimization_problem import OptimizationProblem


class SteadyStateInitializationMixin(OptimizationProblem):
    def constraints(self, ensemble_member):
        c = super().constraints(ensemble_member)
        times = self.times()
        parameters = self.parameters(ensemble_member)
        # Force steady-state initialization at t0 and at t1.
        for reach in ["upstream", "middle", "downstream"]:
            for i in range(int(parameters["{}.n_level_nodes".format(reach)]) + 1):
                state = "{}.Q[{}]".format(reach, i + 1)
                c.append((self.der_at(state, times[0]), 0, 0))
        return c

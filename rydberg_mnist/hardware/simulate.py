"""QuTiP three-level dynamics with always-on finite van der Waals interactions."""
from itertools import combinations

import numpy as np
import qutip as qt


class PulseSimulator:
    def __init__(self, n, profile, noisy=False, seed=7):
        if not 2 <= n <= 8:
            raise ValueError("Supported register sizes: 2..8")
        if noisy and n > 4:
            raise ValueError("Exact Lindblad simulation is limited to <=4 atoms here; "
                             "compile 8-atom schedules for accounting, or add trajectory simulation")
        self.n, self.profile, self.noisy = n, profile, noisy
        self.basis = [qt.basis(3, i) for i in range(3)]
        self.cache = {}
        self.nr = [self.on_site(self.basis[2]*self.basis[2].dag(), q) for q in range(n)]
        self.nr_sum = sum(self.nr)
        self.coordinates = np.column_stack((np.arange(n)*profile["spacing_um"], np.zeros(n)))
        noise = profile["noise"]
        rng = np.random.default_rng(seed)
        if noisy:
            self.coordinates += rng.normal(0, noise["position_std_um"], (n, 2))
        self.drift = 0*self.nr_sum
        for i, j in combinations(range(n), 2):
            distance = np.linalg.norm(self.coordinates[j]-self.coordinates[i])
            self.drift += profile["c6_rad_us_um6"]/distance**6 * self.nr[i]*self.nr[j]
        detunings = rng.normal(0, noise["detuning_std_rad_us"], n) if noisy else np.zeros(n)
        self.drift -= sum(detunings[q]*self.nr[q] for q in range(n))
        self.amplitude_factors = {name: 1+rng.normal(0, noise["amplitude_fraction_std"]) if noisy else 1.
                    for name in ("raman_local", "stark_local", "rydberg_local", "rydberg_global")}
        self.c_ops = []
        if noisy:
            fraction = noise["rydberg_decay_to_zero_fraction"]
            for q in range(n):
                for ground, probability in ((0, fraction), (1, 1-fraction)):
                    if probability:
                        self.c_ops.append(np.sqrt(probability/noise["rydberg_lifetime_us"])*
                                 self.on_site(self.basis[ground]*self.basis[2].dag(), q))
                self.c_ops.append(np.sqrt(2/noise["rydberg_pure_dephasing_us"])*self.nr[q])
                self.c_ops.append(np.sqrt(1/(2*noise["storage_pure_dephasing_us"]))*
                                  self.on_site(qt.Qobj(np.diag([1., -1., 0.])), q))
        self.options = {"method": "bdf", "atol": 1e-9, "rtol": 1e-7,
                        "nsteps": 100000, "normalize_output": False,
                        "store_states": True, "progress_bar": False}

    def on_site(self, operator, q):
        return qt.tensor([operator if j == q else qt.qeye(3) for j in range(self.n)])

    def operator(self, name, targets):
        key = (name, tuple(targets))
        if key not in self.cache:
            pair, axis = name[:2], name[2]
            a, b = {"01": (0, 1), "1r": (1, 2), "0r": (0, 2)}[pair]
            transition = self.basis[a]*self.basis[b].dag()
            if axis == "x":
                local = transition+transition.dag()
            elif axis == "y":
                local = -1j*transition+1j*transition.dag()
            elif axis == "z":
                local = self.basis[a]*self.basis[a].dag()-self.basis[b]*self.basis[b].dag()
            else:
                raise ValueError("Unknown pulse axis")
            self.cache[key] = sum(self.on_site(local/2, q) for q in targets)
        return self.cache[key]

    def initial(self):
        return qt.tensor([self.basis[0]]*self.n)

    def evolve(self, schedule, initial=None):
        state = self.initial() if initial is None else initial
        if self.noisy and state.isket:
            state = qt.ket2dm(state)
        dwell = 0.
        max_normalization_error = 0.
        for segment in schedule:
            duration = segment.duration_us
            if any(q < 0 or q >= self.n for q in segment.targets):
                raise ValueError("Pulse target outside register")
            factor = self.amplitude_factors[segment.channel]
            a, b = factor*segment.amplitude_start, factor*segment.amplitude_stop
            def envelope(t):
                return a+(b-a)*t/duration
            h = [self.drift, [self.operator(segment.operator, segment.targets), envelope]]
            if segment.detuning_start or segment.detuning_stop:
                d0, d1 = segment.detuning_start, segment.detuning_stop
                def detuning(t):
                    return -d0-(d1-d0)*t/duration
                h.append([sum(self.nr[q] for q in segment.targets), detuning])
            times = np.linspace(0, duration, 7)
            if state.isket:
                # High-order explicit integration avoids the norm drift that
                # multistep BDF can accumulate over many short unitary segments.
                options = {**self.options, "method": "dop853", "atol": 1e-11, "rtol": 1e-9}
                result = qt.sesolve(h, state, times, e_ops=[self.nr_sum], options=options)
            else:
                result = qt.mesolve(h, state, times, c_ops=self.c_ops, e_ops=[self.nr_sum], options=self.options)
            dwell += float(np.trapezoid(result.expect[0], times))
            state = result.states[-1]
            normalization = state.norm()**2 if state.isket else state.tr().real
            max_normalization_error = max(max_normalization_error, abs(normalization-1))
        if max_normalization_error > 1e-5:
            raise RuntimeError(f"Integrator normalization error {max_normalization_error}; tighten tolerances")
        probs = np.abs(state.full().ravel())**2 if state.isket else state.diag().real
        if np.min(probs) < -1e-6:
            raise RuntimeError("Unphysical negative populations")
        probs = np.maximum(probs, 0)
        probs /= probs.sum()  # Correct integrator roundoff only, never discard Rydberg population.
        digits = ((np.arange(3**self.n)[:, None] // 3**np.arange(self.n-1, -1, -1)) % 3)
        erasure = float(probs[np.any(digits == 2, axis=1)].sum())
        return state, {"probabilities": probs, "residual_rydberg_probability": erasure,
                       "rydberg_atom_time_us": dwell,
                       "max_normalization_error": max_normalization_error,
                       "dwell_quadrature": "7 output points per linear segment; approximate diagnostic"}

    def features(self, probabilities, shots=0, seed=7):
        digits = ((np.arange(3**self.n)[:, None] // 3**np.arange(self.n-1, -1, -1)) % 3)
        noise = self.profile["noise"]
        fp = noise["readout_false_positive"] if self.noisy else 0.
        fn = noise["readout_false_negative"] if self.noisy else 0.
        if shots:
            rng = np.random.default_rng(seed)
            measured = digits[rng.choice(len(probabilities), size=shots, p=probabilities)]
            flips = rng.random(measured.shape)
            zero, one = measured == 0, measured == 1
            measured[zero & (flips < fp)] = 1
            measured[one & (flips < fn)] = 0
            z = np.array([1., -1., 0.])[measured]
            return np.array([*z.mean(axis=0), *[(z[:, i]*z[:, j]).mean()
                            for i, j in combinations(range(self.n), 2)]])
        # Conditional independence of detection errors at different sites.
        z = np.array([1-2*fp, -(1-2*fn), 0.])[digits]
        observables = np.column_stack([z] + [z[:, i]*z[:, j] for i, j in combinations(range(self.n), 2)])
        return probabilities @ observables


def computational_indices(n):
    bits = (np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1
    return bits @ (3**np.arange(n-1, -1, -1))

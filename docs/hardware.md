# Hardware execution requirements

There is currently **no QPU submission backend** in this repository. All recorded results come from the exact CPU simulator. No paid quantum jobs have been submitted.

The Lu digital–analog baseline needs preparation of arbitrary per-qubit product states, individually controlled RX/RZ rotations, global Rydberg evolution, and switching between hyperfine storage and Rydberg manifolds. A purely global analog interface does not supply all these operations. Matching the Rydberg Hamiltonian alone does not make this circuit runnable on every Rydberg device.

The [ResQ paper](https://openaccess.thecvf.com/content/ICCV2025/papers/DiBrita_ResQ_A_Novel_Framework_to_Implement_Residual_Neural_Networks_on_ICCV_2025_paper.pdf) offers a different, analog-native route: features and trainable parameters are embedded in control pulses. Its Aquila experiments used local detuning described as experimental in the paper. Current availability, device limits and account permissions must be checked before selecting that path.

For a device experiment, first identify the target QPU and supported controls. Then compile the chosen algorithm into valid geometry and waveforms, checking detuning, Rabi frequency, slew rate, minimum separation, duration and measurement conventions. Revalidate with the vendor simulator, followed by a small hardware correctness run. The paper's 4 MHz Rabi setting must not be assumed valid for an arbitrary device.

On hardware, Z and ZZ observables can be estimated together from the same measured bitstrings. The finite-shot simulation implements this joint sampling. The current optimizer uses exact statevector adjoints, so device-based training additionally needs an implemented and validated gradient estimator and an explicit shot budget.

Report hardware accuracy separately from simulator accuracy, including atom-loss handling, readout calibration, device identifiers, shot counts, failure rates, control waveforms, timestamps and training location. For energy estimation, include preparation/readout overheads, powers, efficiencies and classical processing; `0.5 us` of analog evolution per current circuit is not a wall-plug energy estimate.

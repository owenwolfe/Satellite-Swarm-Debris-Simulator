# Satellite Swarm & Space Debris Simulator (OpenMP)

A fast, terminal-based satellite swarm + space debris simulator written in c++.  
It models simple orbital dynamics around Earth and includes optional **spatial-hash collision detection** and a **Kessler-style debris cascade**.

I orignally just starting making this for fun, then turned into a performance project: benchmarkable serial vs. OpenMP, with clear Amdahl’s Law behavior when collisions dominate runtime.

---

## Features
- **Orbital dynamics** (Earth at origin) with a stable symplectic (semi-implicit) Euler integrator
- **OpenMP parallelization** of the per-body integration loop
- Optional **spatial hashing** broadphase collision detection
- Optional **Kessler cascade**: collisions can spawn new debris
- Built-in **benchmark mode** (`--bench`) for serial vs. OpenMP comparisons
- **No graphics dependencies** (runs great on remote servers / SSH)

---

## Build

### Linux
```bash
g++ -O3 -std=c++17 -fopenmp sat_swarm.cpp -o sat_swarm
```
If your system doesn’t support OpenMP, remove -fopenmp and run with --no-omp.

## Run
Basic simulation
```bash
./sat_swarm --n 20000 --steps 2000 --threads 4
```
Benchmark serial vs OpenMP
```bash
./sat_swarm --n 400000 --steps 1500 --threads 4 --no-collisions --bench
```
Collision and Kessler mode
```bash
./sat_swarm --n 50000 --steps 1500 --threads 4
```
To disable debris spawning
```bash
./sat_swarm --n 200000 --steps 1200 --threads 4 --no-kessler
```

## CLI Option
CLI Options
--n N : number of bodies (default 20000)

--steps S : simulation steps (default 4000)

--threads T : OpenMP threads

--no-omp : disable OpenMP

--bench : run serial vs OpenMP benchmark

--no-collisions : disable collision detection entirely (isolates integrator scaling)

--no-kessler : disable debris spawning on collision

--debris K : debris per collision (default 1)

## Performance
Integrator scaling (collisions OFF)

On an Intel i5-5250U (4 logical CPUs), N = 400,000, steps = 1500:
| Mode   | Threads | Total Time | Avg ms/step | Speedup   |
| ------ | ------- | ---------- | ----------- | --------- |
| Serial | 1       | 33,006 ms  | 21.99       | 1.00×     |
| OpenMP | 4       | 22,758 ms  | 15.16       | **1.45×** |

Note: When collisions are enabled, performance is often dominated by serial work
(spatial hash build / collision checks / compaction), so OpenMP speedup can drop below 1× —
a practical example of Amdahl’s Law.


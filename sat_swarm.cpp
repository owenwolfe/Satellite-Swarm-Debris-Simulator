// sat_swarm.cpp
// Space Debris Swarm Simulator (terminal-only, no SFML)
// - OpenMP-parallel orbital integration
// - Optional spatial-hash collision detection + Kessler debris cascade
// - Benchmark mode: serial vs OpenMP

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>
#include <chrono>
#include <algorithm>
#include <string>

#ifdef _OPENMP
  #include <omp.h>
#endif

struct Body {
  float x, y;
  float vx, vy;
  float r;      // collision radius
  uint32_t id;  // stable id (debug/stats)
};

struct SimConfig {
  float mu = 2.0e5f;         // gravity strength
  float dt = 0.0025f;        // timestep
  float softening = 25.0f;   // avoids singularity at origin

  float spawn_r_min = 180.0f;
  float spawn_r_max = 360.0f;

  float kill_radius = 35.0f;   // respawn if hits Earth
  float world_radius = 1200.0f;

  // collision + spatial hash
  float cell_size = 6.0f;      // should be ~2-4x collision radius
  float coll_dist = 2.5f;      // base collision distance
  bool  kessler = true;        // spawn debris on collision
  int   debris_per_collision = 1;

  // run
  int substeps = 6;
};

static inline float inv_sqrt(float x) { return 1.0f / std::sqrt(x); }

static Body spawn_orbiter(std::mt19937 &rng, const SimConfig &cfg, uint32_t id) {
  std::uniform_real_distribution<float> ang(0.f, 2.f * 3.1415926535f);
  std::uniform_real_distribution<float> rr(cfg.spawn_r_min, cfg.spawn_r_max);
  std::uniform_real_distribution<float> jitter(-0.08f, 0.08f);

  float a = ang(rng);
  float r = rr(rng);

  float x = r * std::cos(a);
  float y = r * std::sin(a);

  float v = std::sqrt(cfg.mu / r);
  float tx = -std::sin(a);
  float ty =  std::cos(a);

  Body b;
  b.x = x; b.y = y;
  b.vx = v * tx * (1.0f + jitter(rng));
  b.vy = v * ty * (1.0f + jitter(rng));
  b.r = 1.0f;
  b.id = id;
  return b;
}

static void step_bodies(std::vector<Body> &bodies, const SimConfig &cfg, bool use_openmp) {
  const int n = (int)bodies.size();
  const float dt = cfg.dt;

  auto worker = [&](int i){
    Body &b = bodies[i];
    float x = b.x, y = b.y;

    float r2 = x*x + y*y + cfg.softening;
    float inv_r = inv_sqrt(r2);
    float inv_r3 = inv_r * inv_r * inv_r;

    float ax = -cfg.mu * x * inv_r3;
    float ay = -cfg.mu * y * inv_r3;

    // symplectic Euler
    b.vx += ax * dt;
    b.vy += ay * dt;
    b.x  += b.vx * dt;
    b.y  += b.vy * dt;
  };

#ifdef _OPENMP
  if (use_openmp) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) worker(i);
  } else
#endif
  {
    for (int i = 0; i < n; i++) worker(i);
  }
}

// Spatial hash key for integer cell coords (cx, cy)
static inline uint64_t cell_key(int cx, int cy) {
  // pack two signed 32-bit ints into 64 bits
  return (uint64_t)(uint32_t)cx << 32 | (uint32_t)cy;
}

struct CollisionStats {
  uint64_t checked_pairs = 0;
  uint64_t collisions = 0;
};

static CollisionStats collide_and_kessler(std::vector<Body> &bodies, const SimConfig &cfg,
                                          std::mt19937 &rng, uint32_t &next_id) {
  CollisionStats stats;

  // Build cell lists: map cell_key -> vector of indices
  std::unordered_map<uint64_t, std::vector<int>> cell;
  cell.reserve(bodies.size() * 2);

  const float cs = cfg.cell_size;

  for (int i = 0; i < (int)bodies.size(); i++) {
    int cx = (int)std::floor(bodies[i].x / cs);
    int cy = (int)std::floor(bodies[i].y / cs);
    cell[cell_key(cx, cy)].push_back(i);
  }

  std::vector<char> dead(bodies.size(), 0);
  std::vector<Body> spawned;
  spawned.reserve(1024);

  std::uniform_real_distribution<float> dv(-120.f, 120.f);

  auto check_cell_pair = [&](const std::vector<int> &a, const std::vector<int> &b, bool same){
    for (size_t ii = 0; ii < a.size(); ii++) {
      int i = a[ii];
      if (dead[i]) continue;
      size_t jj0 = same ? (ii + 1) : 0;
      for (size_t jj = jj0; jj < b.size(); jj++) {
        int j = b[jj];
        if (dead[j]) continue;

        stats.checked_pairs++;

        float dx = bodies[i].x - bodies[j].x;
        float dy = bodies[i].y - bodies[j].y;
        float rr = cfg.coll_dist + bodies[i].r + bodies[j].r;
        if (dx*dx + dy*dy <= rr*rr) {
          stats.collisions++;
          dead[i] = dead[j] = 1;

          if (cfg.kessler) {
            for (int k = 0; k < cfg.debris_per_collision; k++) {
              Body nb = bodies[i];
              nb.id = next_id++;
              nb.r = 0.8f;
              nb.vx += dv(rng);
              nb.vy += dv(rng);
              spawned.push_back(nb);
            }
          }
          break;
        }
      }
    }
  };

  // For each occupied cell, check against itself and neighbors (+8)
  for (const auto &kv : cell) {
    uint64_t key = kv.first;
    int cx = (int)(int32_t)(key >> 32);
    int cy = (int)(int32_t)(key & 0xffffffffu);

    const auto &here = kv.second;
    // self
    check_cell_pair(here, here, true);

    // neighbors with ordering to avoid double-checking
    const int dxs[4] = {1, 1, 0, -1};
    const int dys[4] = {0, 1, 1, 1};
    for (int t = 0; t < 4; t++) {
      uint64_t nk = cell_key(cx + dxs[t], cy + dys[t]);
      auto it = cell.find(nk);
      if (it != cell.end()) {
        check_cell_pair(here, it->second, false);
      }
    }
  }

  // Compact bodies: remove dead
  size_t w = 0;
  for (size_t i = 0; i < bodies.size(); i++) {
    if (!dead[i]) bodies[w++] = bodies[i];
  }
  bodies.resize(w);

  // Append spawned debris
  bodies.insert(bodies.end(), spawned.begin(), spawned.end());
  return stats;
}

static void enforce_bounds(std::vector<Body> &bodies, const SimConfig &cfg, std::mt19937 &rng) {
  const float kill2 = cfg.kill_radius * cfg.kill_radius;
  const float max2  = cfg.world_radius * cfg.world_radius;

  for (auto &b : bodies) {
    float d2 = b.x*b.x + b.y*b.y;
    if (d2 < kill2 || d2 > max2) {
      // respawn at a fresh orbit but keep id
      b = spawn_orbiter(rng, cfg, b.id);
    }
  }
}

static double ms_since(const std::chrono::high_resolution_clock::time_point &t0) {
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static void usage() {
  std::cout <<
  "Usage:\n"
  "  ./sat_swarm --n 20000 --steps 2000 --threads 4\n"
  "Options:\n"
  "  --n N                 number of bodies (default 20000)\n"
  "  --steps S             simulation steps (default 4000)\n"
  "  --threads T           OpenMP threads (default: max)\n"
  "  --no-omp              disable OpenMP\n"
  "  --bench               run serial vs OpenMP benchmark\n"
  "  --no-kessler          disable debris spawning on collision\n"
  "  --debris K            debris per collision (default 1)\n"
  "  --no-collisions       disable collision detection entirely (best for OpenMP speedup)\n";
}

static void run_sim(int N, int steps, bool use_openmp, const SimConfig &cfg_in, bool collisions_on) {
  SimConfig cfg = cfg_in;
  std::mt19937 rng(1337);
  uint32_t next_id = 1;

  std::vector<Body> bodies;
  bodies.reserve((size_t)N * 2);
  for (int i = 0; i < N; i++) bodies.push_back(spawn_orbiter(rng, cfg, next_id++));

  auto t0 = std::chrono::high_resolution_clock::now();
  uint64_t total_collisions = 0;
  uint64_t total_pairs = 0;

  for (int s = 1; s <= steps; s++) {
    for (int k = 0; k < cfg.substeps; k++) {
      step_bodies(bodies, cfg, use_openmp);
      enforce_bounds(bodies, cfg, rng);

      if (collisions_on) {
        auto st = collide_and_kessler(bodies, cfg, rng, next_id);
        total_collisions += st.collisions;
        total_pairs += st.checked_pairs;
      }
    }

    if (s % 200 == 0 || s == steps) {
      double ms = ms_since(t0);
      double steps_per_s = (double)s / (ms / 1000.0);
      std::cout
        << "step " << s
        << " | N=" << bodies.size()
        << " | steps/s=" << steps_per_s;

      if (collisions_on) {
        std::cout << " | collisions=" << total_collisions
                  << " | checked_pairs=" << total_pairs;
      }
      std::cout << "\n";
    }
  }

  double total_ms = ms_since(t0);
  std::cout << "Done. total=" << total_ms << " ms, avg=" << (total_ms/steps) << " ms/step\n";
}

int main(int argc, char** argv) {
  int N = 20000;
  int steps = 4000;
  bool use_openmp = true;
  bool bench = false;
  bool collisions_on = true;

  SimConfig cfg;

#ifdef _OPENMP
  int threads = omp_get_max_threads();
#else
  int threads = 1;
#endif

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--n" && i+1 < argc) N = std::atoi(argv[++i]);
    else if (a == "--steps" && i+1 < argc) steps = std::atoi(argv[++i]);
    else if (a == "--threads" && i+1 < argc) threads = std::atoi(argv[++i]);
    else if (a == "--no-omp") use_openmp = false;
    else if (a == "--bench") bench = true;
    else if (a == "--no-kessler") cfg.kessler = false;
    else if (a == "--debris" && i+1 < argc) cfg.debris_per_collision = std::atoi(argv[++i]);
    else if (a == "--no-collisions") collisions_on = false;
    else if (a == "--help") { usage(); return 0; }
  }

#ifdef _OPENMP
  omp_set_num_threads(threads);
#endif

  if (!bench) {
#ifdef _OPENMP
    std::cout << "OpenMP: " << (use_openmp ? "ON" : "OFF") << " | threads=" << threads << "\n";
#else
    std::cout << "OpenMP not enabled at compile time.\n";
    use_openmp = false;
#endif
    if (!collisions_on) std::cout << "Collisions: OFF\n";
    run_sim(N, steps, use_openmp, cfg, collisions_on);
    return 0;
  }

  // benchmark: serial vs omp
  std::cout << "Benchmarking serial vs OpenMP\n";
  if (!collisions_on) std::cout << "Collisions: OFF (isolating integrator scaling)\n";

  auto t0 = std::chrono::high_resolution_clock::now();
  run_sim(N, steps, false, cfg, collisions_on);
  double serial_ms = ms_since(t0);

  t0 = std::chrono::high_resolution_clock::now();
  run_sim(N, steps, true, cfg, collisions_on);
  double omp_ms = ms_since(t0);

  std::cout << "SERIAL: " << serial_ms << " ms\n";
  std::cout << " OPENMP: " << omp_ms  << " ms\n";
  std::cout << "Speedup: " << (serial_ms / omp_ms) << "x\n";
  return 0;
}


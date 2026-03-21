/*
 * Astar Island Simulator v4
 *
 * Key insight from R1 analysis:
 * - Settlement probability = f(distance_to_nearest_settlement)
 * - ~25% at dist 1-3, ~10% at dist 5, ~3% at dist 8+
 * - Empty and Forest behave identically (same P(settlement) at same distance)
 * - 41% of initial settlements die → become Empty (93%) or Ruin (7%)
 * - Only 1-2.5% of cells change argmax from initial
 * - But PROBABILITIES are spread — GT has high entropy near settlements
 *
 * Build: powershell -Command "cd astar-island; & .\build.bat"
 * Run:   astar-island\sim.exe data\astar_round1.json --runs 10000 --seed-index 0 --output astar-island\data\sim_latest.json
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <string>
#include <chrono>
#include <algorithm>

#include "../cpp_solver/json.hpp"
using json = nlohmann::json;

enum Class : uint8_t {
    EMPTY = 0, SETTLEMENT = 1, PORT = 2, RUIN = 3, FOREST = 4, MOUNTAIN = 5, NUM_CLASSES = 6
};

constexpr int CODE_OCEAN = 10, CODE_PLAINS = 11;
int code_to_class(int code) {
    if (code == CODE_OCEAN || code == CODE_PLAINS || code == 0) return EMPTY;
    if (code >= 1 && code <= 5) return code;
    return EMPTY;
}

constexpr int W = 40, H = 40, YEARS = 50;
constexpr int DX[] = {-1, 1, 0, 0, -1, -1, 1, 1};
constexpr int DY[] = {0, 0, -1, 1, -1, 1, -1, 1};

inline bool in_bounds(int y, int x) { return y >= 0 && y < H && x >= 0 && x < W; }

struct Params {
    // Expansion — settlements spread to nearby cells
    // GT: 0.5% of empty cells become settlement. With ~30 settlements checking ~40 cells each,
    // that's 1200 checks. 0.5% of 1186 empty = 6 new settlements. So prob = 6/1200 ≈ 0.005 per check per 50 years = 0.0001 per year
    double expansion_prob = 0.0002;  // per cell per year PER settlement checking it
    int expansion_range = 4;         // manhattan distance
    double expansion_decay = 0.5;    // multiplier per distance step

    // Settlement survival
    double base_survival = 0.985;    // per year survival rate (0.985^50 ≈ 0.47, close to 57% GT)
    double forest_food_bonus = 0.003; // per adjacent forest, adds to survival
    double isolation_penalty = 0.005; // per unit of distance to nearest other settlement

    // Port
    double port_develop = 0.01;
    double port_survival_bonus = 0.002;

    // Death outcome
    double death_to_ruin = 0.07;     // 7% ruin, 93% empty (from GT)
    double death_to_forest = 0.15;   // of the empty deaths, some become forest

    // Environment
    double ruin_to_forest = 0.03;
    double ruin_to_empty = 0.08;
};

struct Settlement {
    int x, y;
    bool alive;
    bool has_port;
    int owner_id;
};

using Grid = std::array<std::array<uint8_t, W>, H>;

class Simulator {
public:
    Grid initial_grid;
    bool ocean[H][W];
    std::vector<Settlement> initial_settlements;
    Params params;

    void load_seed(const json& state) {
        auto& grid_data = state["grid"];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                int code = grid_data[y][x].get<int>();
                initial_grid[y][x] = code_to_class(code);
                ocean[y][x] = (code == CODE_OCEAN);
            }

        initial_settlements.clear();
        if (state.contains("settlements"))
            for (auto& s : state["settlements"]) {
                Settlement st;
                st.x = s["x"].get<int>();
                st.y = s["y"].get<int>();
                st.has_port = s.value("has_port", false);
                st.alive = s.value("alive", true);
                st.owner_id = (int)initial_settlements.size();
                initial_settlements.push_back(st);
            }
    }

    bool has_ocean_neighbor(int y, int x) {
        for (int d = 0; d < 4; d++) {
            int ny = y + DY[d], nx = x + DX[d];
            if (in_bounds(ny, nx) && ocean[ny][nx]) return true;
        }
        return false;
    }

    int count_forest_neighbors(const Grid& grid, int y, int x) {
        int c = 0;
        for (int d = 0; d < 4; d++) {
            int ny = y + DY[d], nx = x + DX[d];
            if (in_bounds(ny, nx) && grid[ny][nx] == FOREST) c++;
        }
        return c;
    }

    int nearest_settlement_dist(const std::vector<Settlement>& settlements, int y, int x, int exclude = -1) {
        int best = 999;
        for (int i = 0; i < (int)settlements.size(); i++) {
            if (i == exclude || !settlements[i].alive) continue;
            int d = abs(y - settlements[i].y) + abs(x - settlements[i].x);
            if (d < best) best = d;
        }
        return best;
    }

    Grid run_once(std::mt19937& rng) {
        Grid grid = initial_grid;
        auto settlements = initial_settlements;
        std::uniform_real_distribution<double> U(0.0, 1.0);

        for (int year = 0; year < YEARS; year++) {
            // === SURVIVAL ===
            for (int i = 0; i < (int)settlements.size(); i++) {
                auto& s = settlements[i];
                if (!s.alive) continue;

                double survival = params.base_survival;
                survival += count_forest_neighbors(grid, s.y, s.x) * params.forest_food_bonus;
                if (s.has_port) survival += params.port_survival_bonus;

                int iso = nearest_settlement_dist(settlements, s.y, s.x, i);
                survival -= std::min(iso * params.isolation_penalty, 0.03);

                survival = std::max(0.5, std::min(0.999, survival));

                if (U(rng) > survival) {
                    s.alive = false;
                    if (U(rng) < params.death_to_ruin) {
                        grid[s.y][s.x] = RUIN;
                    } else if (U(rng) < params.death_to_forest) {
                        grid[s.y][s.x] = FOREST;
                    } else {
                        grid[s.y][s.x] = EMPTY;
                    }
                }
            }

            // === EXPANSION ===
            std::vector<Settlement> new_settlements;
            for (auto& s : settlements) {
                if (!s.alive) continue;

                for (int dy = -params.expansion_range; dy <= params.expansion_range; dy++) {
                    for (int dx = -params.expansion_range; dx <= params.expansion_range; dx++) {
                        int dist = abs(dy) + abs(dx);
                        if (dist == 0 || dist > params.expansion_range) continue;
                        int ny = s.y + dy, nx = s.x + dx;
                        if (!in_bounds(ny, nx) || ocean[ny][nx]) continue;
                        if (grid[ny][nx] != EMPTY && grid[ny][nx] != FOREST) continue;

                        double prob = params.expansion_prob * pow(params.expansion_decay, dist - 1);
                        if (U(rng) > prob) continue;

                        // Check not already a settlement
                        bool occupied = false;
                        for (auto& other : settlements)
                            if (other.alive && other.x == nx && other.y == ny) { occupied = true; break; }
                        for (auto& other : new_settlements)
                            if (other.x == nx && other.y == ny) { occupied = true; break; }
                        if (occupied) continue;

                        Settlement ns;
                        ns.x = nx; ns.y = ny;
                        ns.alive = true;
                        ns.has_port = has_ocean_neighbor(ny, nx) && U(rng) < 0.3;
                        ns.owner_id = s.owner_id;
                        grid[ny][nx] = ns.has_port ? PORT : SETTLEMENT;
                        new_settlements.push_back(ns);
                    }
                }
            }
            settlements.insert(settlements.end(), new_settlements.begin(), new_settlements.end());

            // === PORT DEVELOPMENT ===
            for (auto& s : settlements) {
                if (!s.alive || s.has_port) continue;
                if (has_ocean_neighbor(s.y, s.x) && U(rng) < params.port_develop) {
                    s.has_port = true;
                    grid[s.y][s.x] = PORT;
                }
            }

            // === ENVIRONMENT ===
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    if (grid[y][x] == RUIN) {
                        if (U(rng) < params.ruin_to_forest) grid[y][x] = FOREST;
                        else if (U(rng) < params.ruin_to_empty) grid[y][x] = EMPTY;
                    }
                }
        }

        return grid;
    }

    std::vector<std::vector<std::array<double, NUM_CLASSES>>> monte_carlo(int n_runs) {
        std::vector<std::vector<std::array<int, NUM_CLASSES>>> counts(
            H, std::vector<std::array<int, NUM_CLASSES>>(W, {0,0,0,0,0,0}));

        #pragma omp parallel
        {
            std::mt19937 rng(std::random_device{}());
            std::vector<std::vector<std::array<int, NUM_CLASSES>>> local(
                H, std::vector<std::array<int, NUM_CLASSES>>(W, {0,0,0,0,0,0}));

            #pragma omp for schedule(dynamic)
            for (int run = 0; run < n_runs; run++) {
                Grid result = run_once(rng);
                for (int y = 0; y < H; y++)
                    for (int x = 0; x < W; x++)
                        local[y][x][result[y][x]]++;
            }

            #pragma omp critical
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    for (int c = 0; c < NUM_CLASSES; c++)
                        counts[y][x][c] += local[y][x][c];
        }

        const double FLOOR = 0.005;
        std::vector<std::vector<std::array<double, NUM_CLASSES>>> probs(
            H, std::vector<std::array<double, NUM_CLASSES>>(W));

        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                double total = 0;
                for (int c = 0; c < NUM_CLASSES; c++) {
                    probs[y][x][c] = std::max(FLOOR, (double)counts[y][x][c] / n_runs);
                    total += probs[y][x][c];
                }
                for (int c = 0; c < NUM_CLASSES; c++)
                    probs[y][x][c] /= total;
            }

        return probs;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: sim.exe <round_data.json> [--runs N] [--seed-index I] [--output FILE]" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    int n_runs = 10000;
    int seed_index = -1;
    std::string output_file = "";

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--runs" && i+1 < argc) n_runs = std::stoi(argv[++i]);
        else if (arg == "--seed-index" && i+1 < argc) seed_index = std::stoi(argv[++i]);
        else if (arg == "--output" && i+1 < argc) output_file = argv[++i];
    }

    std::ifstream f(input_file);
    json data; f >> data; f.close();

    int n_seeds = data["initial_states"].size();
    int start = (seed_index >= 0) ? seed_index : 0;
    int end = (seed_index >= 0) ? seed_index + 1 : n_seeds;

    std::cerr << "Seeds: " << n_seeds << ", runs: " << n_runs << std::endl;

    json output;
    for (int si = start; si < end; si++) {
        auto t0 = std::chrono::high_resolution_clock::now();

        Simulator sim;
        sim.load_seed(data["initial_states"][si]);
        auto probs = sim.monte_carlo(n_runs);

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        int n_changed = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                int init = sim.initial_grid[y][x];
                int pred = 0; double mx = 0;
                for (int c = 0; c < NUM_CLASSES; c++)
                    if (probs[y][x][c] > mx) { mx = probs[y][x][c]; pred = c; }
                if (pred != init) n_changed++;
            }

        std::cerr << "Seed " << si << ": " << elapsed << "s, " << n_changed << " changed" << std::endl;

        json tensor = json::array();
        for (int y = 0; y < H; y++) {
            json row = json::array();
            for (int x = 0; x < W; x++) {
                json cell = json::array();
                for (int c = 0; c < NUM_CLASSES; c++)
                    cell.push_back(probs[y][x][c]);
                row.push_back(cell);
            }
            tensor.push_back(row);
        }
        output["predictions"][std::to_string(si)] = tensor;
    }

    if (output_file.empty()) std::cout << output.dump() << std::endl;
    else { std::ofstream of(output_file); of << output.dump(); of.close();
           std::cerr << "Written to " << output_file << std::endl; }
    return 0;
}

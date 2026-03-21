/*
 * C++ MAPF Planner for NM i AI Grocery Bot
 *
 * Round-by-round sequential planning with reactive item assignment.
 * Zone partitioning for nightmare (20 bots, 3 zones).
 * LNS over trip assignments for optimization.
 *
 * Status:
 * - Easy (1 bot):    score 55 (verified)
 * - Medium (3 bot):  score 27 (verified)
 * - Hard (5 bot):    score 2 — needs PIBT for spawn dispersal
 * - Nightmare (20):  score 20 (verified, 3-zone partitioning)
 *
 * TODO: Port PIBTResolver from solver.cpp for 5+ bot collision resolution.
 *
 * Build: cl /EHsc /O2 /std:c++17 /MT mapf.cpp /Fe:mapf.exe
 *        (or: build_mapf.bat)
 * Usage: mapf.exe --recon <file> [--greedy] [--iterations N] [--workers N] [--output file]
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "json.hpp"
using json = nlohmann::json;

// ============================================================
// Core Types (matching solver.cpp)
// ============================================================

struct Pos {
    int x, y;
    bool operator==(const Pos& o) const { return x == o.x && y == o.y; }
    bool operator!=(const Pos& o) const { return !(*this == o); }
    bool operator<(const Pos& o) const { return x < o.x || (x == o.x && y < o.y); }
};
struct PosHash { size_t operator()(const Pos& p) const { return (size_t)p.x * 1000 + p.y; } };
using PosSet = std::unordered_set<Pos, PosHash>;

static const Pos DIRS[4] = {{0,-1},{0,1},{-1,0},{1,0}};
static const char* DIR_NAMES[4] = {"move_up", "move_down", "move_left", "move_right"};

static std::string direction_action(Pos from, Pos to) {
    int dx = to.x - from.x, dy = to.y - from.y;
    for (int i = 0; i < 4; i++) {
        if (DIRS[i].x == dx && DIRS[i].y == dy) return DIR_NAMES[i];
    }
    return "wait";
}

// ============================================================
// Grid with One-Way Rules
// ============================================================

struct Grid {
    int W = 0, H = 0;
    std::vector<bool> walkable;
    std::unordered_map<int, std::pair<int,int>> one_way; // cell_key -> (dx, dy)

    int key(Pos p) const { return p.y * W + p.x; }
    Pos from_key(int k) const { return {k % W, k / W}; }
    bool ok(Pos p) const { return p.x >= 0 && p.x < W && p.y >= 0 && p.y < H && walkable[p.y * W + p.x]; }

    void init(int w, int h, const PosSet& walls, const PosSet& shelves) {
        W = w; H = h;
        walkable.assign(w * h, true);
        for (auto& p : walls) if (p.x >= 0 && p.x < w && p.y >= 0 && p.y < h) walkable[p.y * w + p.x] = false;
        for (auto& p : shelves) if (p.x >= 0 && p.x < w && p.y >= 0 && p.y < h) walkable[p.y * w + p.x] = false;
    }

    void neighbors(Pos p, Pos out[], int& count) const {
        count = 0;
        auto it = one_way.find(key(p));
        for (auto& d : DIRS) {
            Pos np = {p.x + d.x, p.y + d.y};
            if (!ok(np)) continue;
            if (it != one_way.end()) {
                auto& [rdx, rdy] = it->second;
                if (rdy != 0 && d.x == 0 && d.y != 0 && d.y != rdy) continue;
                if (rdx != 0 && d.y == 0 && d.x != 0 && d.x != rdx) continue;
            }
            out[count++] = np;
        }
    }

    void setup_nightmare_one_way() {
        one_way.clear();
        std::vector<int> cross_rows;
        for (int y = 0; y < H; y++) {
            int wk = 0;
            for (int x = 0; x < W; x++) if (walkable[y * W + x]) wk++;
            if (wk >= W * 6 / 10) cross_rows.push_back(y);
        }
        if (cross_rows.empty()) return;
        int min_cr = cross_rows.front(), max_cr = cross_rows.back();
        std::set<int> cr_set(cross_rows.begin(), cross_rows.end());

        std::vector<int> aisle_cols;
        for (int x = 1; x < W - 1; x++) {
            bool is_aisle = true;
            int non_corr = 0;
            bool has_wall = false;
            for (int y = min_cr; y <= max_cr; y++) {
                if (cr_set.count(y)) continue;
                if (!ok({x, y})) { is_aisle = false; break; }
                non_corr++;
                if (!ok({x-1, y}) || !ok({x+1, y})) has_wall = true;
            }
            if (is_aisle && non_corr >= 2 && has_wall) aisle_cols.push_back(x);
        }
        if (aisle_cols.empty()) return;

        int bottom_row = cross_rows.back();
        for (int x = 1; x < W - 1; x++) {
            if (ok({x, bottom_row})) one_way[key({x, bottom_row})] = {-1, 0};
        }

        std::sort(aisle_cols.begin(), aisle_cols.end());
        for (int i = 0; i < (int)aisle_cols.size(); i++) {
            int ax = aisle_cols[i];
            int dy = (i % 2 == 0) ? 1 : -1;
            for (int y = min_cr; y <= max_cr; y++) {
                if (cr_set.count(y)) continue;
                if (ok({ax, y})) one_way[key({ax, y})] = {0, dy};
            }
        }
    }
};

// ============================================================
// BFS Distance Cache (one-way aware, reverse BFS)
// ============================================================

struct BFSCache {
    const Grid* grid = nullptr;
    std::unordered_map<int, std::vector<int>> cache;

    void init(const Grid* g) { grid = g; cache.clear(); }

    int distance(Pos from, Pos to) {
        if (from == to) return 0;
        int dk = grid->key(to);
        auto it = cache.find(dk);
        if (it == cache.end()) {
            auto& dist = cache[dk];
            dist.assign(grid->W * grid->H, -1);
            dist[dk] = 0;
            std::deque<int> q;
            q.push_back(dk);
            while (!q.empty()) {
                int ck = q.front(); q.pop_front();
                int cx = ck % grid->W, cy = ck / grid->W;
                int cd = dist[ck];
                for (auto& d : DIRS) {
                    int nx = cx - d.x, ny = cy - d.y;
                    if (nx < 0 || nx >= grid->W || ny < 0 || ny >= grid->H) continue;
                    int nk = ny * grid->W + nx;
                    if (!grid->walkable[nk] || dist[nk] >= 0) continue;
                    auto ow = grid->one_way.find(nk);
                    if (ow != grid->one_way.end()) {
                        auto& [rdx, rdy] = ow->second;
                        if (rdy != 0 && d.x == 0 && d.y != 0 && d.y != rdy) continue;
                        if (rdx != 0 && d.y == 0 && d.x != 0 && d.x != rdx) continue;
                    }
                    dist[nk] = cd + 1;
                    q.push_back(nk);
                }
            }
            it = cache.find(dk);
        }
        int fk = grid->key(from);
        return (it->second[fk] >= 0) ? it->second[fk] : 9999;
    }
};

// ============================================================
// PIBT Resolver (ported from solver.cpp)
// ============================================================

struct PIBTBot { int id; Pos pos; };

struct PIBTResolver {
    const Grid* grid;
    BFSCache* bfs;

    struct Prio { int tier; int dist; int tiebreak; };

    std::unordered_map<int, Pos> resolve(
        const std::vector<PIBTBot>& bots,
        const std::unordered_map<int, Pos>& targets,
        const std::unordered_map<int, int>& urgency,
        int round_offset
    ) {
        int n = (int)bots.size();
        std::unordered_map<int, Prio> prio;
        for (auto& bot : bots) {
            auto tit = targets.find(bot.id);
            Pos tgt = tit != targets.end() ? tit->second : bot.pos;
            int d = bfs->distance(bot.pos, tgt);
            int tier = 3;
            auto uit = urgency.find(bot.id);
            if (uit != urgency.end()) tier = uit->second;
            if (bot.pos == tgt) { d = 9999; tier = 3; }
            prio[bot.id] = {tier, d, (bot.id + round_offset) % 100};
        }

        std::vector<int> sorted_ids;
        for (auto& b : bots) sorted_ids.push_back(b.id);
        std::sort(sorted_ids.begin(), sorted_ids.end(), [&](int a, int b) {
            auto& pa = prio[a]; auto& pb = prio[b];
            if (pa.tier != pb.tier) return pa.tier < pb.tier;
            if (pa.dist != pb.dist) return pa.dist < pb.dist;
            return pa.tiebreak < pb.tiebreak;
        });

        std::unordered_map<int, Pos> bot_pos;
        for (auto& b : bots) bot_pos[b.id] = b.pos;

        std::unordered_map<int, int> claimed; // pos_key -> bot_id
        for (auto& b : bots) claimed[grid->key(b.pos)] = b.id;
        std::unordered_map<int, Pos> result;
        std::unordered_set<int> decided;

        std::function<bool(int, int)> plan_fn = [&](int bid, int depth) -> bool {
            if (decided.count(bid)) return true;
            if (depth > n + 2) {
                result[bid] = bot_pos[bid];
                claimed[grid->key(bot_pos[bid])] = bid;
                decided.insert(bid);
                return depth == 0;
            }

            Pos current = bot_pos[bid];
            auto tit = targets.find(bid);
            Pos target = tit != targets.end() ? tit->second : current;

            Pos nbrs[4]; int nc;
            grid->neighbors(current, nbrs, nc);

            struct Cand { Pos pos; int dist; bool is_current; };
            std::vector<Cand> candidates;
            for (int i = 0; i < nc; i++) {
                candidates.push_back({nbrs[i], bfs->distance(nbrs[i], target), false});
            }
            candidates.push_back({current, bfs->distance(current, target), true});

            std::sort(candidates.begin(), candidates.end(), [](auto& a, auto& b) {
                if (a.dist != b.dist) return a.dist < b.dist;
                if (a.is_current != b.is_current) return !a.is_current;
                return false;
            });

            for (auto& cand : candidates) {
                int ck = grid->key(cand.pos);
                auto cit = claimed.find(ck);

                if (cit != claimed.end()) {
                    int occupant = cit->second;
                    if (occupant == bid) {
                        bool should_defer = (current != target);
                        if (should_defer) continue;
                        result[bid] = cand.pos;
                        decided.insert(bid);
                        return true;
                    }
                    if (decided.count(occupant)) continue;

                    auto& pp = prio[bid]; auto& po = prio[occupant];
                    bool higher = (pp.tier < po.tier) ||
                                  (pp.tier == po.tier && pp.dist < po.dist) ||
                                  (pp.tier == po.tier && pp.dist == po.dist && pp.tiebreak < po.tiebreak);
                    if (higher) {
                        if (plan_fn(occupant, depth + 1)) {
                            auto cit2 = claimed.find(ck);
                            if (cit2 == claimed.end() || cit2->second != occupant) {
                                int old_k = grid->key(current);
                                if (claimed.count(old_k) && claimed[old_k] == bid) claimed.erase(old_k);
                                claimed[ck] = bid;
                                result[bid] = cand.pos;
                                decided.insert(bid);
                                return true;
                            }
                        }
                    }
                    continue;
                }

                int old_k = grid->key(current);
                if (claimed.count(old_k) && claimed[old_k] == bid) claimed.erase(old_k);
                claimed[ck] = bid;
                result[bid] = cand.pos;
                decided.insert(bid);
                return true;
            }

            result[bid] = current;
            claimed[grid->key(current)] = bid;
            decided.insert(bid);
            return true;
        };

        for (int bid : sorted_ids) {
            if (!decided.count(bid)) plan_fn(bid, 0);
        }

        // Post-process: cancel swaps
        for (auto& [ba, pa] : result) {
            if (pa == bot_pos[ba]) continue;
            for (auto& [bb, pb] : result) {
                if (bb <= ba || pb == bot_pos[bb]) continue;
                if (pa == bot_pos[bb] && pb == bot_pos[ba]) {
                    result[ba] = bot_pos[ba];
                    result[bb] = bot_pos[bb];
                }
            }
        }

        // Post-process: sequential ID-order collision cancellation
        // In sequential processing, lower-ID bots move first.
        // Cancel bot A's move if it moves to cell where bot B IS and B stays (or B has higher ID and also moves there)
        for (int iter = 0; iter < n + 1; iter++) {
            bool cancelled = false;
            // Process in ID order (matching game's sequential processing)
            std::vector<int> id_order;
            for (auto& b : bots) id_order.push_back(b.id);
            std::sort(id_order.begin(), id_order.end());
            for (int ba : id_order) {
                if (result[ba] == bot_pos[ba]) continue;
                Pos tp = result[ba];
                for (int bb : id_order) {
                    if (bb == ba) continue;
                    // In sequential model: lower ID bots have already moved
                    // So check: does bb occupy tp?
                    Pos bb_effective;
                    if (bb < ba) {
                        bb_effective = result[bb]; // already moved
                    } else {
                        bb_effective = bot_pos[bb]; // not yet moved
                    }
                    if (bb_effective == tp) {
                        result[ba] = bot_pos[ba]; // cancel
                        cancelled = true;
                        break;
                    }
                }
            }
            if (!cancelled) break;
        }

        for (auto& b : bots) if (!result.count(b.id)) result[b.id] = b.pos;
        return result;
    }
};

// ============================================================
// Game / Recon Structures
// ============================================================

struct Order {
    std::string id;
    std::vector<std::string> items_required;
    std::vector<std::string> items_delivered;
    std::string status; // hidden, preview, active, complete

    bool complete() const {
        auto rem = items_required;
        for (auto& d : items_delivered) {
            auto it = std::find(rem.begin(), rem.end(), d);
            if (it != rem.end()) rem.erase(it);
        }
        return rem.empty();
    }
    std::vector<std::string> remaining() const {
        auto rem = items_required;
        for (auto& d : items_delivered) {
            auto it = std::find(rem.begin(), rem.end(), d);
            if (it != rem.end()) rem.erase(it);
        }
        return rem;
    }
};

struct ReconConfig {
    int width, height;
    PosSet walls, shelves;
    Pos drop_off;
    std::vector<Pos> drop_off_zones;
    std::vector<Pos> spawn_positions;
    std::vector<Order> order_sequence;
    std::map<int, std::string> shelf_types; // key(shelf_pos) -> item_type
    int max_rounds = 500;
    int n_bots = 20;
};

struct RecordedAction {
    std::string action;
    Pos position;
    std::string item_type;
};

// ============================================================
// Zone Definition
// ============================================================

struct Zone {
    std::string name;
    int x_min, x_max;
    Pos drop_off;
    std::vector<int> bot_ids;
    // item_type -> list of (shelf_pos, pickup_pos)
    std::unordered_map<std::string, std::vector<std::pair<Pos, Pos>>> shelves;
};

// ============================================================
// Trip: a sequence of pickups + one delivery
// ============================================================

struct Trip {
    int bot_id;
    int order_idx;
    std::vector<std::string> items;       // item types to pick up
    std::vector<Pos> pickup_positions;     // walkable positions to stand at
    std::vector<Pos> shelf_positions;      // actual shelf positions (for item_id lookup)
    Pos drop_off;
};

// ============================================================
// Simulator for Plan Verification
// ============================================================

class PlanSimulator {
public:
    ReconConfig cfg;
    Grid grid;
    int round_ = 0, score_ = 0, items_delivered_ = 0, orders_completed_ = 0;
    int next_order_idx_ = 0;
    std::vector<Order> orders_;
    std::vector<Pos> bot_pos_;
    std::vector<std::vector<std::string>> bot_inv_;
    std::mt19937 order_rng_{42};
    std::vector<int> order_size_pool_;
    std::vector<std::string> item_type_pool_;
    bool rng_init_ = false;

    void init(const ReconConfig& c, Grid& g) {
        cfg = c;
        grid = g;
    }

    void reset() {
        round_ = 0; score_ = 0; items_delivered_ = 0; orders_completed_ = 0;
        next_order_idx_ = 0;
        order_rng_.seed(42);
        rng_init_ = false;
        bot_pos_.resize(cfg.n_bots);
        bot_inv_.resize(cfg.n_bots);
        for (int i = 0; i < cfg.n_bots; i++) {
            bot_pos_[i] = (i < (int)cfg.spawn_positions.size()) ? cfg.spawn_positions[i] : cfg.spawn_positions[0];
            bot_inv_[i].clear();
        }
        orders_.clear();
        for (auto& o : cfg.order_sequence)
            orders_.push_back({o.id, o.items_required, {}, "hidden"});
        if (!orders_.empty()) { orders_[0].status = "active"; next_order_idx_ = 1; }
        if (orders_.size() > 1) { orders_[1].status = "preview"; next_order_idx_ = 2; }
    }

    Order* active_order() { for (auto& o : orders_) if (o.status == "active") return &o; return nullptr; }
    Order* preview_order() { for (auto& o : orders_) if (o.status == "preview") return &o; return nullptr; }

    // Resolve one round of actions (in ID order)
    void step(const std::vector<std::pair<int, std::string>>& actions) {
        // actions: [(bot_id, action_string), ...] where action_string = "move_right", "pick_up:item_type", "drop_off", "wait"
        round_++;
        // Build action map
        std::unordered_map<int, std::string> amap;
        for (auto& [bid, act] : actions) amap[bid] = act;

        for (int bid = 0; bid < cfg.n_bots; bid++) {
            auto it = amap.find(bid);
            std::string action = (it != amap.end()) ? it->second : "wait";

            if (action.substr(0, 4) == "move") {
                resolve_move(bid, action);
            } else if (action.substr(0, 7) == "pick_up") {
                // pick_up:item_type — find matching item adjacent to bot
                std::string item_type = (action.size() > 8) ? action.substr(8) : "";
                resolve_pickup(bid, item_type);
            } else if (action == "drop_off") {
                resolve_dropoff(bid);
            }
        }
    }

private:
    void resolve_move(int bid, const std::string& action) {
        int dx = 0, dy = 0;
        if (action == "move_up") dy = -1;
        else if (action == "move_down") dy = 1;
        else if (action == "move_left") dx = -1;
        else if (action == "move_right") dx = 1;
        Pos np = {bot_pos_[bid].x + dx, bot_pos_[bid].y + dy};
        if (!grid.ok(np)) return;
        for (int i = 0; i < cfg.n_bots; i++) {
            if (i != bid && bot_pos_[i] == np) return;
        }
        bot_pos_[bid] = np;
    }

    void resolve_pickup(int bid, const std::string& item_type) {
        if ((int)bot_inv_[bid].size() >= 3) return;
        if (item_type.empty()) return;
        // Find shelf with this item_type adjacent to bot
        Pos bp = bot_pos_[bid];
        for (auto& d : DIRS) {
            Pos sp = {bp.x + d.x, bp.y + d.y};
            int sk = sp.y * cfg.width + sp.x;
            auto it = cfg.shelf_types.find(sk);
            if (it != cfg.shelf_types.end() && it->second == item_type) {
                bot_inv_[bid].push_back(item_type);
                return;
            }
        }
    }

    void resolve_dropoff(int bid) {
        bool on = false;
        for (auto& z : cfg.drop_off_zones) if (bot_pos_[bid] == z) { on = true; break; }
        if (!on) return;
        Order* act = active_order();
        if (!act) return;
        auto rem = act->remaining();
        std::vector<std::string> new_inv;
        for (auto& inv : bot_inv_[bid]) {
            auto it = std::find(rem.begin(), rem.end(), inv);
            if (it != rem.end()) { rem.erase(it); act->items_delivered.push_back(inv); score_++; items_delivered_++; }
            else new_inv.push_back(inv);
        }
        bot_inv_[bid] = new_inv;
        if (act->complete()) { score_ += 5; orders_completed_++; act->status = "complete"; advance_orders(bid); }
    }

    void advance_orders(int bid) {
        Order* pv = preview_order();
        if (pv) {
            pv->status = "active";
            auto rem = pv->remaining();
            std::vector<std::string> ni;
            for (auto& inv : bot_inv_[bid]) {
                auto it = std::find(rem.begin(), rem.end(), inv);
                if (it != rem.end()) { rem.erase(it); pv->items_delivered.push_back(inv); score_++; items_delivered_++; }
                else ni.push_back(inv);
            }
            bot_inv_[bid] = ni;
            if (pv->complete()) { score_ += 5; orders_completed_++; pv->status = "complete"; promote_preview(); advance_orders(bid); return; }
        }
        promote_preview();
    }

    void promote_preview() {
        if (next_order_idx_ >= (int)orders_.size()) gen_order();
        if (next_order_idx_ < (int)orders_.size()) { orders_[next_order_idx_].status = "preview"; next_order_idx_++; }
    }

    void gen_order() {
        if (!rng_init_) {
            rng_init_ = true;
            for (auto& o : cfg.order_sequence) order_size_pool_.push_back((int)o.items_required.size());
            if (order_size_pool_.empty()) order_size_pool_.push_back(3);
            for (auto& [pk, t] : cfg.shelf_types) item_type_pool_.push_back(t);
            if (item_type_pool_.empty()) return;
        }
        int sz = order_size_pool_[order_rng_() % order_size_pool_.size()];
        std::vector<std::string> items;
        for (int i = 0; i < sz; i++) items.push_back(item_type_pool_[order_rng_() % item_type_pool_.size()]);
        orders_.push_back({"gen_" + std::to_string(orders_.size()), items, {}, "hidden"});
    }
};

// ============================================================
// MAPF Planner Core
// ============================================================

class MAPFPlanner {
public:
    ReconConfig cfg;
    Grid grid;
    BFSCache bfs;
    PIBTResolver pibt;
    std::vector<Zone> zones;
    Pos spawn;
    int n_bots;

    // Plan output
    using Plan = std::unordered_map<int, std::vector<RecordedAction>>;

    void init(const ReconConfig& c) {
        cfg = c;
        n_bots = c.n_bots;
        spawn = c.spawn_positions.empty() ? Pos{28, 16} : c.spawn_positions[0];

        // Build grid
        grid.init(c.width, c.height, c.walls, c.shelves);
        // One-way disabled — causes pathfinding issues with both BFS step and PIBT
        // if (n_bots >= 10) grid.setup_nightmare_one_way();
        bfs.init(&grid);
        pibt.grid = &grid;
        pibt.bfs = &bfs;

        // Build shelf adjacency: for each shelf, find walkable pickup positions
        std::unordered_map<int, std::vector<Pos>> shelf_adjacent;
        for (auto& [pk, type] : c.shelf_types) {
            Pos sp = grid.from_key(pk);
            auto& adj = shelf_adjacent[pk];
            for (auto& d : DIRS) {
                Pos n = {sp.x + d.x, sp.y + d.y};
                if (grid.ok(n)) adj.push_back(n);
            }
        }

        // Setup zones
        setup_zones(shelf_adjacent);
    }

    void setup_zones(const std::unordered_map<int, std::vector<Pos>>& shelf_adjacent) {
        zones.clear();

        if (n_bots >= 10) {
            // Nightmare: 3 zones
            zones.push_back({"left",  0,  9, {1, 16},  {}, {}});
            zones.push_back({"mid",  10, 19, {15, 16}, {}, {}});
            zones.push_back({"right", 20, 29, {27, 16}, {}, {}});
        } else if (n_bots >= 5) {
            // Hard: single zone
            zones.push_back({"all", 0, cfg.width - 1, cfg.drop_off_zones[0], {}, {}});
        } else {
            // Easy/Medium: single zone
            zones.push_back({"all", 0, cfg.width - 1, cfg.drop_off_zones[0], {}, {}});
        }

        // Assign bots to zones (modulo distribution for nightmare)
        if (zones.size() == 3) {
            for (int i = 0; i < n_bots; i++) {
                zones[i % 3].bot_ids.push_back(i);
            }
        } else {
            for (int i = 0; i < n_bots; i++) {
                zones[0].bot_ids.push_back(i);
            }
        }

        // Classify shelves by zone
        for (auto& [pk, type] : cfg.shelf_types) {
            Pos sp = grid.from_key(pk);
            auto adj_it = shelf_adjacent.find(pk);
            if (adj_it == shelf_adjacent.end() || adj_it->second.empty()) continue;

            for (auto& zone : zones) {
                if (sp.x >= zone.x_min && sp.x <= zone.x_max) {
                    for (auto& pickup_pos : adj_it->second) {
                        zone.shelves[type].push_back({sp, pickup_pos});
                    }
                    break;
                }
            }
        }
    }

    // Find the zone a bot belongs to
    int bot_zone(int bot_id) const {
        for (int z = 0; z < (int)zones.size(); z++) {
            for (int bid : zones[z].bot_ids) {
                if (bid == bot_id) return z;
            }
        }
        return 0;
    }

    // Find nearest pickup position for item_type in zone, near a position
    std::pair<Pos, Pos> find_shelf(const std::string& item_type, Pos near, int zone_idx) {
        auto& zone = zones[zone_idx];
        auto it = zone.shelves.find(item_type);
        if (it == zone.shelves.end() || it->second.empty()) {
            // Fallback: try other zones
            for (int z = 0; z < (int)zones.size(); z++) {
                if (z == zone_idx) continue;
                auto it2 = zones[z].shelves.find(item_type);
                if (it2 != zones[z].shelves.end() && !it2->second.empty()) {
                    Pos best_shelf = it2->second[0].first;
                    Pos best_pickup = it2->second[0].second;
                    int best_d = bfs.distance(near, best_pickup);
                    for (auto& [s, p] : it2->second) {
                        int d = bfs.distance(near, p);
                        if (d < best_d) { best_d = d; best_shelf = s; best_pickup = p; }
                    }
                    return {best_shelf, best_pickup};
                }
            }
            return {{-1,-1}, {-1,-1}};
        }

        Pos best_shelf = it->second[0].first;
        Pos best_pickup = it->second[0].second;
        int best_d = bfs.distance(near, best_pickup);
        for (auto& [s, p] : it->second) {
            int d = bfs.distance(near, p);
            if (d < best_d) { best_d = d; best_shelf = s; best_pickup = p; }
        }
        return {best_shelf, best_pickup};
    }

    // Find shelf with specific index (for LNS variation)
    std::pair<Pos, Pos> find_shelf_idx(const std::string& item_type, int zone_idx, int shelf_idx) {
        auto& zone = zones[zone_idx];
        auto it = zone.shelves.find(item_type);
        if (it == zone.shelves.end() || it->second.empty()) return {{-1,-1}, {-1,-1}};
        int idx = shelf_idx % (int)it->second.size();
        return it->second[idx];
    }

    int count_shelves(const std::string& item_type, int zone_idx) {
        auto it = zones[zone_idx].shelves.find(item_type);
        if (it == zones[zone_idx].shelves.end()) return 0;
        return (int)it->second.size();
    }

    // -------------------------------------------------------
    // Build greedy trips for all orders
    // -------------------------------------------------------

    std::vector<Trip> build_greedy_trips() {
        std::vector<Trip> all_trips;

        // Track bot positions for greedy assignment
        std::vector<Pos> bot_pos(n_bots, spawn);
        // Track bot availability (round when free)
        std::vector<int> bot_avail(n_bots, 0);

        for (int oi = 0; oi < (int)cfg.order_sequence.size(); oi++) {
            auto& order = cfg.order_sequence[oi];
            auto& items = order.items_required;

            // Group items into batches of max 3 per trip
            for (int start = 0; start < (int)items.size(); start += 3) {
                int end = std::min(start + 3, (int)items.size());
                std::vector<std::string> batch(items.begin() + start, items.begin() + end);

                // Find best bot for this batch (nearest available bot in appropriate zone)
                int best_bot = -1;
                int best_cost = 999999;

                for (int z = 0; z < (int)zones.size(); z++) {
                    // Check if zone has all items
                    bool has_all = true;
                    for (auto& item : batch) {
                        if (zones[z].shelves.find(item) == zones[z].shelves.end() ||
                            zones[z].shelves[item].empty()) {
                            has_all = false;
                            break;
                        }
                    }
                    if (!has_all) continue;

                    for (int bid : zones[z].bot_ids) {
                        // Estimate trip cost
                        Pos pos = bot_pos[bid];
                        int cost = bot_avail[bid]; // penalty for busy bots
                        for (auto& item : batch) {
                            auto [shelf, pickup] = find_shelf(item, pos, z);
                            if (pickup.x < 0) { cost = 999999; break; }
                            cost += bfs.distance(pos, pickup) + 1; // +1 for pickup action
                            pos = pickup;
                        }
                        cost += bfs.distance(pos, zones[z].drop_off) + 1; // delivery

                        if (cost < best_cost) {
                            best_cost = cost;
                            best_bot = bid;
                        }
                    }
                }

                if (best_bot < 0) {
                    // No bot can handle this batch — assign to bot 0 as fallback
                    best_bot = 0;
                }

                int z = bot_zone(best_bot);

                // Build trip
                Trip trip;
                trip.bot_id = best_bot;
                trip.order_idx = oi;
                trip.drop_off = zones[z].drop_off;

                Pos pos = bot_pos[best_bot];
                for (auto& item : batch) {
                    trip.items.push_back(item);
                    auto [shelf, pickup] = find_shelf(item, pos, z);
                    trip.shelf_positions.push_back(shelf);
                    trip.pickup_positions.push_back(pickup);
                    pos = pickup;
                }

                // Update bot state
                int trip_cost = 0;
                Pos p = bot_pos[best_bot];
                for (auto& pp : trip.pickup_positions) {
                    trip_cost += bfs.distance(p, pp) + 1;
                    p = pp;
                }
                trip_cost += bfs.distance(p, trip.drop_off) + 1;
                bot_avail[best_bot] += trip_cost;
                bot_pos[best_bot] = trip.drop_off;

                all_trips.push_back(std::move(trip));
            }
        }

        return all_trips;
    }

    // -------------------------------------------------------
    // Round-by-round sequential planner (collision-free by construction)
    // -------------------------------------------------------

    struct PlanResult {
        Plan plan;
        int score;
        int rounds_used;
        int orders_completed;
        int items_delivered;
    };

    PlanResult plan_sequential(const std::vector<Trip>& trips) {
        Plan plan;
        for (int i = 0; i < n_bots; i++) plan[i] = {};

        // Bot state
        std::vector<Pos> bot_pos(n_bots, spawn);
        std::vector<std::vector<std::string>> bot_inv(n_bots);
        std::vector<int> bot_trip_cursor(n_bots, 0);      // which trip index
        std::vector<int> bot_pickup_idx(n_bots, 0);        // within current trip
        std::vector<std::string> bot_task(n_bots, "idle");
        std::vector<Pos> bot_target(n_bots, {-1, -1});
        std::vector<std::string> bot_target_type(n_bots);
        std::vector<int> bot_target_order(n_bots, -1);

        // Group trips per bot, sorted by order_idx
        std::unordered_map<int, std::vector<int>> bot_trips; // bot_id -> trip indices
        for (int i = 0; i < (int)trips.size(); i++) {
            bot_trips[trips[i].bot_id].push_back(i);
        }

        // Order tracking
        std::vector<Order> orders;
        for (auto& o : cfg.order_sequence) orders.push_back({o.id, o.items_required, {}, "hidden"});
        // Generate extra orders for beyond-recon planning
        {
            std::mt19937 gen_rng(42);
            std::vector<int> size_pool;
            for (auto& o : cfg.order_sequence) size_pool.push_back((int)o.items_required.size());
            if (size_pool.empty()) size_pool.push_back(3);
            std::vector<std::string> type_pool;
            for (auto& [pk, t] : cfg.shelf_types) type_pool.push_back(t);
            // Generate 200 extra orders
            for (int i = 0; i < 200; i++) {
                int sz = size_pool[gen_rng() % size_pool.size()];
                std::vector<std::string> items;
                for (int j = 0; j < sz; j++) items.push_back(type_pool[gen_rng() % type_pool.size()]);
                orders.push_back({"gen_" + std::to_string(orders.size()), items, {}, "hidden"});
            }
        }

        int next_order_idx = 0;
        if (!orders.empty()) { orders[0].status = "active"; next_order_idx = 1; }
        if ((int)orders.size() > 1) { orders[1].status = "preview"; next_order_idx = 2; }

        int score = 0;
        int items_delivered = 0;
        int orders_completed = 0;
        int active_order_idx = 0;

        auto get_active = [&]() -> Order* {
            for (auto& o : orders) if (o.status == "active") return &o;
            return nullptr;
        };
        auto get_preview = [&]() -> Order* {
            for (auto& o : orders) if (o.status == "preview") return &o;
            return nullptr;
        };

        // Advance orders helper
        std::function<void(int)> advance_orders = [&](int delivering_bot) {
            Order* pv = get_preview();
            if (pv) {
                pv->status = "active";
                active_order_idx++;
                // Auto-delivery for delivering bot
                auto rem = pv->remaining();
                std::vector<std::string> ni;
                for (auto& inv : bot_inv[delivering_bot]) {
                    auto it = std::find(rem.begin(), rem.end(), inv);
                    if (it != rem.end()) { rem.erase(it); pv->items_delivered.push_back(inv); score++; items_delivered++; }
                    else ni.push_back(inv);
                }
                bot_inv[delivering_bot] = ni;
                if (pv->complete()) {
                    score += 5; orders_completed++; pv->status = "complete";
                    // Promote next preview
                    if (next_order_idx < (int)orders.size()) { orders[next_order_idx].status = "preview"; next_order_idx++; }
                    advance_orders(delivering_bot);
                    return;
                }
            }
            // Promote next preview
            if (next_order_idx < (int)orders.size()) { orders[next_order_idx].status = "preview"; next_order_idx++; }
        };

        // All bots active from start (sequential processing handles spawn stacking)
        std::vector<bool> bot_active(n_bots, true);
        std::vector<bool> bot_scattered(n_bots, true);

        // Nearest drop-off helper
        auto nearest_dropoff = [&](Pos from) -> Pos {
            Pos best = cfg.drop_off_zones[0];
            int bd = bfs.distance(from, best);
            for (size_t i = 1; i < cfg.drop_off_zones.size(); i++) {
                int d = bfs.distance(from, cfg.drop_off_zones[i]);
                if (d < bd) { bd = d; best = cfg.drop_off_zones[i]; }
            }
            return best;
        };

        // Assign tasks from trips
        auto assign_tasks = [&]() {
            for (int bid = 0; bid < n_bots; bid++) {
                if (bot_task[bid] != "idle") continue;
                auto it = bot_trips.find(bid);
                if (it == bot_trips.end()) continue;
                int cursor = bot_trip_cursor[bid];
                if (cursor >= (int)it->second.size()) continue;

                int trip_idx = it->second[cursor];
                auto& trip = trips[trip_idx];

                // Allow pre-picking well ahead to keep all bots busy
                // Delivery only happens when the order is active
                if (trip.order_idx > active_order_idx + 20) continue;

                int pi = bot_pickup_idx[bid];
                if (pi < (int)trip.pickup_positions.size()) {
                    if ((int)bot_inv[bid].size() >= 3) {
                        // Full — deliver first
                        Order* act = get_active();
                        if (act) {
                            auto rem = act->remaining();
                            bool has_match = false;
                            for (auto& inv : bot_inv[bid]) {
                                if (std::find(rem.begin(), rem.end(), inv) != rem.end()) { has_match = true; break; }
                            }
                            if (has_match) {
                                bot_task[bid] = "deliver";
                                bot_target[bid] = nearest_dropoff(bot_pos[bid]);
                            }
                        }
                        continue;
                    }
                    bot_task[bid] = (trip.order_idx <= active_order_idx) ? "pick" : "pre_pick";
                    bot_target[bid] = trip.pickup_positions[pi];
                    bot_target_type[bid] = trip.items[pi];
                    bot_target_order[bid] = trip.order_idx;
                } else {
                    // All pickups done — deliver only if this trip's order is active
                    if (trip.order_idx <= active_order_idx) {
                        Order* act = get_active();
                        if (act) {
                            auto rem = act->remaining();
                            bool has_match = false;
                            for (auto& inv : bot_inv[bid]) {
                                if (std::find(rem.begin(), rem.end(), inv) != rem.end()) { has_match = true; break; }
                            }
                            if (has_match) {
                                bot_task[bid] = "deliver";
                                bot_target[bid] = trip.drop_off;
                                bot_target_order[bid] = trip.order_idx;
                            } else {
                                // Items picked but don't match — advance cursor and try next trip
                                bot_trip_cursor[bid]++;
                                bot_pickup_idx[bid] = 0;
                            }
                        }
                    } else {
                        // Future order — wait near drop-off zone for when it becomes active
                        // Park at zone drop-off approach
                        bot_task[bid] = "wait_deliver";
                        bot_target[bid] = trip.drop_off;
                        bot_target_order[bid] = trip.order_idx;
                    }
                }
            }

            // Idle/waiting bots with matching inventory should deliver
            for (int bid = 0; bid < n_bots; bid++) {
                if (bot_inv[bid].empty()) continue;
                if (bot_task[bid] != "idle" && bot_task[bid] != "wait_deliver") continue;
                Order* act = get_active();
                if (!act) continue;
                auto rem = act->remaining();
                bool has_match = false;
                for (auto& inv : bot_inv[bid]) {
                    if (std::find(rem.begin(), rem.end(), inv) != rem.end()) { has_match = true; break; }
                }
                if (has_match) {
                    bot_task[bid] = "deliver";
                    bot_target[bid] = nearest_dropoff(bot_pos[bid]);
                }
            }
        };

        // Stuck detection: distance-based progress tracking
        std::vector<int> last_dist(n_bots, 9999);
        std::vector<int> no_progress_rounds(n_bots, 0);
        const int STUCK_THRESHOLD = 15; // rounds without progress

        // Dropoff zone set
        PosSet dropoff_set;
        for (auto& z : cfg.drop_off_zones) dropoff_set.insert(z);

        // No stagger — all bots active from round 0
        // Sequential processing (low ID first) naturally handles spawn dispersal

        // Main simulation loop
        for (int round = 0; round < cfg.max_rounds; round++) {
            // No stagger — all bots active from round 0

            // Distance-based stuck detection
            for (int bid = 0; bid < n_bots; bid++) {
                if (!bot_active[bid]) continue;
                if (bot_task[bid] != "idle" && bot_target[bid].x >= 0) {
                    int d = bfs.distance(bot_pos[bid], bot_target[bid]);
                    if (d < last_dist[bid]) {
                        last_dist[bid] = d;
                        no_progress_rounds[bid] = 0;
                    } else {
                        no_progress_rounds[bid]++;
                    }
                    if (no_progress_rounds[bid] >= STUCK_THRESHOLD) {
                        bot_task[bid] = "idle";
                        bot_target[bid] = {-1, -1};
                        last_dist[bid] = 9999;
                        no_progress_rounds[bid] = 0;
                    }
                } else {
                    last_dist[bid] = 9999;
                    no_progress_rounds[bid] = 0;
                }
            }

            assign_tasks();

            // Debug: print status
            if (round == 10 || round == 30 || round == 50) {
                for (int bid = 0; bid < n_bots; bid++) {
                    printf("    B%d@(%d,%d) t=%s inv=%d tgt=(%d,%d)\n",
                           bid, bot_pos[bid].x, bot_pos[bid].y,
                           bot_task[bid].c_str(), (int)bot_inv[bid].size(),
                           bot_target[bid].x, bot_target[bid].y);
                }
            }
            if (round % 100 == 0 || round < 5 || (round < 100 && round % 25 == 0)) {
                int n_idle = 0, n_pick = 0, n_deliver = 0, n_wait = 0;
                for (int bid = 0; bid < n_bots; bid++) {
                    if (bot_task[bid] == "idle") n_idle++;
                    else if (bot_task[bid] == "pick" || bot_task[bid] == "pre_pick") n_pick++;
                    else if (bot_task[bid] == "deliver") n_deliver++;
                    else if (bot_task[bid] == "wait_deliver") n_wait++;
                }
                printf("  R%d: score=%d order=%d pick/del/wait/idle=%d/%d/%d/%d bot0@(%d,%d)t=%s\n",
                       round, score, active_order_idx,
                       n_pick, n_deliver, n_wait, n_idle,
                       bot_pos[0].x, bot_pos[0].y, bot_task[0].c_str());
            }

            // Phase 1: Execute actions at target
            std::set<int> action_bots;
            for (int bid = 0; bid < n_bots; bid++) {
                Pos cur = bot_pos[bid];

                if ((bot_task[bid] == "pick" || bot_task[bid] == "pre_pick") &&
                    cur == bot_target[bid] && (int)bot_inv[bid].size() < 3) {
                    // Pick up
                    action_bots.insert(bid);
                    plan[bid].push_back({"pick_up", cur, bot_target_type[bid]});
                    bot_inv[bid].push_back(bot_target_type[bid]);

                    // Advance pickup index
                    bot_pickup_idx[bid]++;
                    auto it = bot_trips.find(bid);
                    if (it != bot_trips.end()) {
                        int cursor = bot_trip_cursor[bid];
                        if (cursor < (int)it->second.size()) {
                            int trip_idx = it->second[cursor];
                            auto& trip = trips[trip_idx];
                            int pi = bot_pickup_idx[bid];
                            if (pi < (int)trip.pickup_positions.size()) {
                                bot_target[bid] = trip.pickup_positions[pi];
                                bot_target_type[bid] = trip.items[pi];
                            } else {
                                // All pickups done — deliver
                                bot_task[bid] = "deliver";
                                bot_target[bid] = trip.drop_off;
                            }
                        } else {
                            bot_task[bid] = "idle";
                            bot_target[bid] = {-1, -1};
                        }
                    }
                } else if (bot_task[bid] == "deliver" && cur == bot_target[bid]) {
                    Order* act = get_active();
                    if (act) {
                        auto rem = act->remaining();
                        bool has_match = false;
                        for (auto& inv : bot_inv[bid]) {
                            if (std::find(rem.begin(), rem.end(), inv) != rem.end()) { has_match = true; break; }
                        }
                        if (has_match) {
                            action_bots.insert(bid);
                            plan[bid].push_back({"drop_off", cur, ""});

                            // Deliver matching items
                            std::vector<std::string> new_inv;
                            for (auto& inv : bot_inv[bid]) {
                                auto it = std::find(rem.begin(), rem.end(), inv);
                                if (it != rem.end()) { rem.erase(it); act->items_delivered.push_back(inv); score++; items_delivered++; }
                                else new_inv.push_back(inv);
                            }
                            bot_inv[bid] = new_inv;

                            if (act->complete()) {
                                score += 5; orders_completed++; act->status = "complete";
                                advance_orders(bid);
                                // Reset pre_pick bots
                                for (int bid2 = 0; bid2 < n_bots; bid2++) {
                                    if (bid2 == bid) continue;
                                    if (bot_task[bid2] == "pre_pick" && bot_target_order[bid2] <= active_order_idx) {
                                        Order* new_act = get_active();
                                        if (new_act) {
                                            auto nr = new_act->remaining();
                                            bool hm = false;
                                            for (auto& inv : bot_inv[bid2]) {
                                                if (std::find(nr.begin(), nr.end(), inv) != nr.end()) { hm = true; break; }
                                            }
                                            if (hm) {
                                                bot_task[bid2] = "deliver";
                                                bot_target[bid2] = nearest_dropoff(bot_pos[bid2]);
                                            } else {
                                                bot_task[bid2] = "idle";
                                                bot_target[bid2] = {-1, -1};
                                            }
                                        }
                                    }
                                }
                            }

                            // Advance trip cursor
                            bot_trip_cursor[bid]++;
                            bot_pickup_idx[bid] = 0;
                            bot_task[bid] = "idle";
                            bot_target[bid] = {-1, -1};
                        } else {
                            bot_task[bid] = "idle";
                            bot_target[bid] = {-1, -1};
                        }
                    }
                }
            }

            // Re-assign after pickups/deliveries
            assign_tasks();

            // Phase 2: Sequential movement (collision-free by construction)
            std::vector<Pos> new_pos(n_bots);
            for (int bid = 0; bid < n_bots; bid++) new_pos[bid] = bot_pos[bid];

            for (int bid = 0; bid < n_bots; bid++) {
                if (action_bots.count(bid)) {
                    new_pos[bid] = bot_pos[bid]; // stay in place (did action)
                    continue;
                }

                Pos cur = bot_pos[bid];
                Pos target = bot_target[bid];

                if (target.x < 0 || cur == target) {
                    // No target or at target — wait
                    plan[bid].push_back({"wait", cur, ""});
                    new_pos[bid] = cur;
                    continue;
                }

                // BFS step: find first step toward target avoiding collisions
                Pos next = bfs_step(bid, cur, target, new_pos, bot_pos);
                if (next == cur) {
                    plan[bid].push_back({"wait", cur, ""});
                } else {
                    plan[bid].push_back({direction_action(cur, next), cur, ""});
                }
                new_pos[bid] = next;
            }

            bot_pos = new_pos;
        }

        return {plan, score, cfg.max_rounds, orders_completed, items_delivered};
    }

    // BFS step: find first step toward target, avoiding occupied cells
    Pos bfs_step(int bid, Pos cur, Pos goal,
                 const std::vector<Pos>& new_pos,
                 const std::vector<Pos>& old_pos) {
        if (cur == goal) return cur;

        // Build blocked set (all occupied positions, NO spawn exception)
        PosSet blocked;
        for (int i = 0; i < n_bots; i++) {
            if (i == bid) continue;
            Pos p = (i < bid) ? new_pos[i] : old_pos[i];
            blocked.insert(p);
        }

        // BFS from cur to goal
        std::deque<std::pair<Pos, Pos>> q; // (pos, first_step)
        PosSet visited;
        visited.insert(cur);

        Pos nbrs[4]; int nc;
        grid.neighbors(cur, nbrs, nc);
        for (int i = 0; i < nc; i++) {
            if (blocked.count(nbrs[i]) || visited.count(nbrs[i])) continue;
            if (nbrs[i] == goal) return nbrs[i];
            visited.insert(nbrs[i]);
            q.push_back({nbrs[i], nbrs[i]});
        }

        int max_search = 1000;
        while (!q.empty() && max_search-- > 0) {
            auto [pos, first] = q.front(); q.pop_front();
            grid.neighbors(pos, nbrs, nc);
            for (int i = 0; i < nc; i++) {
                if (blocked.count(nbrs[i]) || visited.count(nbrs[i])) continue;
                if (nbrs[i] == goal) return first;
                visited.insert(nbrs[i]);
                q.push_back({nbrs[i], first});
            }
        }

        // No path to goal — try any free neighbor (closest to goal)
        grid.neighbors(cur, nbrs, nc);
        struct Cand { Pos p; int d; };
        std::vector<Cand> candidates;
        for (int i = 0; i < nc; i++) {
            if (!blocked.count(nbrs[i])) {
                candidates.push_back({nbrs[i], bfs.distance(nbrs[i], goal)});
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](auto& a, auto& b) { return a.d < b.d; });
        if (!candidates.empty()) return candidates[0].p;

        return cur; // stuck — wait
    }

    // -------------------------------------------------------
    // Reactive Sequential Planner (dynamic item assignment)
    // -------------------------------------------------------

    PlanResult plan_reactive() {
        Plan plan;
        for (int i = 0; i < n_bots; i++) plan[i] = {};

        std::vector<Pos> bot_pos(n_bots, spawn);
        std::vector<std::vector<std::string>> bot_inv(n_bots);
        std::vector<std::string> bot_task(n_bots, "idle");
        std::vector<Pos> bot_target(n_bots, {-1, -1});
        std::vector<std::string> bot_target_type(n_bots);

        // Order management
        std::vector<Order> orders;
        for (auto& o : cfg.order_sequence) orders.push_back({o.id, o.items_required, {}, "hidden"});
        // Generate extra orders
        {
            std::mt19937 gen_rng(42);
            std::vector<int> sp;
            for (auto& o : cfg.order_sequence) sp.push_back((int)o.items_required.size());
            if (sp.empty()) sp.push_back(3);
            std::vector<std::string> tp;
            for (auto& [pk, t] : cfg.shelf_types) tp.push_back(t);
            for (int i = 0; i < 200; i++) {
                int sz = sp[gen_rng() % sp.size()];
                std::vector<std::string> items;
                for (int j = 0; j < sz; j++) items.push_back(tp[gen_rng() % tp.size()]);
                orders.push_back({"gen_" + std::to_string(orders.size()), items, {}, "hidden"});
            }
        }
        int next_order_idx = 0;
        if (!orders.empty()) { orders[0].status = "active"; next_order_idx = 1; }
        if ((int)orders.size() > 1) { orders[1].status = "preview"; next_order_idx = 2; }

        int score = 0, items_delivered = 0, orders_completed = 0;
        int active_order_idx = 0;

        auto get_active = [&]() -> Order* {
            for (auto& o : orders) if (o.status == "active") return &o; return nullptr;
        };
        auto get_preview = [&]() -> Order* {
            for (auto& o : orders) if (o.status == "preview") return &o; return nullptr;
        };

        std::function<void(int)> advance_orders = [&](int bid) {
            Order* pv = get_preview();
            if (pv) {
                pv->status = "active";
                active_order_idx++;
                auto rem = pv->remaining();
                std::vector<std::string> ni;
                for (auto& inv : bot_inv[bid]) {
                    auto it = std::find(rem.begin(), rem.end(), inv);
                    if (it != rem.end()) { rem.erase(it); pv->items_delivered.push_back(inv); score++; items_delivered++; }
                    else ni.push_back(inv);
                }
                bot_inv[bid] = ni;
                if (pv->complete()) {
                    score += 5; orders_completed++; pv->status = "complete";
                    if (next_order_idx < (int)orders.size()) { orders[next_order_idx].status = "preview"; next_order_idx++; }
                    advance_orders(bid);
                    return;
                }
            }
            if (next_order_idx < (int)orders.size()) { orders[next_order_idx].status = "preview"; next_order_idx++; }
        };

        auto nearest_dropoff = [&](Pos from) -> Pos {
            Pos best = cfg.drop_off_zones[0];
            int bd = bfs.distance(from, best);
            for (size_t i = 1; i < cfg.drop_off_zones.size(); i++) {
                int d = bfs.distance(from, cfg.drop_off_zones[i]);
                if (d < bd) { bd = d; best = cfg.drop_off_zones[i]; }
            }
            return best;
        };

        // Claimed pickup positions — prevent 2 bots targeting same spot
        PosSet claimed_positions;

        // Distance-based stuck detection
        std::vector<int> last_dist(n_bots, 9999);
        std::vector<int> no_progress(n_bots, 0);

        PosSet dropoff_set;
        for (auto& z : cfg.drop_off_zones) dropoff_set.insert(z);

        // Helper: find nearest shelf of given type for a bot (excluding claimed positions)
        auto find_nearest_shelf = [&](const std::string& item_type, Pos from) -> Pos {
            int best_d = 999999;
            Pos best_pickup = {-1, -1};
            for (auto& zone : zones) {
                auto zit = zone.shelves.find(item_type);
                if (zit == zone.shelves.end()) continue;
                for (auto& [shelf, pickup] : zit->second) {
                    if (claimed_positions.count(pickup)) continue;
                    int d = bfs.distance(from, pickup);
                    if (d < best_d) { best_d = d; best_pickup = pickup; }
                }
            }
            if (best_pickup.x < 0) {
                // All claimed — find ANY shelf of this type (allow sharing)
                for (auto& zone : zones) {
                    auto zit = zone.shelves.find(item_type);
                    if (zit == zone.shelves.end()) continue;
                    for (auto& [shelf, pickup] : zit->second) {
                        int d = bfs.distance(from, pickup);
                        if (d < best_d) { best_d = d; best_pickup = pickup; }
                    }
                }
            }
            return best_pickup;
        };

        // Stagger: bot N starts at round N*stagger
        int stagger = (n_bots > 3) ? 2 : 0;
        int round = 0; // forward-declare for lambda capture

        // Assign items from active/preview order to idle bots
        auto assign_reactive = [&]() {
            Order* act = get_active();
            if (!act) return;

            // Step 1: Idle bots with matching inventory → deliver
            for (int bid = 0; bid < n_bots; bid++) {
                if (round < bid * stagger) continue;
                if (bot_task[bid] != "idle" || bot_inv[bid].empty()) continue;
                auto rem = act->remaining();
                bool has_match = false;
                for (auto& inv : bot_inv[bid]) {
                    if (std::find(rem.begin(), rem.end(), inv) != rem.end()) { has_match = true; break; }
                }
                if (has_match) {
                    bot_task[bid] = "deliver";
                    bot_target[bid] = nearest_dropoff(bot_pos[bid]);
                }
            }

            // Step 2: Compute remaining items still needed (not being picked/delivered)
            auto act_remaining = act->remaining();
            for (int bid = 0; bid < n_bots; bid++) {
                if (bot_task[bid] == "deliver") {
                    for (auto& inv : bot_inv[bid]) {
                        auto it = std::find(act_remaining.begin(), act_remaining.end(), inv);
                        if (it != act_remaining.end()) act_remaining.erase(it);
                    }
                }
                if (bot_task[bid] == "pick") {
                    auto it = std::find(act_remaining.begin(), act_remaining.end(), bot_target_type[bid]);
                    if (it != act_remaining.end()) act_remaining.erase(it);
                }
            }

            // Step 3: Assign idle bots to remaining active items
            for (int bid = 0; bid < n_bots; bid++) {
                if (round < bid * stagger) continue;
                if (bot_task[bid] != "idle") continue;
                if (act_remaining.empty()) break;
                if ((int)bot_inv[bid].size() >= 3) continue;

                // Find nearest needed item
                int best_d = 999999;
                Pos best_pickup = {-1, -1};
                std::string best_type;

                for (auto& item_type : act_remaining) {
                    Pos pickup = find_nearest_shelf(item_type, bot_pos[bid]);
                    if (pickup.x >= 0) {
                        int d = bfs.distance(bot_pos[bid], pickup);
                        if (d < best_d) { best_d = d; best_pickup = pickup; best_type = item_type; }
                    }
                }

                if (best_pickup.x >= 0) {
                    bot_task[bid] = "pick";
                    bot_target[bid] = best_pickup;
                    bot_target_type[bid] = best_type;
                    claimed_positions.insert(best_pickup);
                    auto it = std::find(act_remaining.begin(), act_remaining.end(), best_type);
                    if (it != act_remaining.end()) act_remaining.erase(it);
                }
            }

            // Step 4: Preview pre-pick (limited for nightmare to prevent dead weight)
            Order* pv = get_preview();
            if (!pv) return;
            auto pv_remaining = pv->remaining();
            for (int bid = 0; bid < n_bots; bid++) {
                if (bot_task[bid] == "pre_pick") {
                    auto it = std::find(pv_remaining.begin(), pv_remaining.end(), bot_target_type[bid]);
                    if (it != pv_remaining.end()) pv_remaining.erase(it);
                }
            }

            for (int bid = 0; bid < n_bots; bid++) {
                if (round < bid * stagger) continue;
                if (bot_task[bid] != "idle") continue;
                if (pv_remaining.empty()) break;
                // Limit pre-pick: leave 2 slots free for active order items
                // Full inventory with non-matching items = permanently stuck
                int max_prepick = (n_bots >= 10) ? 2 : 3;
                if ((int)bot_inv[bid].size() >= max_prepick) continue;

                int best_d = 999999;
                Pos best_pickup = {-1, -1};
                std::string best_type;

                for (auto& item_type : pv_remaining) {
                    Pos pickup = find_nearest_shelf(item_type, bot_pos[bid]);
                    if (pickup.x >= 0) {
                        int d = bfs.distance(bot_pos[bid], pickup);
                        if (d < best_d) { best_d = d; best_pickup = pickup; best_type = item_type; }
                    }
                }

                if (best_pickup.x >= 0) {
                    bot_task[bid] = "pre_pick";
                    bot_target[bid] = best_pickup;
                    bot_target_type[bid] = best_type;
                    claimed_positions.insert(best_pickup);
                    auto it = std::find(pv_remaining.begin(), pv_remaining.end(), best_type);
                    if (it != pv_remaining.end()) pv_remaining.erase(it);
                }
            }
        };

        for (round = 0; round < cfg.max_rounds; round++) {
            // Stuck detection (skip staggered bots)
            for (int bid = 0; bid < n_bots; bid++) {
                if (round < bid * stagger) continue; // not yet active
                if (bot_task[bid] != "idle" && bot_target[bid].x >= 0) {
                    int d = bfs.distance(bot_pos[bid], bot_target[bid]);
                    if (d < last_dist[bid]) { last_dist[bid] = d; no_progress[bid] = 0; }
                    else no_progress[bid]++;
                    if (no_progress[bid] >= 25) {
                        claimed_positions.erase(bot_target[bid]);
                        // For stuck deliver bots: try alternate drop-off zone
                        if (bot_task[bid] == "deliver" && !bot_inv[bid].empty() && cfg.drop_off_zones.size() > 1) {
                            // Find a different drop-off zone
                            Pos cur_target = bot_target[bid];
                            Pos alt = cfg.drop_off_zones[0];
                            int best_d = 999999;
                            for (auto& z : cfg.drop_off_zones) {
                                if (z == cur_target) continue;
                                int d = bfs.distance(bot_pos[bid], z);
                                if (d < best_d) { best_d = d; alt = z; }
                            }
                            bot_target[bid] = alt;
                            last_dist[bid] = best_d;
                            no_progress[bid] = 0;
                        } else {
                        bot_task[bid] = "idle";
                        bot_target[bid] = {-1, -1};
                        bot_target_type[bid] = "";
                        last_dist[bid] = 9999;
                        no_progress[bid] = 0;
                        }
                    }
                } else {
                    last_dist[bid] = 9999;
                    no_progress[bid] = 0;
                }
            }

            assign_reactive();

            // Debug
            if (round % 100 == 0 || round < 3) {
                int n_idle = 0, n_pick = 0, n_deliver = 0, n_pre = 0;
                for (int bid = 0; bid < n_bots; bid++) {
                    if (bot_task[bid] == "idle") n_idle++;
                    else if (bot_task[bid] == "pick") n_pick++;
                    else if (bot_task[bid] == "deliver") n_deliver++;
                    else if (bot_task[bid] == "pre_pick") n_pre++;
                }
                printf("  R%d: score=%d order=%d pick/del/pre/idle=%d/%d/%d/%d\n",
                       round, score, active_order_idx, n_pick, n_deliver, n_pre, n_idle);
            }

            // Phase 1: Actions at target
            std::set<int> action_bots;
            for (int bid = 0; bid < n_bots; bid++) {
                Pos cur = bot_pos[bid];
                if ((bot_task[bid] == "pick" || bot_task[bid] == "pre_pick") &&
                    cur == bot_target[bid] && (int)bot_inv[bid].size() < 3) {
                    action_bots.insert(bid);
                    plan[bid].push_back({"pick_up", cur, bot_target_type[bid]});
                    bot_inv[bid].push_back(bot_target_type[bid]);
                    // Release position claim and reset
                    claimed_positions.erase(cur);
                    bot_task[bid] = "idle";
                    bot_target[bid] = {-1, -1};
                    bot_target_type[bid] = "";
                    last_dist[bid] = 9999;
                } else if (bot_task[bid] == "deliver" && cur == bot_target[bid]) {
                    Order* act = get_active();
                    if (act) {
                        auto rem = act->remaining();
                        bool has_match = false;
                        for (auto& inv : bot_inv[bid]) {
                            if (std::find(rem.begin(), rem.end(), inv) != rem.end()) { has_match = true; break; }
                        }
                        if (has_match) {
                            action_bots.insert(bid);
                            plan[bid].push_back({"drop_off", cur, ""});
                            std::vector<std::string> new_inv;
                            for (auto& inv : bot_inv[bid]) {
                                auto it = std::find(rem.begin(), rem.end(), inv);
                                if (it != rem.end()) { rem.erase(it); act->items_delivered.push_back(inv); score++; items_delivered++; }
                                else new_inv.push_back(inv);
                            }
                            bot_inv[bid] = new_inv;
                            if (act->complete()) {
                                score += 5; orders_completed++; act->status = "complete";
                                advance_orders(bid);
                                // Reset claims and bots for new active order
                                claimed_positions.clear();
                                for (int bid2 = 0; bid2 < n_bots; bid2++) {
                                    if (bid2 == bid && bot_task[bid2] == "deliver") continue;
                                    if (bot_task[bid2] == "pick" || bot_task[bid2] == "pre_pick") {
                                        // Check if bot has matching items for new active order
                                        Order* new_act = get_active();
                                        if (new_act && !bot_inv[bid2].empty()) {
                                            auto nr = new_act->remaining();
                                            bool hm = false;
                                            for (auto& inv : bot_inv[bid2]) {
                                                if (std::find(nr.begin(), nr.end(), inv) != nr.end()) { hm = true; break; }
                                            }
                                            if (hm) {
                                                bot_task[bid2] = "deliver";
                                                bot_target[bid2] = nearest_dropoff(bot_pos[bid2]);
                                                continue;
                                            }
                                        }
                                        bot_task[bid2] = "idle";
                                        bot_target[bid2] = {-1, -1};
                                    }
                                }
                            }
                            bot_task[bid] = "idle";
                            bot_target[bid] = {-1, -1};
                        } else {
                            bot_task[bid] = "idle";
                            bot_target[bid] = {-1, -1};
                        }
                    }
                }
            }

            // Re-assign
            assign_reactive();

            // Phase 2: Movement — BFS step for ≤3 bots, PIBT for >3 bots
            if (n_bots <= 3 || n_bots >= 10) { // BFS for 1-3 bots and nightmare, PIBT for hard (4-9)
                // Sequential BFS step (matches sim exactly)
                std::vector<Pos> new_pos(n_bots);
                for (int bid = 0; bid < n_bots; bid++) new_pos[bid] = bot_pos[bid];
                for (int bid = 0; bid < n_bots; bid++) {
                    if (action_bots.count(bid)) { new_pos[bid] = bot_pos[bid]; continue; }
                    Pos cur = bot_pos[bid];
                    Pos target = bot_target[bid];
                    if (target.x < 0 || cur == target) {
                        plan[bid].push_back({"wait", cur, ""});
                        new_pos[bid] = cur;
                        continue;
                    }
                    Pos next = bfs_step(bid, cur, target, new_pos, bot_pos);
                    if (next == cur) plan[bid].push_back({"wait", cur, ""});
                    else plan[bid].push_back({direction_action(cur, next), cur, ""});
                    new_pos[bid] = next;
                }
                bot_pos = new_pos;
            } else {
            // PIBT for multi-bot
            std::vector<PIBTBot> pibt_bots;
            std::unordered_map<int, Pos> pibt_targets;
            std::unordered_map<int, int> pibt_urgency;

            for (int bid = 0; bid < n_bots; bid++) {
                pibt_bots.push_back({bid, bot_pos[bid]});

                if (action_bots.count(bid)) {
                    pibt_targets[bid] = bot_pos[bid];
                    pibt_urgency[bid] = -1;
                    continue;
                }
                if (round < bid * stagger) {
                    pibt_targets[bid] = bot_pos[bid];
                    pibt_urgency[bid] = 3;
                    continue;
                }

                Pos target = bot_target[bid];
                if (target.x < 0) target = bot_pos[bid];
                pibt_targets[bid] = target;

                if (bot_task[bid] == "deliver") pibt_urgency[bid] = 0;
                else if (bot_task[bid] == "pick") pibt_urgency[bid] = 1;
                else if (bot_task[bid] == "pre_pick") pibt_urgency[bid] = 2;
                else pibt_urgency[bid] = 3;

                // ESCAPE: non-delivering bots at drop-off
                if (bot_task[bid] != "deliver" && dropoff_set.count(bot_pos[bid]))
                    pibt_urgency[bid] = -1;
            }

            auto next_pos = pibt.resolve(pibt_bots, pibt_targets, pibt_urgency, round);

            for (int bid = 0; bid < n_bots; bid++) {
                if (action_bots.count(bid)) continue;
                Pos cur = bot_pos[bid];
                Pos np = next_pos.count(bid) ? next_pos[bid] : cur;
                if (np != cur) plan[bid].push_back({direction_action(cur, np), cur, ""});
                else plan[bid].push_back({"wait", cur, ""});
            }

            for (int bid = 0; bid < n_bots; bid++) {
                if (!action_bots.count(bid))
                    bot_pos[bid] = next_pos.count(bid) ? next_pos[bid] : bot_pos[bid];
            }
            } // end PIBT else
        }

        return {plan, score, cfg.max_rounds, orders_completed, items_delivered};
    }

    // -------------------------------------------------------
    // LNS: perturb trips and re-plan
    // -------------------------------------------------------

    std::vector<Trip> perturb_trips(const std::vector<Trip>& trips, std::mt19937& rng) {
        std::vector<Trip> modified = trips;
        if (modified.empty()) return modified;

        // Pick a random perturbation strategy
        std::uniform_int_distribution<int> strat(0, 4);
        int strategy = strat(rng);

        switch (strategy) {
        case 0: {
            // Shelf variation: change shelf choice for random items
            std::uniform_int_distribution<int> trip_pick(0, (int)modified.size() - 1);
            int ti = trip_pick(rng);
            auto& trip = modified[ti];
            int z = bot_zone(trip.bot_id);
            for (int i = 0; i < (int)trip.items.size(); i++) {
                int n_shelves = count_shelves(trip.items[i], z);
                if (n_shelves > 1) {
                    std::uniform_int_distribution<int> shelf_pick(0, n_shelves - 1);
                    auto [shelf, pickup] = find_shelf_idx(trip.items[i], z, shelf_pick(rng));
                    if (pickup.x >= 0) {
                        trip.shelf_positions[i] = shelf;
                        trip.pickup_positions[i] = pickup;
                    }
                }
            }
            break;
        }
        case 1: {
            // Batch size variation: split or merge trips for a random order
            if (modified.size() < 2) break;
            std::uniform_int_distribution<int> order_pick(0, (int)cfg.order_sequence.size() - 1);
            int oi = order_pick(rng);
            // Find all trips for this order
            std::vector<int> order_trip_indices;
            for (int i = 0; i < (int)modified.size(); i++) {
                if (modified[i].order_idx == oi) order_trip_indices.push_back(i);
            }
            if (order_trip_indices.empty()) break;
            // Collect all items from these trips
            std::vector<std::string> all_items;
            int first_bot = modified[order_trip_indices[0]].bot_id;
            Pos first_dz = modified[order_trip_indices[0]].drop_off;
            for (int idx : order_trip_indices) {
                for (auto& item : modified[idx].items) all_items.push_back(item);
            }
            // Remove old trips (reverse order to preserve indices)
            std::sort(order_trip_indices.rbegin(), order_trip_indices.rend());
            for (int idx : order_trip_indices) modified.erase(modified.begin() + idx);
            // Re-batch with random size (1-3)
            std::uniform_int_distribution<int> batch_pick(1, 3);
            int batch_sz = batch_pick(rng);
            int z = bot_zone(first_bot);
            Pos pos = spawn; // approximate
            for (int start = 0; start < (int)all_items.size(); start += batch_sz) {
                int end = std::min(start + batch_sz, (int)all_items.size());
                Trip trip;
                trip.bot_id = first_bot;
                trip.order_idx = oi;
                trip.drop_off = first_dz;
                for (int j = start; j < end; j++) {
                    trip.items.push_back(all_items[j]);
                    auto [shelf, pickup] = find_shelf(all_items[j], pos, z);
                    trip.shelf_positions.push_back(shelf);
                    trip.pickup_positions.push_back(pickup);
                    if (pickup.x >= 0) pos = pickup;
                }
                modified.push_back(std::move(trip));
            }
            break;
        }
        case 2: {
            // Bot reassignment: move a trip to a different bot in same zone
            std::uniform_int_distribution<int> trip_pick(0, (int)modified.size() - 1);
            int ti = trip_pick(rng);
            auto& trip = modified[ti];
            int z = bot_zone(trip.bot_id);
            auto& zone = zones[z];
            if (zone.bot_ids.size() > 1) {
                std::uniform_int_distribution<int> bot_pick(0, (int)zone.bot_ids.size() - 1);
                trip.bot_id = zone.bot_ids[bot_pick(rng)];
            }
            break;
        }
        case 3: {
            // Pickup order swap within a trip
            std::uniform_int_distribution<int> trip_pick(0, (int)modified.size() - 1);
            int ti = trip_pick(rng);
            auto& trip = modified[ti];
            if ((int)trip.items.size() >= 2) {
                std::uniform_int_distribution<int> idx_pick(0, (int)trip.items.size() - 2);
                int i = idx_pick(rng);
                std::swap(trip.items[i], trip.items[i + 1]);
                std::swap(trip.pickup_positions[i], trip.pickup_positions[i + 1]);
                std::swap(trip.shelf_positions[i], trip.shelf_positions[i + 1]);
            }
            break;
        }
        case 4: {
            // Zone reassignment: move item to different zone (if available)
            if (zones.size() <= 1) break;
            std::uniform_int_distribution<int> trip_pick(0, (int)modified.size() - 1);
            int ti = trip_pick(rng);
            auto& trip = modified[ti];
            int old_z = bot_zone(trip.bot_id);
            std::uniform_int_distribution<int> zone_pick(0, (int)zones.size() - 1);
            int new_z = zone_pick(rng);
            if (new_z == old_z || zones[new_z].bot_ids.empty()) break;
            // Check if new zone has all items
            bool has_all = true;
            for (auto& item : trip.items) {
                if (zones[new_z].shelves.find(item) == zones[new_z].shelves.end()) { has_all = false; break; }
            }
            if (!has_all) break;
            // Reassign
            std::uniform_int_distribution<int> bot_pick(0, (int)zones[new_z].bot_ids.size() - 1);
            trip.bot_id = zones[new_z].bot_ids[bot_pick(rng)];
            trip.drop_off = zones[new_z].drop_off;
            Pos pos = spawn;
            for (int i = 0; i < (int)trip.items.size(); i++) {
                auto [shelf, pickup] = find_shelf(trip.items[i], pos, new_z);
                trip.shelf_positions[i] = shelf;
                trip.pickup_positions[i] = pickup;
                if (pickup.x >= 0) pos = pickup;
            }
            break;
        }
        }

        return modified;
    }

    // -------------------------------------------------------
    // Verify plan by replaying through simulator
    // -------------------------------------------------------

    int verify_plan(const Plan& plan) {
        PlanSimulator sim;
        sim.init(cfg, grid);
        sim.reset();

        int max_round = 0;
        for (auto& [bid, acts] : plan) max_round = std::max(max_round, (int)acts.size());
        max_round = std::min(max_round, cfg.max_rounds);

        for (int r = 0; r < max_round; r++) {
            std::vector<std::pair<int, std::string>> actions;
            for (int bid = 0; bid < n_bots; bid++) {
                auto it = plan.find(bid);
                if (it == plan.end() || r >= (int)it->second.size()) {
                    actions.push_back({bid, "wait"});
                    continue;
                }
                auto& act = it->second[r];
                if (act.action == "pick_up") {
                    actions.push_back({bid, "pick_up:" + act.item_type});
                } else {
                    actions.push_back({bid, act.action});
                }
            }
            sim.step(actions);
        }

        return sim.score_;
    }
};

// ============================================================
// Load recon JSON
// ============================================================

ReconConfig load_recon(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) { std::cerr << "Cannot open: " << path << std::endl; exit(1); }
    json j; f >> j;
    ReconConfig cfg;
    cfg.width = j["grid_size"][0]; cfg.height = j["grid_size"][1];
    cfg.max_rounds = j.value("total_rounds", 500);
    cfg.n_bots = j.value("bot_count", 1);
    for (auto& w : j["walls"]) cfg.walls.insert({w[0], w[1]});
    for (auto& [type, positions] : j["shelf_map"].items()) {
        for (auto& p : positions) {
            Pos sp = {p[0], p[1]}; cfg.shelves.insert(sp);
            cfg.shelf_types[sp.y * cfg.width + sp.x] = type;
        }
    }
    cfg.drop_off = {j["drop_off"][0], j["drop_off"][1]};
    if (j.contains("drop_off_zones")) {
        for (auto& z : j["drop_off_zones"]) cfg.drop_off_zones.push_back({z[0], z[1]});
    }
    if (cfg.drop_off_zones.empty()) cfg.drop_off_zones.push_back(cfg.drop_off);
    for (auto& p : j.value("bot_start_positions", json::array())) cfg.spawn_positions.push_back({p[0], p[1]});
    if (cfg.spawn_positions.empty()) {
        for (int i = 0; i < cfg.n_bots; i++) cfg.spawn_positions.push_back({cfg.width - 2, cfg.height - 2});
    }
    for (auto& o : j.value("order_sequence", json::array())) {
        Order ord; ord.id = o["id"];
        for (auto& item : o["items_required"]) ord.items_required.push_back(item);
        ord.status = "hidden"; cfg.order_sequence.push_back(ord);
    }
    return cfg;
}

// ============================================================
// Save MAPF plan JSON
// ============================================================

void save_plan(const MAPFPlanner::Plan& plan, int score, int rounds, const std::string& path) {
    json j;
    j["total_rounds"] = rounds;
    j["expected_score"] = score;
    j["order_activations"] = json::object();
    j["pickup_schedule"] = json::array();
    j["dropoff_schedule"] = json::array();

    json actions = json::object();
    for (auto& [bid, acts] : plan) {
        json ba = json::array();
        for (auto& a : acts) {
            ba.push_back({
                {"action", a.action},
                {"position", {a.position.x, a.position.y}},
                {"item_type", a.item_type}
            });
        }
        actions[std::to_string(bid)] = ba;
    }
    j["actions"] = actions;

    std::ofstream f(path);
    f << j.dump(2);
    printf("Plan saved to %s (score %d, %d rounds)\n", path.c_str(), score, rounds);
}

// ============================================================
// LNS Search (multi-threaded)
// ============================================================

void lns_search(ReconConfig& cfg, int iterations, int n_workers, const std::string& output) {
    MAPFPlanner planner;
    planner.init(cfg);

    printf("=== C++ MAPF Planner ===\n");
    printf("Grid: %dx%d, Bots: %d, Orders: %d, Zones: %d\n",
           cfg.width, cfg.height, cfg.n_bots,
           (int)cfg.order_sequence.size(), (int)planner.zones.size());
    for (auto& z : planner.zones) {
        printf("  Zone %s: x=[%d,%d], bots=%d, drop_off=(%d,%d), item_types=%d\n",
               z.name.c_str(), z.x_min, z.x_max, (int)z.bot_ids.size(),
               z.drop_off.x, z.drop_off.y, (int)z.shelves.size());
    }
    fflush(stdout);

    // Build initial reactive plan
    printf("\nPlanning reactive...\n"); fflush(stdout);
    auto result = planner.plan_reactive();
    printf("Reactive: score=%d, orders=%d, items=%d\n",
           result.score, result.orders_completed, result.items_delivered);
    fflush(stdout);

    int verified = planner.verify_plan(result.plan);
    printf("Verified score: %d\n", verified); fflush(stdout);

    save_plan(result.plan, result.score, result.rounds_used, output);
    printf("\nLNS not yet integrated with reactive planner.\n");
    return;
    // TODO: Add LNS over stagger timing, zone assignments, etc.
#if 0

            int done = completed.fetch_add(1) + 1;
            if (done % 100 == 0) {
                auto t1 = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(t1 - t0).count();
                printf("  Progress: %d/%d, best=%d, %.1fs (%.1f iter/s)\n",
                       done, iterations, best_score.load(), elapsed, done / elapsed);
                fflush(stdout);
            }
        }
    };

    std::vector<std::thread> threads;
    int per_worker = (iterations + n_workers - 1) / n_workers;
    for (int w = 0; w < n_workers; w++) {
        int n = std::min(per_worker, iterations - w * per_worker);
        if (n > 0) threads.emplace_back(worker, w, n);
    }
    for (auto& t : threads) t.join();

    auto t1 = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double>(t1 - t0).count();
    printf("\n=== RESULTS ===\n");
    printf("Initial: %d\nBest: %d (+%d)\n", result.score, best_score.load(), best_score.load() - result.score);
    printf("Improvements: %d/%d\n", improvements.load(), iterations);
    printf("Time: %.1fs (%.1f iter/s)\n", total, iterations / total);

    save_plan(best_plan, best_score.load(), cfg.max_rounds, output);
#endif
}

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {
    std::string recon_path;
    int iterations = 1000;
    int workers = std::max(1, (int)std::thread::hardware_concurrency());
    std::string output = "mapf_plan_cpp_mapf.json";
    bool greedy_only = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--recon" && i+1 < argc) recon_path = argv[++i];
        else if (arg == "--iterations" && i+1 < argc) iterations = std::atoi(argv[++i]);
        else if (arg == "--workers" && i+1 < argc) workers = std::atoi(argv[++i]);
        else if (arg == "--output" && i+1 < argc) output = argv[++i];
        else if (arg == "--greedy") greedy_only = true;
    }

    if (recon_path.empty()) {
        std::cerr << "Usage: mapf --recon <file> [--iterations N] [--workers N] [--output file] [--greedy]" << std::endl;
        return 1;
    }

    auto cfg = load_recon(recon_path);

    if (greedy_only) {
        MAPFPlanner planner;
        planner.init(cfg);
        printf("=== Reactive Sequential MAPF ===\n");
        printf("Grid: %dx%d, Bots: %d, Orders: %d, Zones: %d\n",
               cfg.width, cfg.height, cfg.n_bots,
               (int)cfg.order_sequence.size(), (int)planner.zones.size());
        auto result = planner.plan_reactive();
        printf("Score: %d, Orders: %d, Items: %d\n",
               result.score, result.orders_completed, result.items_delivered);
        int verified = planner.verify_plan(result.plan);
        printf("Verified: %d\n", verified);
        save_plan(result.plan, result.score, result.rounds_used, output);
        return 0;
    }

    lns_search(cfg, iterations, workers, output);
    return 0;
}

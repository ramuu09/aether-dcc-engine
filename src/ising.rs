//! Host-side 64-node Ising graphs. The GPU shader is the same Jacobi rule.

pub const ISING_NODES: usize = 64;
pub const ISING_DEG: usize = 8;
pub const ISING_WAVES: u32 = 16;
pub const ISING_PUSH: u32 = 4;

#[derive(Clone, Debug)]
pub struct IsingGraph {
    pub neighbors: [[i32; ISING_DEG]; ISING_NODES],
    pub coupling: [[i32; ISING_DEG]; ISING_NODES],
    pub bias: [i32; ISING_NODES],
    pub edges: Vec<(usize, usize)>,
}

impl IsingGraph {
    pub fn new() -> Self {
        Self {
            neighbors: [[-1; ISING_DEG]; ISING_NODES],
            coupling: [[0; ISING_DEG]; ISING_NODES],
            bias: [0; ISING_NODES],
            edges: Vec::new(),
        }
    }

    pub fn degree(&self, node: usize) -> usize {
        self.neighbors[node].iter().filter(|slot| **slot >= 0).count()
    }

    pub fn add_edge(&mut self, a: usize, b: usize) {
        assert!(a < ISING_NODES && b < ISING_NODES && a != b, "edge {a} {b}");
        self.attach(a, b as i32, -1);
        self.attach(b, a as i32, -1);
        self.edges.push((a, b));
    }

    fn attach(&mut self, node: usize, neighbor: i32, coupling: i32) {
        for slot in 0..ISING_DEG {
            if self.neighbors[node][slot] < 0 {
                self.neighbors[node][slot] = neighbor;
                self.coupling[node][slot] = coupling;
                return;
            }
        }
        panic!("node {node} degree exceeds {ISING_DEG}");
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct RelaxResult {
    pub spins: [i32; ISING_NODES],
    pub t_pos: [u32; ISING_NODES],
    pub t_neg: [u32; ISING_NODES],
    pub y: [f64; ISING_NODES],
    pub cut: i32,
    pub energy: i32,
}

pub fn graph_from_raw(neighbors: &[i32], coupling: &[i32], bias: &[i32]) -> Result<IsingGraph, String> {
    if neighbors.len() < ISING_NODES * ISING_DEG || coupling.len() < ISING_NODES * ISING_DEG || bias.len() < ISING_NODES {
        return Err("ising buffers are short of 64 nodes by degree 8".into());
    }
    let mut graph = IsingGraph::new();
    for node in 0..ISING_NODES {
        graph.bias[node] = bias[node];
        for slot in 0..ISING_DEG {
            let neighbor = neighbors[node * ISING_DEG + slot];
            let couple = coupling[node * ISING_DEG + slot];
            if neighbor < -1 || neighbor >= ISING_NODES as i32 {
                return Err(format!("neighbor {neighbor} at node {node}"));
            }
            graph.neighbors[node][slot] = neighbor;
            graph.coupling[node][slot] = couple;
            if neighbor > node as i32 && couple != 0 {
                graph.edges.push((node, neighbor as usize));
            }
        }
    }
    Ok(graph)
}

pub fn score_cut(graph: &IsingGraph, spins: &[i32; ISING_NODES]) -> (i32, i32) {
    let mut cut = 0i32;
    let mut energy = 0i32;
    for &(a, b) in &graph.edges {
        energy += spins[a] * spins[b];
        if spins[a] != 0 && spins[b] != 0 && spins[a] != spins[b] {
            cut += 1;
        }
    }
    (cut, energy)
}

pub fn binary_greedy(graph: &IsingGraph) -> (i32, i32, u32, u32) {
    let mut spins = [0i32; ISING_NODES];
    for node in 0..ISING_NODES {
        spins[node] = if node % 4 < 2 { 1 } else { -1 };
    }
    let mut sweeps = 0u32;
    let mut flips = 0u32;
    while sweeps < 128 {
        sweeps += 1;
        let mut improved = false;
        for node in 0..ISING_NODES {
            let old = spins[node];
            if old == 0 {
                continue;
            }
            let (before, _) = score_cut(graph, &spins);
            spins[node] = -old;
            let (after, _) = score_cut(graph, &spins);
            if after > before {
                flips += 1;
                improved = true;
            } else {
                spins[node] = old;
            }
        }
        if !improved {
            break;
        }
    }
    let (cut, energy) = score_cut(graph, &spins);
    (cut, energy, sweeps, flips)
}

/// `0/0` keeps the previous spin. Equal nonzero banks reset the spin to 0.
pub fn reference_relax(graph: &IsingGraph) -> RelaxResult {
    let mut spins = [0i32; ISING_NODES];
    let mut t_pos = [0u32; ISING_NODES];
    let mut t_neg = [0u32; ISING_NODES];
    for wave in 0..ISING_WAVES {
        t_pos = [0; ISING_NODES];
        t_neg = [0; ISING_NODES];
        if wave == 0 {
            for node in 0..ISING_NODES {
                if graph.bias[node] > 0 {
                    t_pos[node] = t_pos[node].saturating_add(ISING_PUSH);
                } else if graph.bias[node] < 0 {
                    t_neg[node] = t_neg[node].saturating_add(ISING_PUSH);
                }
            }
        }
        let current = spins;
        for node in 0..ISING_NODES {
            if current[node] == 0 {
                continue;
            }
            for slot in 0..ISING_DEG {
                let neighbor = graph.neighbors[node][slot];
                let coupling = graph.coupling[node][slot];
                if neighbor < 0 || coupling == 0 {
                    continue;
                }
                let dest = neighbor as usize;
                let to_pos = if coupling < 0 {
                    current[node] < 0
                } else {
                    current[node] > 0
                };
                if to_pos {
                    t_pos[dest] = t_pos[dest].saturating_add(ISING_PUSH);
                } else {
                    t_neg[dest] = t_neg[dest].saturating_add(ISING_PUSH);
                }
            }
        }
        for node in 0..ISING_NODES {
            if t_pos[node] == 0 && t_neg[node] == 0 {
                continue;
            } else if t_pos[node] > t_neg[node] {
                spins[node] = 1;
            } else if t_neg[node] > t_pos[node] {
                spins[node] = -1;
            } else {
                spins[node] = 0;
            }
        }
    }
    let mut y = [0.0; ISING_NODES];
    for node in 0..ISING_NODES {
        y[node] = crate::cpu_alu::contention_ratio(t_pos[node], t_neg[node]);
    }
    let (cut, energy) = score_cut(graph, &spins);
    RelaxResult {
        spins,
        t_pos,
        t_neg,
        y,
        cut,
        energy,
    }
}

fn xs(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Seven disjoint 8-leaf stars. Node 63 stays disconnected.
pub fn star_forest() -> IsingGraph {
    let mut graph = IsingGraph::new();
    for hub_index in 0..7 {
        let hub = hub_index * 9;
        graph.bias[hub] = 1;
        for leaf in 1..9 {
            graph.add_edge(hub, hub + leaf);
        }
    }
    graph
}

/// MÃ¶bius ladder on 60 vertices: the cycle plus every antipodal rung.
pub fn dense_mobius_ladder() -> IsingGraph {
    let mut graph = IsingGraph::new();
    let n = 30usize;
    let nodes = n * 2;
    for index in 0..nodes {
        graph.add_edge(index, (index + 1) % nodes);
        if index < n {
            graph.add_edge(index, index + n);
        }
    }
    graph.bias[0] = 1;
    graph
}

/// Random bipartite AF spin glass. A +1 field on partition A breaks symmetry.
pub fn random_spin_glass() -> IsingGraph {
    let mut graph = IsingGraph::new();
    let mut state = 0xA5A5u64;
    let mut seen = std::collections::HashSet::<(usize, usize)>::new();
    let mut guard = 0u32;
    while graph.edges.len() < 96 && guard < 50_000 {
        guard += 1;
        state = xs(state);
        let a = (state % 32) as usize;
        state = xs(state);
        let b = 32 + (state % 32) as usize;
        if !seen.insert((a, b)) {
            continue;
        }
        if graph.degree(a) >= ISING_DEG || graph.degree(b) >= ISING_DEG {
            seen.remove(&(a, b));
            continue;
        }
        graph.add_edge(a, b);
    }
    assert_eq!(graph.edges.len(), 96, "spin glass did not reach 96 edges");
    for node in 0..32 {
        graph.bias[node] = 1;
    }
    graph
}

pub fn bipartite_grid() -> IsingGraph {
    let mut graph = IsingGraph::new();
    let at = |row: usize, col: usize| row * 8 + col;
    for row in 0..8 {
        for col in 0..8 {
            if col + 1 < 8 {
                graph.add_edge(at(row, col), at(row, col + 1));
            }
            if row + 1 < 8 {
                graph.add_edge(at(row, col), at(row + 1, col));
            }
        }
    }
    graph.bias[0] = 1;
    graph
}

/// Twelve disjoint 5-cycles. Each ring is an odd frustrated loop.
pub fn odd_cycle_web() -> IsingGraph {
    let mut graph = IsingGraph::new();
    let mut base = 0usize;
    for _ in 0..12 {
        let nodes: [usize; 5] = std::array::from_fn(|index| base + index);
        base += 5;
        for step in 0..5 {
            graph.add_edge(nodes[step], nodes[(step + 1) % 5]);
        }
        graph.bias[nodes[0]] = 1;
        graph.bias[nodes[1]] = -1;
    }
    graph
}

//! 64 â†’ 16 â†’ 10 ternary digit net.
//! Weights are in `{-1, 0, +1}`. A zero weight never contributes a tick.
//! Centered hidden state is `0` for an all-`-1` canvas, so every class score is `0`.

include!("bitnet_tables.rs");

pub const PIXELS: usize = 64;
pub const HIDDEN: usize = 16;
pub const CLASSES: usize = 10;

#[derive(Clone, Copy, Debug)]
pub struct HostDigit {
    pub hidden: [i32; HIDDEN],
    pub ternary: [i32; HIDDEN],
    pub scores: [i32; CLASSES],
    pub class: i32,
}

pub fn weight_census() -> (usize, usize) {
    let mut zeros = 0usize;
    let mut total = 0usize;
    for row in W1 {
        for weight in row {
            total += 1;
            if weight == 0 {
                zeros += 1;
            }
        }
    }
    for row in W2 {
        for weight in row {
            total += 1;
            if weight == 0 {
                zeros += 1;
            }
        }
    }
    (zeros, total)
}

pub fn glyph_f32(digit: usize) -> [f32; PIXELS] {
    let mut pixels = [0.0f32; PIXELS];
    for index in 0..PIXELS {
        pixels[index] = f32::from(GLYPHS[digit][index]);
    }
    pixels
}

pub fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

pub fn corrupt(pixels: &[f32; PIXELS], level: usize, state: &mut u64) -> [f32; PIXELS] {
    let mut order: [usize; PIXELS] = std::array::from_fn(|index| index);
    for index in (1..PIXELS).rev() {
        *state = xorshift64(*state);
        let swap_with = (*state as usize) % (index + 1);
        order.swap(index, swap_with);
    }
    let mut noisy = *pixels;
    for &pixel in order.iter().take(level) {
        noisy[pixel] = -noisy[pixel];
    }
    noisy
}

pub fn reference(pixels: &[f32; PIXELS]) -> HostDigit {
    let mut hidden = [0i32; HIDDEN];
    let mut ternary = [0i32; HIDDEN];
    for neuron in 0..HIDDEN {
        let mut pos = 0i32;
        let mut neg = 0i32;
        let mut sum_w = 0i32;
        for pixel in 0..PIXELS {
            let weight = i32::from(W1[neuron][pixel]);
            if weight == 0 {
                continue;
            }
            sum_w += weight;
            let sample = pixels[pixel];
            let toward_pos = (weight > 0 && sample > 0.0) || (weight < 0 && sample < 0.0);
            if toward_pos {
                pos += 1;
            } else {
                neg += 1;
            }
        }
        let centered = (pos - neg) + sum_w;
        hidden[neuron] = centered;
        ternary[neuron] = if centered > 4 {
            1
        } else if centered < 4 {
            -1
        } else {
            0
        };
    }
    let mut scores = [0i32; CLASSES];
    for class in 0..CLASSES {
        let mut pos = 0i32;
        let mut neg = 0i32;
        for neuron in 0..HIDDEN {
            let weight = i32::from(W2[class][neuron]);
            let value = hidden[neuron];
            if weight == 0 || value == 0 {
                continue;
            }
            if weight > 0 {
                if value > 0 {
                    pos += value;
                } else {
                    neg += -value;
                }
            } else if value > 0 {
                neg += value;
            } else {
                pos += -value;
            }
        }
        scores[class] = pos - neg;
    }
    HostDigit {
        class: argmax(scores),
        hidden,
        ternary,
        scores,
    }
}

pub fn argmax(scores: [i32; CLASSES]) -> i32 {
    let mut best = 0usize;
    let mut ties = 1u32;
    for index in 1..CLASSES {
        if scores[index] > scores[best] {
            best = index;
            ties = 1;
        } else if scores[index] == scores[best] {
            ties += 1;
        }
    }
    if ties == 1 { best as i32 } else { -1 }
}

pub fn ratio(pos: i32, neg: i32) -> f64 {
    let sum = pos + neg;
    if sum == 0 {
        0.0
    } else {
        f64::from(pos - neg) / f64::from(sum)
    }
}

#[derive(Clone, Debug)]
pub struct TernaryInferenceResult {
    pub scores: Vec<i32>,
    pub predicted_class: i32,
    pub winner_score: i32,
    pub confidence_margin: i32,
    pub hidden_centered: Vec<i32>,
}

#[derive(Clone, Debug)]
pub struct TernaryModel {
    pub name: String,
    pub input_features: usize,
    pub hidden_features: usize,
    pub output_classes: usize,
    pub w1: Vec<i8>,
    pub w2: Vec<i8>,
}

impl TernaryModel {
    /// Loads the built-in 64->16->10 geometric digit model
    pub fn default_digits() -> Self {
        let mut w1 = Vec::with_capacity(16 * 64);
        for row in W1 {
            w1.extend_from_slice(&row);
        }
        let mut w2 = Vec::with_capacity(10 * 16);
        for row in W2 {
            w2.extend_from_slice(&row);
        }
        Self {
            name: "BitNet-Digits-64-16-10".into(),
            input_features: PIXELS,
            hidden_features: HIDDEN,
            output_classes: CLASSES,
            w1,
            w2,
        }
    }

    pub fn new(
        name: impl Into<String>,
        input_features: usize,
        hidden_features: usize,
        output_classes: usize,
        w1: Vec<i8>,
        w2: Vec<i8>,
    ) -> Result<Self, String> {
        if w1.len() != hidden_features * input_features {
            return Err(format!("w1 size mismatch: expected {}, got {}", hidden_features * input_features, w1.len()));
        }
        if w2.len() != output_classes * hidden_features {
            return Err(format!("w2 size mismatch: expected {}, got {}", output_classes * hidden_features, w2.len()));
        }
        Ok(Self {
            name: name.into(),
            input_features,
            hidden_features,
            output_classes,
            w1,
            w2,
        })
    }

    pub fn sparsity(&self) -> (usize, usize, f64) {
        let mut zeros = 0usize;
        let total = self.w1.len() + self.w2.len();
        for &w in self.w1.iter().chain(self.w2.iter()) {
            if w == 0 {
                zeros += 1;
            }
        }
        let pct = if total == 0 { 0.0 } else { (zeros as f64 / total as f64) * 100.0 };
        (zeros, total, pct)
    }

    pub fn infer(&self, input: &[f32]) -> Result<TernaryInferenceResult, String> {
        if input.len() != self.input_features {
            return Err(format!("input length mismatch: expected {}, got {}", self.input_features, input.len()));
        }
        let mut hidden = vec![0i32; self.hidden_features];
        for h in 0..self.hidden_features {
            let mut pos = 0i32;
            let mut neg = 0i32;
            let mut sum_w = 0i32;
            let offset = h * self.input_features;
            for i in 0..self.input_features {
                let w = i32::from(self.w1[offset + i]);
                if w == 0 {
                    continue;
                }
                sum_w += w;
                let val = input[i];
                if (w > 0 && val > 0.0) || (w < 0 && val < 0.0) {
                    pos += 1;
                } else {
                    neg += 1;
                }
            }
            hidden[h] = (pos - neg) + sum_w;
        }

        let mut scores = vec![0i32; self.output_classes];
        for c in 0..self.output_classes {
            let mut score = 0i32;
            let offset = c * self.hidden_features;
            for h in 0..self.hidden_features {
                let w = i32::from(self.w2[offset + h]);
                let hv = hidden[h];
                if w != 0 && hv != 0 {
                    score += w * hv;
                }
            }
            scores[c] = score;
        }

        let mut best = 0usize;
        let mut ties = 1u32;
        let mut second = i32::MIN;
        for c in 1..self.output_classes {
            if scores[c] > scores[best] {
                second = scores[best];
                best = c;
                ties = 1;
            } else if scores[c] == scores[best] {
                ties += 1;
            } else if scores[c] > second {
                second = scores[c];
            }
        }
        let predicted = if ties == 1 { best as i32 } else { -1 };
        let winner = scores[best];
        let margin = if ties == 1 && second != i32::MIN { winner - second } else { 0 };

        Ok(TernaryInferenceResult {
            scores,
            predicted_class: predicted,
            winner_score: winner,
            confidence_margin: margin,
            hidden_centered: hidden,
        })
    }
}

// Builds report/marl_pong_presentation.pptx from results/ artifacts.
// Run: NODE_PATH=$(npm root -g) node report/build_deck.js
const pptxgen = require("pptxgenjs");

const W = 13.33, H = 7.5, M = 0.5;
// Palette derived from the game render: team A (80,160,255), team B (255,120,90)
// Light mode throughout: white background, navy primary, blue/red accents.
const DARK = "FFFFFF", BG = "FFFFFF", PRIMARY = "1C2A4A", ACCENT = "2B7FFF",
  RED = "E85D3F", TEXT = "182238", MUTED = "5B677F", TINT = "EEF4FE",
  DARKCARD = "F5F7FA", DARKLINE = "D8DEE9", DARKTEXT = "182238", DARKMUTED = "5B677F";
const F = "Arial";

let p = new pptxgen();
p.layout = "LAYOUT_WIDE";
p.author = "Aryan Mehesare";
p.title = "Emergent Coordination in 2v2 Pong via Independent PPO";

const bu = () => ({ code: "2013", indent: 12, color: MUTED });

function arena(slide, x, y, w, h, opts = {}) {
  // Schematic 2v2 arena: paddles, ball, center line, overlap band.
  slide.addShape(p.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.06, fill: { color: DARKCARD },
    line: { color: DARKLINE, width: 1 },
  });
  const cx = x + w / 2;
  slide.addShape(p.shapes.LINE, {
    x: cx, y: y + 0.12, w: 0, h: h - 0.24, line: { color: DARKLINE, width: 1, dashType: "dash" },
  });
  if (opts.overlapBand) {
    // Shared middle band where both teammates' ranges overlap.
    slide.addShape(p.shapes.RECTANGLE, {
      x, y: y + h * 0.3, w, h: h * 0.4, fill: { color: ACCENT, transparency: 88 },
    });
  }
  const pw = 0.045, ph = h * 0.34;
  const py1 = y + h * 0.14, py2 = y + h * 0.52;
  // Team A (blue, left) — two paddles; Team B (red, right)
  slide.addShape(p.shapes.ROUNDED_RECTANGLE, { x: x + w * 0.05, y: py1, w: pw, h: ph, rectRadius: 0.02, fill: { color: ACCENT } });
  slide.addShape(p.shapes.ROUNDED_RECTANGLE, { x: x + w * 0.05, y: py2, w: pw, h: ph, rectRadius: 0.02, fill: { color: "2B6CB0" } });
  slide.addShape(p.shapes.ROUNDED_RECTANGLE, { x: x + w * 0.95 - pw, y: py1, w: pw, h: ph, rectRadius: 0.02, fill: { color: RED } });
  slide.addShape(p.shapes.ROUNDED_RECTANGLE, { x: x + w * 0.95 - pw, y: py2, w: pw, h: ph, rectRadius: 0.02, fill: { color: "C24C38" } });
  slide.addShape(p.shapes.OVAL, { x: x + w * 0.42, y: y + h * 0.48, w: 0.09, h: 0.09, fill: { color: PRIMARY } });
  slide.addText(opts.label || "", {
    x, y: y + h + 0.05, w, h: 0.3, align: "center", fontSize: 12, fontFace: F,
    color: opts.labelColor || MUTED, margin: 0,
  });
}

function srcLine(slide, text) {
  slide.addText(text, {
    x: M, y: H - 0.42, w: W - 2 * M, h: 0.3, fontSize: 11, fontFace: F,
    color: MUTED, margin: 0,
  });
}

function title(slide, kicker, main) {
  slide.addText(kicker, {
    x: M, y: 0.32, w: W - 2 * M, h: 0.3, fontSize: 12, fontFace: F, bold: true,
    color: ACCENT, charSpacing: 3, margin: 0,
  });
  slide.addText(main, {
    x: M, y: 0.6, w: W - 2 * M, h: 0.75, fontSize: 32, fontFace: F, bold: true,
    color: TEXT, margin: 0,
  });
}

// ---------- 1 · Cover ----------
let s = p.addSlide();
s.background = { color: DARK };
s.addText("COURSE PROJECT · MULTI-AGENT REINFORCEMENT LEARNING", {
  x: M, y: 1.0, w: W - 2 * M, h: 0.35, fontSize: 14, fontFace: F, bold: true,
  color: ACCENT, charSpacing: 4, margin: 0,
});
s.addText("Emergent Coordination in 2v2 Pong", {
  x: M, y: 1.45, w: W - 2 * M, h: 1.6, fontSize: 54, fontFace: F, bold: true,
  color: DARKTEXT, margin: 0,
});
s.addText("Four independent PPO agents · one shared team reward · no communication, no role assignment, no opponent observations", {
  x: M, y: 3.05, w: 9.5, h: 0.8, fontSize: 20, fontFace: F, color: DARKMUTED, margin: 0,
});
arena(s, 3.1, 4.3, 7.1, 2.2, { label: "A1 + A2 (blue, left)  vs  B1 + B2 (red, right)", labelColor: DARKMUTED });
s.addText("Results: 3 seeds, 1M steps per 2v2 run · 60-episode deterministic eval · report/report.md", {
  x: M, y: 7.0, w: W - 2 * M, h: 0.3, fontSize: 12, fontFace: F, color: DARKMUTED, margin: 0,
});
s.addNotes("Open with the question, then play the footage. Frame: can teammates coordinate with only local state, teammate state, and a shared win/lose signal? Li et al. 2025 supplies the framing: non-stationarity and credit assignment.");

// ---------- 2 · Question & setup ----------
s = p.addSlide();
s.background = { color: BG };
title(s, "THE EXPERIMENT", "Coordination without communication");
arena(s, 7.6, 1.75, 5.2, 2.9, { overlapBand: true, label: "Overlapping vertical ranges — nobody owns a region" });
const rows = [
  ["8-dim observations", "own paddle y/vy, ball x/y/vx/vy, teammate y/vy — opponents excluded by design"],
  ["Shared team reward", "+1 / -1 per point, identical signal to both teammates"],
  ["3 actions per paddle", "stay / up / down, applied simultaneously every step"],
  ["Independent PPO ×4", "separate 64×64 actor-critic MLPs, CPU-only; no shared critic, no parameter sharing"],
];
rows.forEach(([h, d], i) => {
  const y = 1.8 + i * 1.15;
  s.addText(h, { x: M, y, w: 6.6, h: 0.35, fontSize: 18, fontFace: F, bold: true, color: PRIMARY, margin: 0 });
  s.addText(d, { x: M, y: y + 0.38, w: 6.6, h: 0.55, fontSize: 14, fontFace: F, color: MUTED, margin: 0 });
  if (i < rows.length - 1) s.addShape(p.shapes.LINE, { x: M, y: y + 1.0, w: 6.6, h: 0, line: { color: "E2E8F0", width: 0.75 } });
});
s.addNotes("Stress the design constraint: excluding opponent observations is deliberate — it keeps the research question about local+teammate coordination. The overlap band is where both teammates can roam; no fixed regions are assigned.");

// ---------- 3 · Protocol ----------
s = p.addSlide();
s.background = { color: BG };
title(s, "PROTOCOL", "Small enough to run, honest enough to trust");
const stats = [
  ["4", "independent PPO learners", "64×64 actor-critic MLPs, one per paddle, CPU"],
  ["3", "seeds per condition", "fresh initialization every run — no warm starts"],
  ["1M", "steps per 2v2 run", "1v1 sanity baseline at 300k; all runs logged per iteration"],
  ["60", "eval episodes × 3 opponents", "deterministic greedy play vs self, scripted tracker, random"],
];
stats.forEach(([n, h, d], i) => {
  const x = M + i * 3.2;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, {
    x, y: 2.0, w: 2.95, h: 3.3, rectRadius: 0.08, fill: { color: TINT },
  });
  s.addText(n, { x: x + 0.25, y: 2.25, w: 2.45, h: 1.1, fontSize: 66, fontFace: F, bold: true, color: ACCENT, margin: 0 });
  s.addText(h, { x: x + 0.25, y: 3.45, w: 2.45, h: 0.7, fontSize: 16, fontFace: F, bold: true, color: PRIMARY, margin: 0 });
  s.addText(d, { x: x + 0.25, y: 4.2, w: 2.45, h: 0.95, fontSize: 12.5, fontFace: F, color: MUTED, margin: 0 });
});
s.addText("Every number in this deck comes from committed checkpoints, logs, and eval records — reproducible from the repo.", {
  x: M, y: 5.9, w: W - 2 * M, h: 0.5, fontSize: 15, fontFace: F, color: TEXT, margin: 0,
});
srcLine(s, "Source: results/models/, results/logs/, results/eval_*.json (marl-pong repo)");
s.addNotes("If asked about MLflow/SB3 deviations: custom PPO was TASK.md option (b); CSV logs replace MLflow; both documented in report section 2.");

// ---------- 4 · Finding 1: no equilibrium ----------
s = p.addSlide();
s.background = { color: BG };
title(s, "FINDING 1 · NON-STATIONARITY", "Learning is a wave, not a trend");
s.addImage({ path: "results/plots/curves_2v2.png", x: M, y: 1.75, w: 6.9, h: 3.94 });
s.addText("Self-play team A return, smoothed — one curve per seed", {
  x: M, y: 5.75, w: 6.9, h: 0.3, fontSize: 12, fontFace: F, color: MUTED, margin: 0, italic: true,
});
s.addText("Oscillation, not convergence", {
  x: 7.9, y: 1.9, w: 5.0, h: 0.8, fontSize: 24, fontFace: F, bold: true, color: PRIMARY, margin: 0,
});
s.addText([
  { text: "Dominance swings in ~300k-step waves — each team's improvement is the other's environment change", options: { bullet: bu(), breakLine: true } },
  { text: "All three seeds decline together after ~800k steps", options: { bullet: bu(), breakLine: true } },
  { text: "Which team leads at 1M is a coin flip: seed 1 → A, seeds 0/2 → B", options: { bullet: bu() } },
], { x: 7.9, y: 2.8, w: 5.0, h: 2.6, fontSize: 15, fontFace: F, color: TEXT, paraSpaceAfter: 10, margin: 0 });
s.addText("\u201CThe opponent is the environment.\u201D", {
  x: 7.9, y: 5.3, w: 5.0, h: 0.5, fontSize: 16, fontFace: F, italic: true, color: ACCENT, margin: 0,
});
srcLine(s, "Source: results/logs/2v2_seed{0,1,2}_1M.csv — per-iteration returns, 51-iter rolling mean");
s.addNotes("This is the non-stationarity challenge from Li et al. demonstrated, not just cited. Nobody converged; the optimization target keeps moving. Present the decline at the end honestly: an arms race, not undertraining.");

// ---------- 5 · Finding 2: vs opposition ----------
s = p.addSlide();
s.background = { color: BG };
title(s, "FINDING 2 · SKILL CALIBRATION", "Beats random, loses to a hand-written tracker");
s.addChart(p.charts.BAR, [
  { name: "2v2 @ 1M", labels: ["vs random", "vs self-play", "vs scripted tracker"], values: [0.68, 0.43, 0.02] },
  { name: "1v1 @ 300k", labels: ["vs random", "vs self-play", "vs scripted tracker"], values: [0.48, 0.39, 0.0] },
], {
  x: M, y: 1.8, w: 7.6, h: 4.5, barDir: "col", barGapWidthPct: 60,
  chartColors: [ACCENT, "8B98B3"],
  chartArea: { fill: { color: "FFFFFF" } },
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED,
  catAxisLabelFontSize: 13, valAxisLabelFontSize: 12,
  valAxisMaxVal: 1.0, valAxisMinVal: 0, valAxisMajorUnit: 0.25,
  valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: TEXT, dataLabelFontSize: 12, dataLabelFormatCode: "0.00",
  showLegend: true, legendPos: "b", legendColor: MUTED, legendFontSize: 13,
  showTitle: true, title: "Team A win rate (blue team), 60 episodes per bar", titleColor: TEXT, titleFontSize: 14,
});
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 8.6, y: 2.1, w: 4.2, h: 1.7, rectRadius: 0.08, fill: { color: TINT } });
s.addText("±0.27", { x: 8.85, y: 2.25, w: 3.7, h: 0.7, fontSize: 40, fontFace: F, bold: true, color: ACCENT, margin: 0 });
s.addText("seed spread on the 0.68 win rate vs random — real skill, fragile across seeds", {
  x: 8.85, y: 2.95, w: 3.7, h: 0.8, fontSize: 13, fontFace: F, color: TEXT, margin: 0,
});
s.addText("A scripted ball tracker wins ~99% against every trained team. PPO-learned play is qualitatively behind hand-written defense at this budget.", {
  x: 8.6, y: 4.1, w: 4.2, h: 1.6, fontSize: 14, fontFace: F, color: TEXT, margin: 0,
});
srcLine(s, "Source: results/summary.md — 3 seeds, 60 greedy episodes per run per opponent");
s.addNotes("Calibrated honesty: PPO learned something real (0.68 vs random) but far from scripted-perfect. The spread matters: one seed wins ~90%, another ~50%. Do not oversell.");

// ---------- 6 · Finding 3: structure emerges ----------
s = p.addSlide();
s.background = { color: BG };
title(s, "FINDING 3 · COORDINATION EVIDENCE", "Positional roles emerge — nobody assigned them");
s.addImage({ path: "results/plots/coverage_heatmaps_seed0.png", x: 1.17, y: 2.0, w: 11.0, h: 3.6 });
s.addText("Paddle y-position over normalized episode time, seed 0: each paddle holds a home band plus chase excursions", {
  x: 1.17, y: 5.7, w: 11.0, h: 0.35, fontSize: 13, fontFace: F, color: MUTED, margin: 0, italic: true, align: "center",
});
s.addText("A1 anchors mid-upper, A2 mid-lower with excursions to the top wall — the mirror structure appears on team B. No communication, no role assignment.", {
  x: 1.7, y: 6.15, w: 10.0, h: 0.7, fontSize: 15, fontFace: F, color: TEXT, align: "center", margin: 0,
});
srcLine(s, "Source: results/traj_2v2_seed0_1M.npz — 20 greedy self-play episodes");
s.addNotes("This is the centerpiece: the actual research question answered visually. Structure is unambiguous in the heatmaps. Seeds 1 and 2 show different partitions — the deck's next slides quantify that.");

// ---------- 7 · Specialization twist ----------
s = p.addSlide();
s.background = { color: BG };
title(s, "FINDING 3 · CONTINUED", "Specialization is real — and not what we predicted");
s.addImage({ path: "results/plots/overlap_over_training.png", x: M, y: 1.85, w: 6.9, h: 3.94 });
s.addText("Teammate coverage overlap at checkpoints (lower = more specialized)", {
  x: M, y: 5.85, w: 6.9, h: 0.3, fontSize: 12, fontFace: F, color: MUTED, margin: 0, italic: true,
});
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 7.9, y: 1.95, w: 5.0, h: 1.75, rectRadius: 0.08, fill: { color: TINT } });
s.addText([
  { text: "0.01", options: { fontSize: 40, bold: true, color: ACCENT, breakLine: true } },
  { text: "coverage overlap in seed 1 — a near-disjoint band partition, and the strongest team (0.67 win rate)", options: { fontSize: 13, color: TEXT } },
], { x: 8.15, y: 2.1, w: 4.5, h: 1.5, fontFace: F, margin: 0 });
s.addText([
  { text: "Overlap rises over training in all seeds — early policies barely move; later ones chase hard", options: { bullet: bu(), breakLine: true } },
  { text: "So complementary roles mean distinct anchors with shared pursuit, not disjoint coverage", options: { bullet: bu(), breakLine: true } },
  { text: "n = 3: the specialization-success link is suggestive, not causal", options: { bullet: bu() } },
], { x: 7.9, y: 4.0, w: 5.0, h: 2.2, fontSize: 15, fontFace: F, color: TEXT, paraSpaceAfter: 10, margin: 0 });
srcLine(s, "Source: overlap sweep over iter_*.pt checkpoints, 4 short greedy probes each");
s.addNotes("The metric went the 'wrong' way and that is the interesting part: rising overlap reflects vigorous shared chasing on top of stable home bands. Presenting this honestly beats pretending it trended down.");

// ---------- 8 · Asymmetry ----------
s = p.addSlide();
s.background = { color: BG };
title(s, "FINDING 4 · CREDIT ASSIGNMENT", "Which team dominates is a seed lottery");
s.addChart(p.charts.BAR, [
  { name: "Team A (blue)", labels: ["seed 0", "seed 1", "seed 2"], values: [0.37, 0.67, 0.25] },
  { name: "Team B (red)", labels: ["seed 0", "seed 1", "seed 2"], values: [0.58, 0.08, 0.7] },
], {
  x: M, y: 1.8, w: 7.6, h: 4.5, barDir: "col", barGapWidthPct: 60,
  chartColors: [ACCENT, RED],
  chartArea: { fill: { color: "FFFFFF" } },
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED,
  catAxisLabelFontSize: 13, valAxisLabelFontSize: 12,
  valAxisMaxVal: 1.0, valAxisMinVal: 0, valAxisMajorUnit: 0.25,
  valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: TEXT, dataLabelFontSize: 12, dataLabelFormatCode: "0.00",
  showLegend: true, legendPos: "b", legendColor: MUTED, legendFontSize: 13,
  showTitle: true, title: "Self-play win rates, 2v2 @ 1M (60 episodes)", titleColor: TEXT, titleFontSize: 14,
});
s.addText("Asymmetry entrenches", { x: 8.6, y: 2.1, w: 4.2, h: 0.5, fontSize: 22, fontFace: F, bold: true, color: PRIMARY, margin: 0 });
s.addText([
  { text: "Shared reward cannot attribute a point to a paddle", options: { bullet: bu(), breakLine: true } },
  { text: "Whichever team stumbles into a better joint equilibrium early keeps it", options: { bullet: bu(), breakLine: true } },
  { text: "Same setup, same budget — opposite winners. That variance is the credit-assignment caveat made visible", options: { bullet: bu() } },
], { x: 8.6, y: 2.7, w: 4.2, h: 3.0, fontSize: 15, fontFace: F, color: TEXT, paraSpaceAfter: 10, margin: 0 });
srcLine(s, "Source: results/eval_2v2_seed{0,1,2}_1M.json — self-play, 60 episodes each");
s.addNotes("This explains the footage: the recorded clip shows blue winning 4-1, but across 60 episodes red wins that seed 0.58-0.37. One match is anecdote; the table is the finding. Seed 1 is where blue genuinely dominates.");

// ---------- 9 · Verdict ----------
s = p.addSlide();
s.background = { color: DARK };
s.addText("THE VERDICT", {
  x: M, y: 1.0, w: W - 2 * M, h: 0.4, fontSize: 14, fontFace: F, bold: true,
  color: ACCENT, charSpacing: 4, margin: 0,
});
s.addText("Partial yes.", {
  x: M, y: 1.5, w: W - 2 * M, h: 1.3, fontSize: 64, fontFace: F, bold: true, color: DARKTEXT, margin: 0,
});
s.addText("Complementary positional roles emerge without communication or role assignment — but coordination is a seed lottery, it does not grow monotonically, and win rate alone would have overstated it. Beating random requires zero coordination; the heatmaps are the evidence.", {
  x: M, y: 3.0, w: 11.0, h: 1.6, fontSize: 20, fontFace: F, color: DARKMUTED, margin: 0,
});
const verd = [
  ["Emerges", "distinct home bands in every seed, from identical rewards and local views"],
  ["Varies", "coverage overlap spans 0.01 - 0.28 across seeds at eval"],
  ["Correlates", "the best-partitioned seed is also the strongest team (0.67 win rate)"],
];
verd.forEach(([h, d], i) => {
  const x = M + i * 4.2;
  s.addText(h, { x, y: 5.1, w: 3.9, h: 0.45, fontSize: 20, fontFace: F, bold: true, color: ACCENT, margin: 0 });
  s.addText(d, { x, y: 5.55, w: 3.9, h: 1.0, fontSize: 13.5, fontFace: F, color: DARKMUTED, margin: 0 });
});
s.addNotes("Say the verdict sentence exactly like this, then stop. This is the slide to linger on.");

// ---------- 10 · Closing ----------
s = p.addSlide();
s.background = { color: DARK };
s.addText("LIMITATIONS & NEXT STEPS", {
  x: M, y: 0.9, w: W - 2 * M, h: 0.4, fontSize: 14, fontFace: F, bold: true,
  color: ACCENT, charSpacing: 4, margin: 0,
});
s.addText("What this evidence cannot yet claim", {
  x: M, y: 1.35, w: W - 2 * M, h: 0.9, fontSize: 36, fontFace: F, bold: true, color: DARKTEXT, margin: 0,
});
const lims = [
  ["3 seeds", "the specialization-success correlation needs more runs to be more than suggestive"],
  ["Unswept overlap", "paddle overlap stayed at its 0.4 default — the frozen-before-training sweep was never run"],
  ["No frozen-opponent run", "skill improvement and opponent drift remain entangled — the one experiment that would disentangle them"],
];
lims.forEach(([h, d], i) => {
  const y = 2.7 + i * 1.15;
  s.addText(h, { x: M, y, w: 3.4, h: 0.4, fontSize: 18, fontFace: F, bold: true, color: DARKTEXT, margin: 0 });
  s.addText(d, { x: 4.2, y, w: 8.4, h: 0.9, fontSize: 15, fontFace: F, color: DARKMUTED, margin: 0 });
  if (i < lims.length - 1) s.addShape(p.shapes.LINE, { x: M, y: y + 1.0, w: W - 2 * M, h: 0, line: { color: DARKLINE, width: 0.75 } });
});
s.addText("Repo: marl-pong — checkpoints, logs, eval records, analysis code, and report/report.md are all committed", {
  x: M, y: 6.7, w: W - 2 * M, h: 0.35, fontSize: 12, fontFace: F, color: DARKMUTED, margin: 0,
});
s.addNotes("Close on future work: frozen-opponent condition first, then more seeds. Offer the repo for anyone who wants to rerun the eval.");

p.writeFile({ fileName: "report/marl_pong_presentation.pptx" }).then(() => console.log("deck written"));

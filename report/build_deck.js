// Builds report/marl_pong_presentation.pptx from results/ artifacts.
// Run: npm install --prefix /tmp/deckdeps pptxgenjs && NODE_PATH=/tmp/deckdeps/node_modules node report/build_deck.js
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
s.addText("Reported run s1_5M: 5M self-play steps, seed 0 · 60-episode deterministic eval per opponent · report/report.md", {
  x: M, y: 7.0, w: W - 2 * M, h: 0.3, fontSize: 12, fontFace: F, color: DARKMUTED, margin: 0,
});
s.addNotes("Open with the question, then play the footage (results/videos/). Frame: can teammates coordinate with only local state, teammate state, and a shared win/lose signal? Li et al. 2025 supplies the framing: non-stationarity and credit assignment. The clips use seeds where points are scored; they illustrate, the tables are the evidence.");

// ---------- 2 · Question & setup ----------
s = p.addSlide();
s.background = { color: BG };
title(s, "THE EXPERIMENT", "Coordination without communication");
arena(s, 7.6, 1.75, 5.2, 2.9, { overlapBand: true, label: "Overlapping vertical ranges — nobody owns the middle" });
const rows = [
  ["8-dim observations", "own paddle y/vy, ball x/y/vx/vy, teammate y/vy — opponents excluded by design"],
  ["Shared team reward", "+1 / -1 per point, identical signal to both teammates, no hit shaping"],
  ["3 actions per paddle", "stay / up / down, applied simultaneously every step"],
  ["Independent PPO ×4", "separate 64×64 actor and critic MLPs per paddle, CPU-only; no shared critic, no parameter sharing"],
];
rows.forEach(([h, d], i) => {
  const y = 1.8 + i * 1.15;
  s.addText(h, { x: M, y, w: 6.6, h: 0.35, fontSize: 18, fontFace: F, bold: true, color: PRIMARY, margin: 0 });
  s.addText(d, { x: M, y: y + 0.38, w: 6.6, h: 0.55, fontSize: 14, fontFace: F, color: MUTED, margin: 0 });
  if (i < rows.length - 1) s.addShape(p.shapes.LINE, { x: M, y: y + 1.0, w: 6.6, h: 0, line: { color: "E2E8F0", width: 0.75 } });
});
s.addNotes("Stress the design constraint: excluding opponent observations is deliberate; it keeps the research question about local+teammate coordination. Overlapping teammates resolve a contact by closest paddle, so neither has a built-in priority in the shared band.");

// ---------- 3 · Protocol ----------
s = p.addSlide();
s.background = { color: BG };
title(s, "PROTOCOL", "Calibrate first, then train, then evaluate greedily");
const stats = [
  ["0.15", "frozen paddle overlap", "picked with scripted agents only, before any PPO run (+ ball speed ×1.15)"],
  ["4", "independent PPO learners", "rollout 4096 · 6 epochs · minibatch 512 · λ 0.99 · entropy 0.01→0.001"],
  ["5M", "self-play steps", "seed 0; chosen from a hyperparameter sweep and longer 3-seed runs"],
  ["60", "eval episodes × 4 opponents", "greedy play vs self-play, random, reactive and predictive trackers"],
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
s.addText("Every number in this deck comes from the committed checkpoint, log, and eval records — reproducible from the repo.", {
  x: M, y: 5.9, w: W - 2 * M, h: 0.5, fontSize: 15, fontFace: F, color: TEXT, margin: 0,
});
srcLine(s, "Source: results/models/s1_5M/run_config.json, results/calibration_v2.json, results/eval_s1_5M*.json");
s.addNotes("If asked about MLflow/SB3 deviations: custom PPO was TASK.md option (b); CSV logs replace MLflow. Be upfront that the showcased run is one seed selected from several candidates; slide 8 shows the cross-seed picture.");

// ---------- 4 · Finding 1: non-stationarity ----------
s = p.addSlide();
s.background = { color: BG };
title(s, "FINDING 1 · NON-STATIONARITY", "The opponent is the environment");
s.addImage({ path: "results/plots/curves_mean_return_A.png", x: M, y: 1.75, w: 6.9, h: 3.94 });
s.addText("Self-play team A return (51-iteration rolling mean)", {
  x: M, y: 5.75, w: 6.9, h: 0.3, fontSize: 12, fontFace: F, color: MUTED, margin: 0, italic: true,
});
s.addText("Collapse, recover, settle", {
  x: 7.9, y: 1.9, w: 5.0, h: 0.8, fontSize: 24, fontFace: F, bold: true, color: PRIMARY, margin: 0,
});
s.addText([
  { text: "Team B learns to return first: team A's return dives to −3.4 by ~400k steps", options: { bullet: bu(), breakLine: true } },
  { text: "Team A catches up and is positive again by ~1.2M; peaks near +1 at 2.5M", options: { bullet: bu(), breakLine: true } },
  { text: "Last 2M steps settle in a +0.3 to +0.5 band — stable, but not symmetric", options: { bullet: bu() } },
], { x: 7.9, y: 2.8, w: 5.0, h: 2.6, fontSize: 15, fontFace: F, color: TEXT, paraSpaceAfter: 10, margin: 0 });
srcLine(s, "Source: results/logs/s1_5M.csv — per-iteration returns");
s.addNotes("Non-stationarity from Li et al. shown, not just cited: each team's improvement is the other team's environment change. This run does settle, with team A ahead.");

// ---------- 5 · Finding 2: skill calibration ----------
s = p.addSlide();
s.background = { color: BG };
title(s, "FINDING 2 · SKILL CALIBRATION", "Beats the reactive tracker, not the predictive one");
s.addChart(p.charts.BAR, [
  { name: "Team A win", labels: ["vs random", "vs reactive tracker", "vs self-play", "vs predictive tracker"], values: [1.0, 0.63, 0.53, 0.0] },
  { name: "Team B win", labels: ["vs random", "vs reactive tracker", "vs self-play", "vs predictive tracker"], values: [0.0, 0.07, 0.03, 0.13] },
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
  showTitle: true, title: "Win rates, 60 greedy episodes per opponent (the rest are draws)", titleColor: TEXT, titleFontSize: 14,
});
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 8.6, y: 2.1, w: 4.2, h: 1.7, rectRadius: 0.08, fill: { color: TINT } });
s.addText("0.63", { x: 8.85, y: 2.25, w: 3.7, h: 0.7, fontSize: 40, fontFace: F, bold: true, color: ACCENT, margin: 0 });
s.addText("win rate vs the reactive scripted tracker (0.07 losses), +0.68 points per game", {
  x: 8.85, y: 2.95, w: 3.7, h: 0.8, fontSize: 13, fontFace: F, color: TEXT, margin: 0,
});
s.addText("Tracker games all hit the 2000-step cap: wins are low-scoring leads, not first-to-5. The predictive tracker is unbeaten — 87% scoreless draws.", {
  x: 8.6, y: 4.1, w: 4.2, h: 1.6, fontSize: 14, fontFace: F, color: TEXT, margin: 0,
});
srcLine(s, "Source: results/summary.md, results/eval_s1_5M*.json");
s.addNotes("Calibrated honesty: real skill (perfect vs random, majority vs the reactive tracker), but a projected-intercept controller still defends better. Self-play is lopsided: A 0.53, B 0.03.");

// ---------- 6 · Finding 3: structure emerges ----------
s = p.addSlide();
s.background = { color: BG };
title(s, "FINDING 3 · COORDINATION EVIDENCE", "Both guard the middle, each sweeps its side");
s.addImage({ path: "results/plots/coverage_heatmaps.png", x: 1.17, y: 2.0, w: 11.0, h: 3.6 });
s.addText("Paddle y-position over normalized episode time, 60 greedy self-play episodes (+y is down)", {
  x: 1.17, y: 5.7, w: 11.0, h: 0.35, fontSize: 13, fontFace: F, color: MUTED, margin: 0, italic: true, align: "center",
});
s.addText("Each paddle parks at the inner edge of its range (the shared middle band) and sweeps out toward its own wall. Team B mirrors it. No communication, no role assignment.", {
  x: 1.7, y: 6.15, w: 10.0, h: 0.7, fontSize: 15, fontFace: F, color: TEXT, align: "center", margin: 0,
});
srcLine(s, "Source: results/traj_s1_5M.npz");
s.addNotes("The occupancy structure is clear, but occupancy alone is not coordination; the next slide shows who actually returns the ball where.");

// ---------- 7 · Functional roles ----------
s = p.addSlide();
s.background = { color: BG };
title(s, "FINDING 3 · CONTINUED", "A primary middle defender emerges");
s.addImage({ path: "results/plots/overlap_over_training.png", x: M, y: 1.85, w: 6.9, h: 3.94 });
s.addText("Team A coverage overlap at checkpoints (short greedy probes)", {
  x: M, y: 5.85, w: 6.9, h: 0.3, fontSize: 12, fontFace: F, color: MUTED, margin: 0, italic: true,
});
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 7.9, y: 1.95, w: 5.0, h: 1.75, rectRadius: 0.08, fill: { color: TINT } });
s.addText([
  { text: "~70%", options: { fontSize: 40, bold: true, color: ACCENT, breakLine: true } },
  { text: "of shared-middle returns made by A2 (2.0 vs 0.9 per game), a role nobody assigned", options: { fontSize: 13, color: TEXT } },
], { x: 8.15, y: 2.1, w: 4.5, h: 1.5, fontFace: F, margin: 0 });
s.addText([
  { text: "Outer-band split (A1 low, A2 high) is forced by the ranges, not evidence by itself", options: { bullet: bu(), breakLine: true } },
  { text: "Overlap rises 0.04 → 0.3 as both converge on the middle: lower overlap ≠ better coordination", options: { bullet: bu(), breakLine: true } },
  { text: "Shared reward, uneven work: A2 takes 57% of team contacts", options: { bullet: bu() } },
], { x: 7.9, y: 4.0, w: 5.0, h: 2.2, fontSize: 15, fontFace: F, color: TEXT, paraSpaceAfter: 10, margin: 0 });
srcLine(s, "Source: results/eval_s1_5M.json (behavior.*.contact_region_share), overlap sweep over iter_*.pt");
s.addNotes("This is the functional evidence: both paddles can reach the middle band, and the team allocates it unevenly but consistently across all four evaluation opponents. Discount the outer bands; geometry decides those.");

// ---------- 8 · Selection & robustness ----------
s = p.addSlide();
s.background = { color: BG };
title(s, "ROBUSTNESS", "More budget, stronger team");
s.addChart(p.charts.BAR, [
  { name: "vs random", labels: ["1M (1 seed)", "3M (3-seed mean)", "5M (s1_5M)"], values: [0.97, 0.98, 1.0] },
  { name: "vs reactive tracker", labels: ["1M (1 seed)", "3M (3-seed mean)", "5M (s1_5M)"], values: [0.12, 0.24, 0.63] },
], {
  x: M, y: 1.8, w: 7.6, h: 4.5, barDir: "col", barGapWidthPct: 60,
  chartColors: ["8B98B3", ACCENT],
  chartArea: { fill: { color: "FFFFFF" } },
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED,
  catAxisLabelFontSize: 13, valAxisLabelFontSize: 12,
  valAxisMaxVal: 1.0, valAxisMinVal: 0, valAxisMajorUnit: 0.25,
  valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: TEXT, dataLabelFontSize: 12, dataLabelFormatCode: "0.00",
  showLegend: true, legendPos: "b", legendColor: MUTED, legendFontSize: 13,
  showTitle: true, title: "Team A win rate, same configuration, growing budget", titleColor: TEXT, titleFontSize: 14,
});
s.addText("Why this run", { x: 8.6, y: 2.1, w: 4.2, h: 0.5, fontSize: 22, fontFace: F, bold: true, color: PRIMARY, margin: 0 });
s.addText([
  { text: "Rollout 4096 beat rollout 1024, higher entropy, γ 0.999 and lower lr in a 1M sweep", options: { bullet: bu(), breakLine: true } },
  { text: "Head-to-head, s1_5M's Team A finishes net-ahead of every other candidate's Team B", options: { bullet: bu(), breakLine: true } },
  { text: "Across 3 seeds at 3M the tracker win rate spans 0.12–0.40: seed matters", options: { bullet: bu() } },
], { x: 8.6, y: 2.7, w: 4.2, h: 3.0, fontSize: 15, fontFace: F, color: TEXT, paraSpaceAfter: 10, margin: 0 });
srcLine(s, "Source: report/report.md §4.4 — archived comparison runs, same 60-episode protocol");
s.addNotes("Be explicit: s1_5M is the best observed run, selected after comparison. The 3-seed 3M run is the honest cross-seed estimate; the trend with budget is the robust part.");

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
s.addText("Without communication or assigned roles, teammates learn a consistent division of labour (each owns its outer band, one becomes the primary middle defender), and that team beats random play and a reactive tracker. It does not match a predictive hand-written defender, and it rests on one showcased seed.", {
  x: M, y: 3.0, w: 11.5, h: 1.6, fontSize: 20, fontFace: F, color: DARKMUTED, margin: 0,
});
const verd = [
  ["Emerges", "a primary middle defender (A2, ~70% of shared-band returns) from identical rewards"],
  ["Works", "1.00 vs random, 0.63 vs the reactive tracker over 60 greedy games each"],
  ["Bounded", "predictive tracker unbeaten (87% draws); single-seed showcase"],
];
verd.forEach(([h, d], i) => {
  const x = M + i * 4.2;
  s.addText(h, { x, y: 5.1, w: 3.9, h: 0.45, fontSize: 20, fontFace: F, bold: true, color: ACCENT, margin: 0 });
  s.addText(d, { x, y: 5.55, w: 3.9, h: 1.0, fontSize: 13.5, fontFace: F, color: DARKMUTED, margin: 0 });
});
s.addNotes("Say the verdict sentence, then stop. This is the slide to linger on.");

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
  ["Single seed", "s1_5M was selected from several runs; train seeds 1–2 at 5M to make it a protocol-grade claim"],
  ["Step-capped games", "matches against trackers never reach 5 points; wins are low-scoring leads at the cap"],
  ["No frozen-opponent run", "skill growth and opponent drift remain entangled; this is the experiment that would separate them"],
];
lims.forEach(([h, d], i) => {
  const y = 2.7 + i * 1.15;
  s.addText(h, { x: M, y, w: 3.4, h: 0.4, fontSize: 18, fontFace: F, bold: true, color: DARKTEXT, margin: 0 });
  s.addText(d, { x: 4.2, y, w: 8.4, h: 0.9, fontSize: 15, fontFace: F, color: DARKMUTED, margin: 0 });
  if (i < lims.length - 1) s.addShape(p.shapes.LINE, { x: M, y: y + 1.0, w: W - 2 * M, h: 0, line: { color: DARKLINE, width: 0.75 } });
});
s.addText("Repo: marl-pong — checkpoint, logs, eval records, analysis code, and report/report.md are all committed", {
  x: M, y: 6.7, w: W - 2 * M, h: 0.35, fontSize: 12, fontFace: F, color: DARKMUTED, margin: 0,
});
s.addNotes("Close on future work: more seeds at 5M first, then the frozen-opponent condition. Offer the repo for anyone who wants to rerun the eval.");

p.writeFile({ fileName: "report/marl_pong_presentation.pptx" }).then(() => console.log("deck written"));

#!/usr/bin/env node
// design-sync.mjs - one-way code -> design-doc drift checker.
//
// The architecture editor annotates each diagram node with a `codeRef` such as
//   src/physical/hbm/hbm_device.hpp:57::HbmConfig
// The authoritative topology and map live in app.js. This script validates the
// diagram/node/edge/abstraction contract and resolves every ref against the
// source tree, so neither topology nor code linkage can silently go stale.
//
// Usage:
//   node tools/design-sync.mjs            # read-only drift check
//   node tools/design-sync.mjs --fix      # rewrite moved line numbers in app.js
//   node tools/design-sync.mjs --json p   # explicitly write a machine-readable report
//
// Exit code: 0 if the architecture model is valid and all refs are OK (or only
// `planned:`); 1 for any structural error or unresolved ref.

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const editorDir = path.resolve(scriptDir, "..");
const repoRoot = path.resolve(editorDir, "..", "..");
const appJsPath = path.join(editorDir, "app.js");

const args = process.argv.slice(2);
let doFix = false;
let jsonOut = null;

for (let i = 0; i < args.length; i += 1) {
  const arg = args[i];
  if (arg === "--fix") {
    doFix = true;
  } else if (arg === "--json") {
    const value = args[i + 1];
    if (!value || value.startsWith("--")) {
      throw new Error("--json requires an output path");
    }
    jsonOut = path.resolve(value);
    i += 1;
  } else if (arg === "--help" || arg === "-h") {
    console.log("Usage: node tools/design-sync.mjs [--fix] [--json OUTPUT]");
    process.exit(0);
  } else {
    throw new Error(`unknown argument: ${arg}`);
  }
}

// ---- Extract the CODE_REFS object literal out of app.js (data-only, so eval is safe) ----
function extractCodeRefs(appSource) {
  const marker = "const CODE_REFS = {";
  const start = appSource.indexOf(marker);
  if (start < 0) throw new Error("CODE_REFS not found in app.js");
  let i = start + marker.length - 1; // at the '{'
  let depth = 0;
  for (; i < appSource.length; i += 1) {
    const ch = appSource[i];
    if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) break;
    }
  }
  const objText = appSource.slice(start + marker.length - 1, i + 1);
  // eslint-disable-next-line no-eval
  return eval(`(${objText})`);
}

function extractTemplates(appSource) {
  const marker = "const templates = {";
  const start = appSource.indexOf(marker);
  if (start < 0) throw new Error("templates not found in app.js");
  let i = start + marker.length - 1;
  let depth = 0;
  let quote = "";
  let escaped = false;
  for (; i < appSource.length; i += 1) {
    const ch = appSource[i];
    if (quote) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === quote) quote = "";
      continue;
    }
    if (ch === '"' || ch === "'" || ch === "`") {
      quote = ch;
      continue;
    }
    if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) break;
    }
  }
  if (depth !== 0) throw new Error("unterminated templates object in app.js");
  const objText = appSource.slice(start + marker.length - 1, i + 1);
  let edgeCounter = 0;
  const defaultLevel = {
    system: "contract",
    hbm: "device",
    hbf: "device",
    logic: "resource",
    media: "media",
    flow: "resource",
    maintenance: "media",
    output: "evidence",
  };
  const n = (
    id,
    label,
    subtitle,
    type,
    x,
    y,
    w,
    h,
    level = defaultLevel[type] || "resource",
  ) => ({
    id, label, subtitle, type, level, x, y, w, h, codeRef: "",
  });
  const e = (source, target, label = "") => ({
    id: `edge_${edgeCounter += 1}`,
    source,
    target,
    label,
  });
  // The extracted literal is repository-owned architecture data. Supplying
  // inert n/e constructors avoids executing browser/editor behavior.
  // eslint-disable-next-line no-new-func
  return Function("n", "e", `"use strict"; return (${objText});`)(n, e);
}

const VALID_TYPES = new Set([
  "system", "hbm", "hbf", "external", "logic", "media", "flow",
  "maintenance", "output",
]);
const VALID_LEVELS = new Set([
  "source", "contract", "composition", "device", "resource", "media", "evidence",
]);
const REQUIRED_ENTITIES = {
  system_map: [
    "synthetic", "astra", "qwen", "frontier", "trace", "guard", "request",
    "runner", "direct", "streaming", "hbm", "hbf", "external", "summary",
    "timing", "heatmap", "guards",
  ],
  compositions: [
    "input", "window", "direct", "streaming", "hbm", "hbf_ftl",
    "static_map", "static_fabric",
    "stream_ctrl", "buffers", "d2d", "backing", "external",
    "external_pipeline",
    "result",
  ],
  full: [
    "hbm_ctrl", "hbm_pch", "hbf_stack", "logic_die", "mapping", "tsu",
    "hbio", "fch", "fdie", "plane", "subarray", "media_lane", "page_buffer",
    "block", "page",
  ],
  hbm_flow: ["req", "map", "queue", "sched", "row", "prep", "cas", "bus", "refresh", "done"],
  hbf_read: [
    "req", "ingress", "queue", "trans", "split", "tsu", "array",
    "media_lane", "page_buffer", "post", "ecc", "payload", "done",
  ],
  hbf_write: [
    "req", "lookup", "buffer", "flush", "alloc", "gc_check", "gc", "encode",
    "prog", "map", "map_batch", "done",
  ],
  observability: [
    "request", "completion", "hbm_stats", "hbf_stats",
    "external_backing_stats",
    "composition_stats", "traffic", "waf", "heatmap_acc", "summary", "wall",
    "latency", "stage", "busy", "domains", "time_html", "heatmap_html",
    "provenance", "sanity",
  ],
};

function validateArchitectureModel(appSource, codeRefs) {
  const templates = extractTemplates(appSource);
  const problems = [];
  for (const [diagramKey, requiredNodes] of Object.entries(REQUIRED_ENTITIES)) {
    const diagram = templates[diagramKey];
    if (!diagram) {
      problems.push(`missing required diagram ${diagramKey}`);
      continue;
    }
    if (diagram.id !== diagramKey) {
      problems.push(`${diagramKey}: diagram.id is ${JSON.stringify(diagram.id)}`);
    }
    if (!diagram.name || !diagram.description) {
      problems.push(`${diagramKey}: name and description are required`);
    }
    if (!Array.isArray(diagram.nodes) || !Array.isArray(diagram.edges)) {
      problems.push(`${diagramKey}: nodes and edges must be arrays`);
      continue;
    }
    const nodeIds = new Set();
    for (const node of diagram.nodes) {
      const nodePath = `${diagramKey}.${node.id || "<missing-id>"}`;
      if (!node.id || nodeIds.has(node.id)) {
        problems.push(`${nodePath}: missing or duplicate node id`);
      }
      nodeIds.add(node.id);
      if (!VALID_TYPES.has(node.type)) {
        problems.push(`${nodePath}: invalid entity type ${node.type}`);
      }
      if (!VALID_LEVELS.has(node.level)) {
        problems.push(`${nodePath}: invalid abstraction level ${node.level}`);
      }
      if (!node.label || !node.subtitle) {
        problems.push(`${nodePath}: label and subtitle are required`);
      }
      for (const field of ["x", "y", "w", "h"]) {
        if (!Number.isFinite(node[field])) {
          problems.push(`${nodePath}: ${field} must be finite`);
        }
      }
      if (!(node.w > 0) || !(node.h >= 70)) {
        problems.push(`${nodePath}: node must have positive width and height >= 70`);
      }
      if (!Object.prototype.hasOwnProperty.call(codeRefs, nodePath)) {
        problems.push(`${nodePath}: missing CODE_REFS entry`);
      }
    }
    for (const required of requiredNodes) {
      if (!nodeIds.has(required)) {
        problems.push(`${diagramKey}: missing required entity ${required}`);
      }
    }
    for (const edge of diagram.edges) {
      if (!nodeIds.has(edge.source)) {
        problems.push(`${diagramKey}: edge source ${edge.source} does not exist`);
      }
      if (!nodeIds.has(edge.target)) {
        problems.push(`${diagramKey}: edge target ${edge.target} does not exist`);
      }
      if (edge.source === edge.target) {
        problems.push(`${diagramKey}: self-edge on ${edge.source}`);
      }
    }
  }
  for (const diagramKey of Object.keys(templates)) {
    if (!Object.prototype.hasOwnProperty.call(REQUIRED_ENTITIES, diagramKey)) {
      problems.push(`unregistered diagram ${diagramKey}; add it to REQUIRED_ENTITIES`);
    }
  }
  if (/\bCMT\b/.test(appSource)) {
    problems.push(
      "retired CMT terminology remains; use resident per-stack L2P plus checkpoints",
    );
  }
  return problems;
}

// ---- Parse one concrete ref segment "relpath:line[:col]::Symbol::Path" ----
function parseRefSegment(segment) {
  const trimmed = segment.trim();
  if (!trimmed || /^planned:/i.test(trimmed)) return { planned: true, raw: trimmed };
  const symbolStart = trimmed.indexOf("::");
  const location = symbolStart >= 0 ? trimmed.slice(0, symbolStart) : trimmed;
  const symbol = symbolStart >= 0 ? trimmed.slice(symbolStart + 2) : "";
  const m = location.match(/^(.+):(\d+)(?::(\d+))?$/);
  if (!m) return { invalid: true, raw: trimmed };
  return { relPath: m[1].replace(/^\.\//, ""), line: Number(m[2]), symbol, raw: trimmed };
}

const reEsc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

// Return definition/declaration lines for the referenced symbol. Qualified
// function definitions are deliberately preferred over bare `leaf(...)`
// occurrences: the latter also match call sites and used to "repair" refs to
// whichever invocation happened to be nearest the stale line number.
function definitionLines(lines, symbol) {
  const parts = symbol.split("::").filter(Boolean);
  const leaf = parts[parts.length - 1] || symbol;
  const L = reEsc(leaf);
  const qualifiedFunction = parts.length >= 2
    ? new RegExp(`\\b${parts.map(reEsc).join("::")}\\s*\\(`)
    : null;
  if (qualifiedFunction) {
    const matches = lines.flatMap((text, index) =>
      qualifiedFunction.test(text) ? [index + 1] : []);
    if (matches.length) return matches;
  }

  const typeDefinition = new RegExp(`\\b(?:struct|class|enum(?:\\s+class)?)\\s+${L}\\b`);
  const typeMatches = lines.flatMap((text, index) =>
    typeDefinition.test(text) ? [index + 1] : []);
  if (typeMatches.length) return typeMatches;

  const pythonFunction = new RegExp(`^\\s*(?:async\\s+)?def\\s+${L}\\s*\\(`);
  const pythonMatches = lines.flatMap((text, index) =>
    pythonFunction.test(text) ? [index + 1] : []);
  if (pythonMatches.length) return pythonMatches;

  const plainFunction = new RegExp(
    `^\\s*[\\w:<>,*&\\[\\] ]+\\s+${L}\\s*\\(`,
  );
  const plainMatches = lines.flatMap((text, index) =>
    plainFunction.test(text) ? [index + 1] : []);
  if (plainMatches.length) return plainMatches;

  const suffix = new RegExp(`^${L}\\b\\s*(?:[=;:{\\[])`);
  return lines.flatMap((text, index) => {
    const match = text.match(new RegExp(`\\b${L}\\b`));
    if (!match) return [];
    const prefix = text.slice(0, match.index).trim();
    const tail = text.slice(match.index);
    // A member declaration has a type-like prefix. Exclude assignments and
    // member access (`obj.leaf`, `obj->leaf`) that merely use the symbol.
    if (!prefix || !/^[\w:<>,*&\[\] ]+$/.test(prefix) || prefix.includes("=")) return [];
    return suffix.test(tail) ? [index + 1] : [];
  });
}

function resolveSegment(seg) {
  if (seg.planned) return { status: "PLANNED", ...seg };
  if (seg.invalid) return { status: "INVALID", ...seg };
  const abs = path.join(repoRoot, seg.relPath);
  if (!fs.existsSync(abs)) return { status: "MISSING_FILE", ...seg };

  const lines = fs.readFileSync(abs, "utf8").split("\n");
  const segs = seg.symbol.split("::").filter(Boolean);
  const leaf = segs[segs.length - 1] || seg.symbol;
  const wordRe = new RegExp(`\\b${reEsc(leaf)}\\b`);
  const defLines = definitionLines(lines, seg.symbol);
  if (defLines.includes(seg.line)) return { status: "OK", leaf, ...seg };

  // Prefer the nearest real definition/declaration to the recorded line.
  const anyLines = [];
  for (let i = 0; i < lines.length; i += 1) {
    if (leaf && wordRe.test(lines[i])) {
      anyLines.push(i + 1);
    }
  }
  if (defLines.length) {
    defLines.sort((a, b) => Math.abs(a - seg.line) - Math.abs(b - seg.line));
    return { status: "MOVED", leaf, foundLine: defLines[0], ...seg };
  }
  if (anyLines.length) return { status: "UNCERTAIN", leaf, foundLine: anyLines[0], ...seg };
  return { status: "MISSING_SYMBOL", leaf, ...seg };
}

function analyze(appSource) {
  const codeRefs = extractCodeRefs(appSource);
  const rows = [];
  const counts = {
    OK: 0,
    MOVED: 0,
    MISSING_FILE: 0,
    MISSING_SYMBOL: 0,
    UNCERTAIN: 0,
    PLANNED: 0,
    INVALID: 0,
  };

  for (const [key, value] of Object.entries(codeRefs)) {
    const segments = String(value).split(/\s+\/\s+/);
    for (const segText of segments) {
      const result = resolveSegment(parseRefSegment(segText));
      counts[result.status] = (counts[result.status] || 0) + 1;
      rows.push({ node: key, ...result });
    }
  }
  return { codeRefs, counts, rows };
}

function refWithLine(raw, line) {
  const symbolStart = raw.indexOf("::");
  const location = symbolStart >= 0 ? raw.slice(0, symbolStart) : raw;
  const symbol = symbolStart >= 0 ? raw.slice(symbolStart) : "";
  const match = location.match(/^(.+):\d+(:\d+)?$/);
  if (!match) return raw;
  return `${match[1]}:${line}${match[2] || ""}${symbol}`;
}

function replaceRefSegment(source, raw, replacement) {
  // A symbol can prefix another symbol at the same recorded line, e.g.
  // `PseudoChannelState` and `PseudoChannelState::bus_ready_ns`. Match the
  // segment boundary so repairing the type cannot partially rewrite the
  // member ref before that member gets its own (different) line update.
  const segment = new RegExp(`${reEsc(raw)}(?=\\s+\\/\\s+|["'])`, "g");
  return source.replace(segment, () => replacement);
}

// ---- Run ----
const appSource = fs.readFileSync(appJsPath, "utf8");
const initial = analyze(appSource);
let fixedSource = appSource;

if (doFix) {
  for (const result of initial.rows) {
    if (result.status !== "MOVED") continue;
    const replacement = refWithLine(result.raw, result.foundLine);
    fixedSource = replaceRefSegment(fixedSource, result.raw, replacement);
  }
  if (fixedSource !== appSource) {
    fs.writeFileSync(appJsPath, fixedSource);
    console.log("Applied --fix: updated drifted line numbers in app.js");
  }
}

const finalSource = doFix ? fixedSource : appSource;
const { codeRefs, counts, rows } = analyze(finalSource);
const modelProblems = validateArchitectureModel(finalSource, codeRefs);

const ICON = {
  OK: "OK",
  MOVED: "MOVED",
  MISSING_FILE: "MISS",
  MISSING_SYMBOL: "MISS",
  UNCERTAIN: "UNCERTAIN",
  PLANNED: "PLANNED",
  INVALID: "INVALID",
};

// ---- Print grouped report ----
const order = ["MISSING_FILE", "MISSING_SYMBOL", "MOVED", "UNCERTAIN", "INVALID", "PLANNED", "OK"];
const broken = rows.filter((r) => r.status !== "OK" && r.status !== "PLANNED");

console.log(`\nDesign-sync: ${rows.length} code references across ${Object.keys(codeRefs).length} nodes\n`);
for (const status of order) {
  const group = rows.filter((r) => r.status === status);
  if (!group.length) continue;
  for (const r of group) {
    const where = r.foundLine ? `  (recorded ${r.line} -> found ${r.foundLine})` : "";
    const detail = r.status === "OK" ? r.raw : `${r.raw}${where}`;
    console.log(`  ${ICON[r.status].padEnd(9)} ${r.status.padEnd(14)} ${r.node.padEnd(22)} ${detail}`);
  }
  console.log("");
}

const summary = order
  .filter((s) => counts[s])
  .map((s) => `${s.toLowerCase()}=${counts[s]}`)
  .join("   ");
console.log(`Summary: ${summary}\n`);
if (modelProblems.length) {
  console.log("Architecture-model errors:\n");
  modelProblems.forEach((problem) => console.log(`  INVALID   ${problem}`));
  console.log("");
} else {
  console.log("Architecture model: OK\n");
}

if (jsonOut) {
  fs.writeFileSync(jsonOut, JSON.stringify({
    generatedFrom: "app.js templates + CODE_REFS",
    counts,
    rows,
    architectureModel: {
      valid: modelProblems.length === 0,
      problems: modelProblems,
    },
  }, null, 2));
  console.log(`Report written to ${path.relative(repoRoot, jsonOut)}`);
}

process.exit(broken.length || modelProblems.length ? 1 : 0);

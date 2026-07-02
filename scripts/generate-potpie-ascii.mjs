#!/usr/bin/env node
/**
 * Generates ASCII art from potpie/cli/ui/assets/potpie.svg (sharp).
 * Run from the repository root:
 *   node scripts/generate-potpie-ascii.mjs
 *
 * Same algorithm as potpie-vscode-extension/scripts/generate-potpie-ascii.mjs
 * Writes potpie/cli/ui/assets/potpie-logo-static.json
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");
const svgPath = path.join(
  projectRoot,
  "potpie/cli/ui/assets/potpie.svg",
);
const outputPath = path.join(
  projectRoot,
  "potpie/cli/ui/assets/potpie-logo-static.json",
);

// Extension default is 56×28; scaled to fit intro viewport (96×20) without clipping
const HEIGHT = 16;
const WIDTH = 32;

const VIEWPORT_WIDTH = 96;
const VIEWPORT_HEIGHT = 20;
const LOGO_COLOR = "#B6E343";

const CHARS = " .:-=+*#%@";

function chompToken(lines) {
  const preferred = "@#%*+=";
  for (const ch of preferred) {
    for (const line of lines) {
      if (line.includes(ch)) {
        return ch.repeat(2);
      }
    }
  }
  for (const line of lines) {
    for (const ch of line) {
      if (ch !== " " && ch !== ".") {
        return ch.repeat(2);
      }
    }
  }
  return "@@";
}

async function main() {
  let sharp;
  try {
    const sharpModule = await import("sharp");
    sharp = sharpModule.default;
  } catch {
    console.warn("⚠️  'sharp' not found. Install it with: npm install --save-dev sharp");
    process.exit(1);
  }

  if (!fs.existsSync(svgPath)) {
    console.error(`SVG not found: ${svgPath}`);
    process.exit(1);
  }

  const { data, info } = await sharp(svgPath)
    .resize(WIDTH, HEIGHT, { fit: "fill" })
    .raw()
    .ensureAlpha()
    .toBuffer({ resolveWithObject: true });

  const { width, height, channels } = info;
  const lines = [];

  for (let y = 0; y < height; y++) {
    let row = "";
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * channels;
      const a = channels === 4 ? data[i + 3] : 255;
      const alpha = a / 255;
      const idx = Math.min(Math.floor(alpha * CHARS.length), CHARS.length - 1);
      row += CHARS[idx];
    }
    lines.push(row);
  }

  while (lines.length > 0 && /^\s*$/.test(lines[0])) lines.shift();
  while (lines.length > 0 && /^\s*$/.test(lines[lines.length - 1])) lines.pop();

  const payload = {
    version: 2,
    source: "generate-potpie-ascii.mjs",
    viewport_width: VIEWPORT_WIDTH,
    viewport_height: VIEWPORT_HEIGHT,
    color: LOGO_COLOR,
    lines,
    chomp_token: chompToken(lines),
  };

  fs.writeFileSync(outputPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  console.log(
    `Wrote ${outputPath} (${lines.length} lines, token=${JSON.stringify(payload.chomp_token)})`,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

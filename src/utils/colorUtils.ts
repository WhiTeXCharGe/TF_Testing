// Deterministic color utilities — mirrors yaml_to_suother_like_excel.py palette logic.

/** 30-color module palette (exact copy of MODULE_PALETTE in the Python tool). */
const MODULE_PALETTE = [
  '#FFE599','#FFD966','#F9CB9C','#F6B26B','#FCE5CD',
  '#B6D7A8','#93C47D','#6AA84F','#D9EAD3','#E2EFDA',
  '#9FC5E8','#6FA8DC','#3D85C6','#A4C2F4','#CFE2F3',
  '#B4A7D6','#8E7CC3','#674EA7','#D9D2E9','#EAD1DC',
  '#A2C4C9','#76A5AF','#45818E','#D0E0E3','#DDEBF7',
  '#C27BA0','#D5A6BD','#E6B8AF','#F8CBAD','#FFF2CC',
];

/** Fallback module fill for unknown codes (matches Python unknown_fill). */
export const MODULE_UNKNOWN_COLOR = '#FFF2CC';

/** 16-color company palette (exact copy of PALETTE_FIXED in the Python tool). */
const COMPANY_PALETTE = [
  '#FFF2CC','#D9EAD3','#CFE2F3','#D9D2E9','#FCE5CD','#EAD1DC','#D0E0E3','#E2EFDA',
  '#DEEAF6','#E7E6E6','#C9DAF8','#D9E1F2','#D0CECE','#E2F0D9','#DDEBF7','#F8CBAD',
];

/**
 * Deterministic unsigned-32-bit XOR hash of a string.
 * Equivalent to: h = 5381; for each char: h = ((h << 5) + h) ^ char
 */
function hashStr(s: string): number {
  let h = 5381;
  for (let i = 0; i < s.length; i++) {
    h = Math.imul(h, 33) ^ s.charCodeAt(i);
  }
  return h >>> 0; // unsigned 32-bit
}

/**
 * Return the text before the first '_' in a module code.
 * "530N02621A_P1" → "530N02621A"
 * "530N02621A"    → "530N02621A"
 */
export function normalizeModuleCode(code: string): string {
  const idx = code.indexOf('_');
  return idx === -1 ? code : code.substring(0, idx);
}

/** Stable hex color for a module base code (hash-based — used as a fallback). */
export function moduleColor(baseCode: string): string {
  return MODULE_PALETTE[hashStr(baseCode) % MODULE_PALETTE.length];
}

/**
 * Assign module colors by FIRST-OCCURRENCE ORDER (faithful to the Python
 * build_module_fill_by_order): the first time a normalized base code appears,
 * it takes the next palette slot; suffix variants of the same base reuse it.
 *
 * @param baseCodesInOrder normalized base codes in workflow_task_list order
 * @returns Map<baseCode, hexColor>
 */
export function assignModuleColorsByOrder(baseCodesInOrder: string[]): Map<string, string> {
  const baseToColor = new Map<string, string>();
  let idx = 0;
  for (const base of baseCodesInOrder) {
    if (!baseToColor.has(base)) {
      baseToColor.set(base, MODULE_PALETTE[idx % MODULE_PALETTE.length]);
      idx++;
    }
  }
  return baseToColor;
}

/** Stable hex color for a company key. */
export function companyColor(key: string): string {
  return COMPANY_PALETTE[hashStr(key) % COMPANY_PALETTE.length];
}

/** Hex color → rgba string with alpha. */
export function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

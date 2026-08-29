import { SCAN_ITEMS, type ScanResult } from "./receipt-scan-items";

/** เรียก Google AI Studio (Gemini API) ตรง ๆ ด้วยคีย์ของร้านเอง */
const API_BASE = "https://generativelanguage.googleapis.com/v1beta";

/** โมเดลที่ลองตามลำดับ: เร็ว -> ละเอียด (override ได้ด้วย env GEMINI_MODEL คั่นด้วย ,) */
const DEFAULT_MODELS = ["gemini-3.6-flash", "gemini-2.5-pro"];

const properties = Object.fromEntries(
  SCAN_ITEMS.map((i) => [i.key, { type: "number", description: `จำนวนที่ขายได้ของ ${i.label}` }]),
);

/** Gemini structured output: บังคับให้ตอบเป็น JSON ตามสคีมานี้ */
const RESPONSE_SCHEMA = {
  type: "object",
  properties,
  required: SCAN_ITEMS.map((i) => i.key),
};

const SYSTEM =
  "คุณคือผู้ช่วยอ่านใบสรุปยอดขาย (ภาษาไทย) จากรูปถ่าย " +
  "อ่านทุกบรรทัดในรูปอย่างละเอียด รวมถึงตัวเลขที่พิมพ์เบาหรือเอียง " +
  "ให้ดึงเฉพาะจำนวนที่ขายได้ (จำนวนหน่วย/ชิ้น ไม่ใช่ยอดเงิน ไม่ใช่ราคาต่อหน่วย) ของรายการที่กำหนดในสคีมาเท่านั้น " +
  "กฎสำคัญ: " +
  "(1) รายการน้ำอัดลมแก้ว ให้จับเฉพาะชื่อที่ลงท้ายด้วยขนาดแก้ว 16oz / 22oz / 32oz (หรือ 16 ออนซ์ / 22 ออนซ์ / 32 ออนซ์) แล้วรวมทุกรสของขนาดเดียวกันเข้าช่องขนาดนั้น " +
  "(2) รายการโปรโมชั่น เช่น 'ซื้อโค้ก 32oz แก้วที่ 2 ราคา 15 บาท' หรือโปรโมชั่นแก้วที่ 2 ให้ใส่จำนวนลงช่องโปรโมชั่น ห้ามนำไปรวมกับแก้ว 32oz " +
  "(3) น้ำอัดลมกระป๋อง ให้จับเฉพาะชื่อที่ลงท้ายด้วยคำว่า 'กระป๋อง' (หรือ 'ขวด' สำหรับชเวปส์/มินิทเมด) ห้ามนำยอดของกระป๋องไปใส่ช่องแบบแก้ว และห้ามนำยอดแบบแก้วมาใส่ช่องกระป๋อง " +
  "(4) น้ำทิพย์ (น้ำดื่มขวด) ให้ใส่ช่องน้ำทิพย์ " +
  "(5) นอกเหนือจากนี้ห้ามใส่เลย เช่น กาแฟ/เครื่องดื่มร้อน ชา นม เบเกอรี่ รีฟิว กระบอกน้ำ ของแถม ห้ามสรุปหรือเดา " +
  "ถ้ารายการเดียวกันมีหลายบรรทัด ให้รวมจำนวนกัน " +
  "รายการใดที่ไม่ปรากฏในใบสรุปเลย ให้ใส่ 0 ห้ามเดา";

type KeyedError = Error & { status?: number };

const keyedError = (message: string, status?: number): KeyedError => {
  const err = new Error(message) as KeyedError;
  if (status != null) err.status = status;
  return err;
};

let envLoaded = false;

/** dev: ดึงคีย์จาก .env.local / .env เข้ามาใน process.env (prod ใช้ env จริงของโฮสต์) */
function loadEnvFilesOnce(): void {
  if (envLoaded) return;
  envLoaded = true;
  const proc = process as unknown as { loadEnvFile?: (path?: string) => void };
  if (typeof proc.loadEnvFile !== "function") return;
  for (const file of [".env.local", ".env"]) {
    try {
      proc.loadEnvFile(file);
    } catch {
      /* ไม่มีไฟล์นั้น — ข้าม */
    }
  }
}

/** อ่านคีย์จากทุกที่ที่รันไทม์อาจเก็บไว้ (process.env หรือ binding ของ worker) */
function readApiKey(): string | undefined {
  loadEnvFilesOnce();
  const fromProcess = process.env["GEMINI_API_KEY"] || process.env["GOOGLE_AI_API_KEY"];
  if (fromProcess) return fromProcess;
  const g = globalThis as Record<string, unknown>;
  const cf = g["Cloudflare"] as Record<string, unknown> | undefined;
  for (const holder of [g["env"], g["__env__"], cf?.["env"]]) {
    if (holder && typeof holder === "object") {
      const v =
        (holder as Record<string, unknown>)["GEMINI_API_KEY"] ??
        (holder as Record<string, unknown>)["GOOGLE_AI_API_KEY"];
      if (typeof v === "string" && v) return v;
    }
  }
  return undefined;
}

function readModels(): string[] {
  const raw = process.env["GEMINI_MODEL"]?.trim();
  if (!raw) return DEFAULT_MODELS;
  const list = raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  return list.length ? list : DEFAULT_MODELS;
}

/** แยก data:image/jpeg;base64,xxxx -> { mimeType, data(base64) } */
function parseImageDataUrl(dataUrl: string): { mimeType: string; data: string } {
  const match = /^data:([^;,]+);base64,(.+)$/s.exec(dataUrl);
  if (!match) throw keyedError("รูปนี้ส่งให้ระบบอ่านไม่ได้ กรุณาถ่ายใหม่", 400);
  return { mimeType: match[1] || "image/jpeg", data: match[2]! };
}

function stripJsonFence(text: string): string {
  const trimmed = text.trim();
  const fenced = /^```(?:json)?\s*([\s\S]*?)\s*```$/.exec(trimmed);
  return fenced ? fenced[1]!.trim() : trimmed;
}

async function callGemini(
  apiKey: string,
  model: string,
  image: { mimeType: string; data: string },
): Promise<Record<string, unknown> | undefined> {
  const res = await fetch(`${API_BASE}/models/${encodeURIComponent(model)}:generateContent`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-goog-api-key": apiKey,
    },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: SYSTEM }] },
      contents: [
        {
          role: "user",
          parts: [
            {
              text: "อ่านใบสรุปยอดนี้ให้ครบทุกบรรทัด แล้วส่งจำนวนที่ขายได้ของรายการที่กำหนด",
            },
            { inlineData: { mimeType: image.mimeType, data: image.data } },
          ],
        },
      ],
      generationConfig: {
        temperature: 0,
        responseMimeType: "application/json",
        responseSchema: RESPONSE_SCHEMA,
      },
    }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw keyedError(`gemini ${res.status}: ${text.slice(0, 300)}`, res.status);
  }

  const json = (await res.json()) as {
    candidates?: { content?: { parts?: { text?: string }[] } }[];
  };
  const text = (json.candidates?.[0]?.content?.parts ?? [])
    .map((p) => p.text ?? "")
    .join("")
    .trim();
  if (!text) return undefined;
  try {
    const parsed = JSON.parse(stripJsonFence(text)) as unknown;
    return parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>) : undefined;
  } catch {
    return undefined;
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export async function scanReceipt(imageDataUrl: string): Promise<ScanResult> {
  const apiKey = readApiKey();
  if (!apiKey) throw new Error("ระบบ AI ยังไม่พร้อมใช้งาน (ยังไม่ได้ตั้งค่า GEMINI_API_KEY)");

  const image = parseImageDataUrl(imageDataUrl);
  const models = readModels();

  let lastError: unknown;
  for (let attempt = 0; attempt < models.length; attempt++) {
    try {
      const parsed = await callGemini(apiKey, models[attempt]!, image);
      if (parsed) {
        const out: ScanResult = {};
        for (const item of SCAN_ITEMS) {
          const v = Number(parsed[item.key]);
          if (Number.isFinite(v) && v > 0) out[item.key] = Math.round(v);
        }
        if (Object.keys(out).length > 0) return out;
      }
      lastError = new Error("อ่านตัวเลขจากรูปไม่ได้");
    } catch (error) {
      lastError = error;
      const status = (error as KeyedError).status;
      // คำขอผิด/คีย์ผิด/รูปเสีย การส่งซ้ำจะได้ผลเดิม
      if (status === 400 || status === 401 || status === 403) break;
    }
    if (attempt < models.length - 1) await sleep(700 * (attempt + 1));
  }

  const status = (lastError as KeyedError | undefined)?.status;
  if (status === 429) throw new Error("ระบบ AI ใช้งานหนักอยู่ รอสักครู่แล้วลองอีกครั้ง");
  if (status === 401 || status === 403)
    throw new Error("ระบบ AI เชื่อมต่อไม่สำเร็จ กรุณาตรวจสอบ GEMINI_API_KEY");
  if (status === 400)
    throw new Error("รูปนี้ส่งให้ระบบอ่านไม่ได้ กรุณาถ่ายใหม่ให้เห็นกระดาษเต็มแผ่น");
  throw new Error("อ่านตัวเลขจากรูปไม่ได้ ลองถ่ายให้ชัด/ตรง แล้วลองอีกครั้ง");
}

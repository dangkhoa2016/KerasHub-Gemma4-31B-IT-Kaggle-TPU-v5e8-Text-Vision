#!/usr/bin/env node
import fs from "node:fs";

const baseUrl = (
  process.env.GEMMA4_URL ||
  "http://127.0.0.1:7860"
).replace(/\/$/, "");

const apiKey =
  process.env.GEMMA4_API_KEY || "";

if (!apiKey) {
  throw new Error("Set GEMMA4_API_KEY");
}

async function request(path, options = {}) {
  const response = await fetch(
    baseUrl + path,
    {
      ...options,
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    },
  );
  const body = await response.json();
  return [response.status, body];
}

async function poll(
  jobId,
  timeoutMs = 1800000,
) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const [status, body] = await request(
      `/result/${jobId}`
    );
    if (status === 200) return body;
    if (status !== 202) {
      throw new Error(JSON.stringify(body));
    }
    await new Promise(
      r => setTimeout(r, 1000)
    );
  }
  throw new Error(
    `Timed out waiting for ${jobId}`
  );
}

export async function generate(
  prompt,
  system = "",
  maxNewTokens = 128,
) {
  const [status, body] = await request(
    "/generate/async",
    {
      method: "POST",
      body: JSON.stringify({
        prompt,
        system,
        max_new_tokens: maxNewTokens,
      }),
    },
  );
  if (status !== 202) {
    throw new Error(JSON.stringify(body));
  }
  return poll(body.job_id);
}

export async function generateImage(
  path,
  prompt,
  system = "",
  maxNewTokens = 128,
) {
  const image_base64 =
    fs.readFileSync(path).toString("base64");

  const [status, body] = await request(
    "/generate/image/async",
    {
      method: "POST",
      body: JSON.stringify({
        prompt,
        system,
        max_new_tokens: maxNewTokens,
        image_base64,
      }),
    },
  );

  if (status !== 202) {
    throw new Error(JSON.stringify(body));
  }
  return poll(body.job_id);
}

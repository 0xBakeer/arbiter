// The only network code in the showcase. The browser UIs and `measure.mjs` both go through it,
// so the two timings mean the same thing everywhere:
//
//   inference_ms   `latency_ms` from the response body: what the server spent, queue included
//   roundtrip_ms   what the caller waited for: inference plus HTTP, JSON and the event loop
//
// Everything the showcase reports as "decisions per second" is derived from roundtrip_ms.

const now = () => (globalThis.performance ? globalThis.performance.now() : Date.now());

export function createClient({ baseUrl = "", model = "auto", apiKey = null, fetchImpl = null } = {}) {
  const doFetch = fetchImpl || globalThis.fetch.bind(globalThis);
  const url = `${String(baseUrl).replace(/\/+$/, "")}/v1/systemone`;
  const counters = { calls: 0, questions: 0, tokens: 0, errors: 0 };

  // One call = one forward pass on the server, whatever the number of questions. That is the
  // property the `mines` game exists to show.
  async function ask(state, questions) {
    const headers = { "content-type": "application/json" };
    if (apiKey) headers.authorization = `Bearer ${apiKey}`;
    const body = JSON.stringify({ state, model, questions });
    const t0 = now();
    const res = await doFetch(url, { method: "POST", headers, body });
    const text = await res.text();
    const roundtrip_ms = now() - t0;
    if (!res.ok) {
      counters.errors += 1;
      throw new Error(`systemone ${res.status}: ${text.slice(0, 300)}`);
    }
    const out = JSON.parse(text);
    counters.calls += 1;
    counters.questions += Object.keys(questions).length;
    counters.tokens += (out.usage && out.usage.input_tokens) || 0;
    return {
      answers: out.answers,
      routing: out.routing,
      usage: out.usage,
      model: out.model,
      latency_ms: out.latency_ms,
      inference_ms: out.latency_ms,
      roundtrip_ms,
    };
  }

  async function ready() {
    const res = await doFetch(`${String(baseUrl).replace(/\/+$/, "")}/readyz`);
    if (!res.ok) return { ready: false, status: res.status };
    return { ready: true, ...(await res.json()) };
  }

  return { ask, ready, counters, url, model };
}

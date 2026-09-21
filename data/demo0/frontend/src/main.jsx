import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const palette = [
  [230, 57, 70], [46, 196, 182], [69, 123, 255], [255, 183, 3],
  [131, 56, 236], [0, 168, 107], [247, 127, 0], [255, 99, 164],
];

function heatColor(value) {
  const v = Math.max(0, Math.min(1, value));
  const r = Math.round(255 * Math.min(1, 1.7 * v));
  const g = Math.round(255 * Math.max(0, 1 - Math.abs(v - 0.55) * 2.2));
  const b = Math.round(255 * Math.max(0, 1 - 1.8 * v));
  return `rgb(${r},${g},${b})`;
}

function CanvasView({ imageUrl, query, kind, alpha, onToken }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    if (!imageUrl || !query) return;
    const image = new Image();
    image.onload = () => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      canvas.width = image.width;
      canvas.height = image.height;
      ctx.drawImage(image, 0, 0);
      const { width, height, stride } = query.grid;
      if (kind === "reference") {
        const heatmap = query.dense_similarity_heatmap;
        ctx.globalAlpha = alpha;
        for (let y = 0; y < height; y += 1) {
          for (let x = 0; x < width; x += 1) {
            ctx.fillStyle = heatColor(heatmap[y][x]);
            ctx.fillRect(x * stride, y * stride, stride, stride);
          }
        }
        ctx.globalAlpha = 1;
        query.candidates.forEach((candidate, index) => {
          const [x, y] = candidate.reference_xy;
          ctx.strokeStyle = index === 0 ? "#ffffff" : "#0b1020";
          ctx.lineWidth = index === 0 ? 3 : 2;
          ctx.strokeRect(x * stride + 1, y * stride + 1, stride - 2, stride - 2);
          ctx.fillStyle = "#ffffff";
          ctx.font = "bold 11px system-ui";
          ctx.fillText(String(candidate.rank), x * stride + 3, y * stride + 12);
        });
        const [px, py] = query.predicted_reference_xy;
        ctx.strokeStyle = "#00ff9d";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc((px + 0.5) * stride, (py + 0.5) * stride, 7, 0, Math.PI * 2);
        ctx.stroke();
      } else {
        const [x0, y0, x1, y1] = query.token.nominal_pixel_box;
        ctx.fillStyle = "rgba(0,255,157,.22)";
        ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
        ctx.strokeStyle = "#00ff9d";
        ctx.lineWidth = 3;
        ctx.strokeRect(x0 + 1, y0 + 1, x1 - x0 - 2, y1 - y0 - 2);
      }
    };
    image.src = imageUrl;
  }, [imageUrl, query, kind, alpha]);

  function click(event) {
    if (kind !== "current" || !query || !onToken) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (event.clientX - rect.left) * canvasRef.current.width / rect.width;
    const y = (event.clientY - rect.top) * canvasRef.current.height / rect.height;
    const tx = Math.max(0, Math.min(query.grid.width - 1, Math.floor(x / query.grid.stride)));
    const ty = Math.max(0, Math.min(query.grid.height - 1, Math.floor(y / query.grid.stride)));
    onToken(ty * query.grid.width + tx);
  }

  return <canvas ref={canvasRef} onClick={click} className={kind === "current" ? "clickable" : ""} />;
}

function App() {
  const [runs, setRuns] = useState([]);
  const [runId, setRunId] = useState("");
  const [clips, setClips] = useState([]);
  const [clipId, setClipId] = useState("");
  const [frameId, setFrameId] = useState(1);
  const [tokenId, setTokenId] = useState(288);
  const [topK, setTopK] = useState(8);
  const [alpha, setAlpha] = useState(0.52);
  const [query, setQuery] = useState(null);
  const [error, setError] = useState("");

  const clip = useMemo(() => clips.find((item) => item.clip_id === clipId), [clips, clipId]);

  useEffect(() => {
    fetch("/api/runs").then((r) => r.json()).then((value) => {
      setRuns(value);
      if (value.length) setRunId(value[0].run_id);
    }).catch((reason) => setError(String(reason)));
  }, []);

  useEffect(() => {
    if (!runId) return;
    fetch(`/api/runs/${encodeURIComponent(runId)}/clips`).then((r) => r.json()).then((value) => {
      setClips(value);
      if (value.length) {
        setClipId(value[0].clip_id);
        setFrameId(Math.min(1, value[0].frame_count - 1));
        setTokenId(Math.floor(value[0].token_grid.count / 2));
      }
    }).catch((reason) => setError(String(reason)));
  }, [runId]);

  useEffect(() => {
    if (!runId || !clipId) return;
    const url = `/api/query?run_id=${encodeURIComponent(runId)}&clip_id=${encodeURIComponent(clipId)}&frame_id=${frameId}&token_id=${tokenId}&top_k=${topK}`;
    fetch(url).then(async (r) => {
      if (!r.ok) throw new Error(await r.text());
      return r.json();
    }).then((value) => { setQuery(value); setError(""); }).catch((reason) => setError(String(reason)));
  }, [runId, clipId, frameId, tokenId, topK]);

  const currentUrl = runId && clipId ? `/api/frame/${runId}/${clipId}/${frameId}` : "";
  const referenceUrl = runId && clipId ? `/api/frame/${runId}/${clipId}/0` : "";

  return <main>
    <header>
      <div><span className="eyebrow">Vote-and-Verify DiT</span><h1>Demo 0 · 对应关系检查台</h1></div>
      <span className="badge">PROVISIONAL · 无几何真值</span>
    </header>

    <section className="controls panel">
      <label>Run<select value={runId} onChange={(e) => setRunId(e.target.value)}>{runs.map((r) => <option key={r.run_id}>{r.run_id}</option>)}</select></label>
      <label>视频<select value={clipId} onChange={(e) => { setClipId(e.target.value); setFrameId(1); }}>{clips.map((c) => <option key={c.clip_id} value={c.clip_id}>{c.display_name}</option>)}</select></label>
      <label>帧 {frameId}<input type="range" min="0" max={Math.max(0, (clip?.frame_count || 1) - 1)} value={frameId} onChange={(e) => setFrameId(Number(e.target.value))} /></label>
      <label>Token ID<input type="number" min="0" max={Math.max(0, (clip?.token_grid?.count || 1) - 1)} value={tokenId} onChange={(e) => setTokenId(Number(e.target.value))} /></label>
      <label>Top‑K<select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>{[1, 4, 8].map((k) => <option key={k}>{k}</option>)}</select></label>
      <label>热力图透明度 {alpha.toFixed(2)}<input type="range" min="0" max="0.9" step="0.05" value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} /></label>
      <label>扩散步<select disabled><option>S=0（Demo0 无扩散）</option></select></label>
      <label>层 / 头<select disabled><option>ResNet18 layer3 · 单投影</option></select></label>
    </section>

    {error && <div className="error">{error}</div>}
    {query && <>
      <section className="visual-grid">
        <article className="panel"><h2>当前帧 · 点击选择 token</h2><CanvasView imageUrl={currentUrl} query={query} kind="current" alpha={alpha} onToken={setTokenId} /><p>绿色框是 token 的<strong>名义像素方框</strong>；真实 ResNet 感受野更大，不能把框外像素视为不可见。</p></article>
        <article className="panel"><h2>首帧 · 全候选相似度热力图</h2><CanvasView imageUrl={referenceUrl} query={query} kind="reference" alpha={alpha} /><p>数字框为原始 Top‑L；绿色圆圈是验证后的加权对应位置。</p></article>
      </section>

      <section className="summary-grid">
        <article className="panel stat"><span>选中槽</span><strong>{query.selected_motion_slot < 0 ? "拒绝" : `slot ${query.selected_motion_slot}`}</strong></article>
        <article className="panel stat"><span>拒绝概率</span><strong>{query.reject_probability.toFixed(3)}</strong></article>
        <article className="panel stat"><span>置信度</span><strong>{query.confidence.toFixed(3)}</strong></article>
        <article className="panel stat"><span>预测首帧坐标</span><strong>({query.predicted_reference_xy.map((v) => v.toFixed(2)).join(", ")})</strong></article>
      </section>

      <section className="panel"><h2>运动假设</h2><div className="slots">{query.motion_hypotheses.map((slot) => <div className="slot" key={slot.slot_id}><i style={{background: `rgb(${palette[slot.slot_id].join(",")})`}}></i><div><b>slot {slot.slot_id}</b><span>({slot.dx.toFixed(2)}, {slot.dy.toFixed(2)}) · p={slot.token_soft_probability.toFixed(3)}</span></div></div>)}</div></section>

      <section className="panel table-wrap"><h2>候选：最像 ≠ 验证后允许读取</h2><table><thead><tr><th>Rank</th><th>首帧 token / 坐标</th><th>位移 (dx,dy)</th><th>原始相似度</th><th>原始 attention</th><th>验证后 value 权重</th><th>槽 / 拒绝</th></tr></thead><tbody>{query.candidates.map((c) => <tr key={c.rank}><td>{c.rank}</td><td>{c.reference_token_id} / ({c.reference_xy.join(",")})</td><td>({c.displacement_query_minus_reference.join(",")})</td><td>{c.similarity.toFixed(4)}</td><td>{c.raw_dense_attention_weight.toExponential(2)}</td><td className="verified">{c.verified_value_route_weight.toFixed(4)}</td><td>s{c.best_motion_slot} / {c.candidate_reject_probability.toFixed(3)}</td></tr>)}</tbody></table></section>
    </>}
  </main>;
}

createRoot(document.getElementById("root")).render(<App />);

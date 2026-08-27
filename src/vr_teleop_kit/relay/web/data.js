/* Data & Training page.
 *
 * Three views over one API (see vr_teleop_kit/data/api.py):
 *   datasets  what has been recorded
 *   review    watch each episode, mark the failures
 *   train     launch a policy on the kept episodes and watch it learn
 *
 * No framework and no external assets: this is served by the teleop
 * relay, which has no network egress guarantees and is often reached
 * over a USB tether.
 */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };

  const JOINT_COLORS = ["var(--j0)", "var(--j1)", "var(--j2)", "var(--j3)", "var(--j4)"];
  const GRIP_CLOSED = 40, GRIP_OPEN = 70;   // must match data/grade.py
  const POLL_MS = 2000;
  /* Playback rates. Most of a demonstration is the approach; 2-4x makes
     a 25-episode review bearable, and 0.5x is for deciding whether a
     grasp actually closed on the object or just near it. */
  const SPEEDS = [0.5, 1, 1.5, 2, 4];

  const S = {
    env: null,
    datasets: [],
    ds: null,          // dataset detail
    ep: 0,             // selected episode index (array position)
    trace: null,
    videoKey: null,
    videoSrcKey: null,
    runs: [],
    run: null,         // selected run record
    metrics: [],
    metricsOffset: 0,
    logOffset: 0,
    logText: "",
    timer: null,
    rate: 1,
  };

  // ── helpers ────────────────────────────────────────────────────────────
  async function api(path, opts) {
    const res = await fetch(path, opts);
    let body = null;
    try { body = await res.json(); } catch (_) { /* empty body */ }
    if (!res.ok) {
      const msg = (body && body.detail) || `${res.status} ${res.statusText}`;
      const err = new Error(msg);
      err.status = res.status;
      throw err;
    }
    return body;
  }
  const post = (path, payload) =>
    api(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });

  function toast(msg, kind) {
    const n = el("div", "toast" + (kind ? " " + kind : ""), msg);
    $("toast").appendChild(n);
    setTimeout(() => n.remove(), kind === "err" ? 9000 : 4000);
  }

  const fmtBytes = (b) =>
    b > 1e9 ? (b / 1e9).toFixed(1) + " GB" : (b / 1e6).toFixed(0) + " MB";
  const fmtDur = (s) => {
    s = Math.round(s || 0);
    const m = Math.floor(s / 60);
    return m ? `${m}m ${String(s % 60).padStart(2, "0")}s` : `${s}s`;
  };
  const fmtNum = (v, d = 3) =>
    v === null || v === undefined || Number.isNaN(v) ? "—"
      : Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-3) ? v.toExponential(1) : v.toFixed(d);

  function showView(name) {
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    document.querySelectorAll(".tabs button").forEach((b) =>
      b.classList.toggle("active", b.dataset.view === name));
    $("view-" + name).classList.add("active");
    if (name === "train") { loadRuns(); refreshTrainForm(); }
    startPolling(name === "train");
  }
  document.querySelectorAll(".tabs button").forEach((b) =>
    b.addEventListener("click", () => { if (!b.disabled) showView(b.dataset.view); }));

  // ── environment chips ──────────────────────────────────────────────────
  async function loadEnv() {
    try { S.env = await api("/api/env"); } catch (e) { toast("env: " + e.message, "err"); return; }
    const box = $("env-chips");
    box.textContent = "";
    const chip = (label, value, warn) => {
      const c = el("span", "chip" + (warn ? " warn" : ""));
      c.appendChild(document.createTextNode(label + " "));
      c.appendChild(el("b", null, value));
      box.appendChild(c);
    };
    chip("GPU", S.env.gpus.length ? S.env.gpus[0].split(",")[0] : "none", !S.env.gpus.length);
    // Anything holding the serial bus blocks a rollout, so it is worth
    // seeing before you get as far as the launch dialog.
    const holders = S.env.serial.holders || [];
    if (!S.env.serial.exists) chip("arm", "no " + S.env.serial.port, true);
    else if (holders.length) chip("arm", "busy (pid " + holders[0].pid + ")", true);
    else chip("arm", S.env.serial.port);
    if (S.env.relay_cameras.length)
      chip("relay cams", S.env.relay_cameras.map((c) => c.id).join(", "), true);
    if (!S.env.is_local && !S.env.allow_remote_control) chip("remote", "read-only", true);
  }

  // ── datasets ───────────────────────────────────────────────────────────
  async function loadDatasets() {
    const grid = $("ds-grid");
    grid.textContent = "";
    let data;
    try { data = await api("/api/datasets"); }
    catch (e) { grid.appendChild(el("p", "empty", e.message)); return; }
    S.datasets = data.datasets;
    $("ds-sub").textContent = `${S.datasets.length} in ${S.env ? S.env.hf_home : ""}`;
    if (!S.datasets.length) {
      grid.appendChild(el("p", "empty",
        "Nothing recorded yet. Run examples/record_so101.py to collect demonstrations."));
      return;
    }
    for (const d of S.datasets) {
      const card = el("div", "card" + (d.is_backup ? " backup" : ""));
      const head = el("div", "card-head");
      head.appendChild(el("div", "card-title", d.repo_id));
      const acts = el("div", "card-actions");
      const act = (label, cls, fn) => {
        const b = el("button", cls, label);
        b.addEventListener("click", (ev) => { ev.stopPropagation(); fn(); });
        acts.appendChild(b);
      };
      act("rename", "", () => openRenameModal(d));
      act("delete", "danger", () => openDeleteDatasetModal(d));
      head.appendChild(acts);
      card.appendChild(head);
      card.appendChild(el("div", "card-task",
        d.is_backup ? "backup left by an in-place prune"
                    : (d.tasks[0] || "no task recorded")));
      const stats = el("div", "card-stats");
      const stat = (label, value) => {
        const s = el("span");
        s.appendChild(el("b", null, value));
        s.appendChild(document.createTextNode(" " + label));
        stats.appendChild(s);
      };
      stat("episodes", d.total_episodes);
      stat("", fmtDur(d.seconds));
      stat("", fmtBytes(d.size_bytes));
      if (d.review.reject) stat("rejected", d.review.reject);
      card.appendChild(stats);
      const bar = el("div", "bar");
      const seg = (cls, n) => {
        if (!n) return;
        const i = el("i", cls);
        i.style.flex = String(n);
        bar.appendChild(i);
      };
      seg("keep", d.review.keep); seg("unset", d.review.unset); seg("reject", d.review.reject);
      card.appendChild(bar);
      card.addEventListener("click", () => openDataset(d.repo_id));
      grid.appendChild(card);
    }
  }
  $("ds-refresh").addEventListener("click", () => { loadEnv(); loadDatasets(); });

  // ── review ─────────────────────────────────────────────────────────────
  async function openDataset(repoId, keepEpisode) {
    let detail;
    try { detail = await api("/api/dataset?repo_id=" + encodeURIComponent(repoId)); }
    catch (e) {
      if (e.status === 409) { toast(e.message, "err"); return; }
      toast(e.message, "err");
      return;
    }
    S.ds = detail;
    S.videoKey = detail.video_keys[0] || null;
    S.videoSrcKey = null;
    document.querySelector('.tabs button[data-view="review"]').disabled = false;
    showView("review");
    renderReview();
    selectEpisode(keepEpisode !== undefined ? keepEpisode : 0);
  }

  /* Which episodes will be held out for eval loss.
   *
   * Answered by the server rather than recomputed here: the rule has to
   * match what the trainer actually does, and two implementations of it
   * would eventually disagree — with the `val` badges quietly lying
   * about which demonstrations the policy never saw.
   *
   * Cached per (dataset, split, seed, mode) so re-rendering the episode
   * list does not refetch. */
  let SPLIT = { key: null, evalSet: new Set(), train: [], eval: [] };

  function splitParams() {
    return {
      split: parseFloat($("tr-eval-split").value) || 0,
      seed: Number($("tr-eval-seed").value) || 0,
      mode: $("tr-eval-mode").value,
    };
  }

  async function refreshSplit(repoId) {
    const { split, seed, mode } = splitParams();
    const key = `${repoId}|${split}|${seed}|${mode}`;
    if (SPLIT.key === key) return SPLIT;
    if (!repoId || split <= 0) {
      SPLIT = { key, evalSet: new Set(), train: [], eval: [] };
      return SPLIT;
    }
    try {
      const plan = await api(`/api/dataset/split?repo_id=${encodeURIComponent(repoId)}` +
        `&eval_split=${split}&seed=${seed}&mode=${encodeURIComponent(mode)}`);
      SPLIT = { key, evalSet: new Set(plan.eval), train: plan.train, eval: plan.eval };
    } catch (e) {
      SPLIT = { key, evalSet: new Set(), train: [], eval: [] };
    }
    return SPLIT;
  }

  function renderReview() {
    const d = S.ds;
    $("rv-title").textContent = d.repo_id;
    const kept = d.keep_list.length;
    $("rv-sub").textContent =
      `${d.total_episodes} episodes · ${fmtDur(d.seconds)} · ${kept} kept · ${d.tasks[0] || "no task"}`;

    const banner = $("rv-banner");
    banner.className = "banner";
    const stale = d.stale_episodes || [];
    if (stale.length) {
      // Name the affected episodes: pruning renumbers survivors, so only
      // the marks whose episode changed underneath them are suspect —
      // condemning the whole dataset would be both wrong and ignorable.
      banner.classList.add("show", "err");
      banner.textContent =
        `${stale.length} mark(s) no longer match the episode they were made about ` +
        `(${stale.map((i) => "idx " + i).join(", ")}). That happens when a dataset is ` +
        "pruned and the survivors are renumbered. Re-check those episodes, or reset the marks.";
    }

    const list = $("rv-list");
    list.textContent = "";
    const evalSet = SPLIT.evalSet;
    // Fetch in the background and re-render once, if the answer changed.
    refreshSplit(d.repo_id).then((plan) => {
      if (plan.evalSet !== evalSet && S.ds && S.ds.repo_id === d.repo_id) renderReview();
    });

    d.episodes.forEach((e, i) => {
      const li = el("li", "ep" + (e.status === "reject" ? " rejected" : ""));
      li.dataset.i = String(i);
      if (stale.includes(e.episode_index)) li.classList.add("stale");

      const row = el("div", "ep-row");
      const title = el("span", "ep-title");
      title.appendChild(document.createTextNode("Ep " + e.label + " "));
      // The true index rides along: episodes are talked about 1-based and
      // stored 0-based, and that gap is how the wrong one gets deleted.
      title.appendChild(el("span", "ep-idx", "idx " + e.episode_index));
      row.appendChild(title);
      row.appendChild(el("span", "tag " + e.verdict, e.verdict));
      if (evalSet.has(e.episode_index) && e.status !== "reject")
        row.appendChild(el("span", "tag val", "val"));
      row.appendChild(el("span", "ep-meta",
        `${e.seconds.toFixed(1)}s · ${(e.share * 100).toFixed(0)}%`));
      li.appendChild(row);
      li.appendChild(el("div", "ep-why", e.why));

      const actions = el("div", "ep-actions");
      const mk = (label, status) => {
        const b = el("button", "mark" + (e.status === status ? " on-" + status : ""), label);
        b.addEventListener("click", (ev) => {
          ev.stopPropagation();
          mark(e.episode_index, e.status === status ? "unset" : status);
        });
        actions.appendChild(b);
      };
      mk("keep", "keep");
      mk("reject", "reject");
      const note = el("input", "note-input");
      note.type = "text";
      note.placeholder = "note";
      note.value = e.note || "";
      note.addEventListener("click", (ev) => ev.stopPropagation());
      note.addEventListener("change", () => mark(e.episode_index, e.status, note.value));
      actions.appendChild(note);
      li.appendChild(actions);

      li.addEventListener("click", () => selectEpisode(i));
      list.appendChild(li);
    });
    highlightEpisode();

    const legend = $("rv-joint-legend");
    legend.textContent = "";
    d.joints.forEach((name, j) => {
      const s = el("span");
      const i = el("i");
      i.style.background = JOINT_COLORS[j % JOINT_COLORS.length];
      s.appendChild(i);
      s.appendChild(document.createTextNode(name.replace(/_/g, " ")));
      legend.appendChild(s);
    });
  }

  function highlightEpisode() {
    document.querySelectorAll("#rv-list .ep").forEach((n) =>
      n.classList.toggle("sel", Number(n.dataset.i) === S.ep));
  }

  async function mark(episodeIndex, status, note) {
    try {
      S.ds = await post("/api/dataset/review", {
        repo_id: S.ds.repo_id, episode: episodeIndex, status, note,
      });
      renderReview();
    } catch (e) { toast(e.message, "err"); }
  }

  const video = $("rv-video");

  function currentEpisode() { return S.ds ? S.ds.episodes[S.ep] : null; }
  function currentSlice() {
    const e = currentEpisode();
    if (!e || !S.videoKey) return null;
    return e.videos[S.videoKey] || null;
  }

  async function selectEpisode(i) {
    if (!S.ds || i < 0 || i >= S.ds.episodes.length) return;
    S.ep = i;
    highlightEpisode();
    const e = currentEpisode();
    const slice = currentSlice();

    if (slice) {
      // v3.0 packs many episodes per mp4, so switching episode is usually
      // a seek, not a load. Only a different chunk/file needs a new src.
      const srcKey = `${S.videoKey}/${slice.chunk_index}/${slice.file_index}`;
      if (srcKey !== S.videoSrcKey) {
        S.videoSrcKey = srcKey;
        video.src = `/api/dataset/video?repo_id=${encodeURIComponent(S.ds.repo_id)}` +
          `&key=${encodeURIComponent(S.videoKey)}&chunk=${slice.chunk_index}&file=${slice.file_index}`;
        video.load();
        video.addEventListener("loadedmetadata", () => seekToStart(), { once: true });
      } else {
        seekToStart();
      }
    }

    try {
      S.trace = await api(`/api/dataset/trace?repo_id=${encodeURIComponent(S.ds.repo_id)}` +
        `&episode=${e.episode_index}`);
    } catch (err) { S.trace = null; toast(err.message, "err"); }
    drawCharts();
    updateScrub();
  }

  function seekToStart() {
    const slice = currentSlice();
    if (slice) { try { video.currentTime = slice.from_timestamp; } catch (_) {} }
    updateScrub();
  }

  video.addEventListener("timeupdate", () => {
    const slice = currentSlice();
    // The mp4 runs on past this episode into the next one; stop at the
    // boundary so "watch episode 3" means episode 3.
    if (slice && video.currentTime >= slice.to_timestamp - 0.01) {
      video.pause();
      try { video.currentTime = slice.to_timestamp - 0.02; } catch (_) {}
    }
    updateScrub();
    drawPlayhead();
  });
  video.addEventListener("play", () => { $("rv-play").textContent = "❚❚"; });
  video.addEventListener("pause", () => { $("rv-play").textContent = "▶"; });

  function episodeTime() {
    const slice = currentSlice();
    if (!slice) return { t: 0, dur: 0 };
    return {
      t: Math.max(0, Math.min(video.currentTime - slice.from_timestamp,
                              slice.to_timestamp - slice.from_timestamp)),
      dur: slice.to_timestamp - slice.from_timestamp,
    };
  }

  function updateScrub() {
    const { t, dur } = episodeTime();
    $("rv-scrub-fill").style.width = dur ? (100 * t / dur).toFixed(2) + "%" : "0";
    $("rv-time").textContent = `${t.toFixed(1)} / ${dur.toFixed(1)} s`;
  }

  $("rv-play").addEventListener("click", togglePlay);
  function togglePlay() {
    const slice = currentSlice();
    if (!slice) return;
    if (video.paused) {
      if (video.currentTime < slice.from_timestamp ||
          video.currentTime >= slice.to_timestamp - 0.05) seekToStart();
      video.play().catch((e) => toast("playback: " + e.message, "err"));
    } else video.pause();
  }

  /* Playback speed. `playbackRate` is a property of the media element and
     resets to 1 every time the source is swapped, so it is re-applied on
     each load rather than set once. */
  function applyRate() {
    video.playbackRate = S.rate;
    const btn = $("rv-rate");
    btn.textContent = (S.rate % 1 === 0 ? S.rate : S.rate.toFixed(1)) + "\u00d7";
    btn.classList.toggle("fast", S.rate !== 1);
  }

  function setRate(rate) {
    S.rate = rate;
    applyRate();
    try { localStorage.setItem("vrteleop.rate", String(rate)); } catch (_) { /* private mode */ }
  }

  function stepRate(delta) {
    const i = SPEEDS.indexOf(S.rate);
    const next = SPEEDS[Math.min(SPEEDS.length - 1, Math.max(0, (i < 0 ? 1 : i) + delta))];
    setRate(next);
  }

  $("rv-rate").addEventListener("click", () => {
    const i = SPEEDS.indexOf(S.rate);
    setRate(SPEEDS[(i + 1) % SPEEDS.length]);
  });
  video.addEventListener("loadedmetadata", applyRate);

  $("rv-scrub").addEventListener("click", (ev) => {
    const slice = currentSlice();
    if (!slice) return;
    const r = ev.currentTarget.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width));
    video.currentTime = slice.from_timestamp + frac * (slice.to_timestamp - slice.from_timestamp);
  });

  // ── charts ─────────────────────────────────────────────────────────────
  const SVG_NS = "http://www.w3.org/2000/svg";
  function svgEl(tag, attrs) {
    const n = document.createElementNS(SVG_NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  }

  function linePath(xs, ys, x, y) {
    let d = "";
    for (let i = 0; i < xs.length; i++)
      d += (i ? "L" : "M") + x(xs[i]).toFixed(1) + " " + y(ys[i]).toFixed(1);
    return d;
  }

  function drawCharts() {
    drawGripChart();
    drawJointChart();
    drawPlayhead();
  }

  const PAD = { l: 34, r: 6, t: 8, b: 14 };

  function scales(svg, tMax, lo, hi) {
    const vb = svg.getAttribute("viewBox").split(" ").map(Number);
    const [w, h] = [vb[2], vb[3]];
    const span = (hi - lo) || 1;
    return {
      w, h,
      x: (t) => PAD.l + (tMax ? (t / tMax) : 0) * (w - PAD.l - PAD.r),
      y: (v) => PAD.t + (1 - (v - lo) / span) * (h - PAD.t - PAD.b),
    };
  }

  function axis(svg, sc, ticks, fmt) {
    for (const v of ticks) {
      const yy = sc.y(v);
      svg.appendChild(svgEl("line", {
        x1: PAD.l, x2: sc.w - PAD.r, y1: yy, y2: yy,
        stroke: "var(--border)", "stroke-width": 1, "vector-effect": "non-scaling-stroke",
      }));
      const label = svgEl("text", {
        x: PAD.l - 5, y: yy + 3, "text-anchor": "end",
        fill: "#5c6270", "font-size": 9, "font-family": "ui-monospace, monospace",
      });
      label.textContent = fmt ? fmt(v) : String(v);
      svg.appendChild(label);
    }
  }

  function drawGripChart() {
    const svg = $("rv-chart-grip");
    svg.textContent = "";
    if (!S.trace) return;
    const gi = S.trace.state.length - 1;         // gripper is the last channel
    const ys = S.trace.state[gi];
    const ts = S.trace.t;
    const tMax = ts.length ? ts[ts.length - 1] : 1;
    const sc = scales(svg, tMax, 0, 100);
    axis(svg, sc, [0, 50, 100]);

    // Threshold guides: the grade heuristics call a grasp only when the
    // trace crosses below one and back above the other, so showing them
    // makes the verdict checkable by eye.
    for (const v of [GRIP_CLOSED, GRIP_OPEN]) {
      svg.appendChild(svgEl("line", {
        x1: PAD.l, x2: sc.w - PAD.r, y1: sc.y(v), y2: sc.y(v),
        stroke: "#5c6270", "stroke-width": 1, "stroke-dasharray": "3 3",
        "vector-effect": "non-scaling-stroke",
      }));
    }
    svg.appendChild(svgEl("path", {
      d: linePath(ts, ys, sc.x, sc.y), fill: "none", stroke: "var(--grip)",
      "stroke-width": 1.6, "vector-effect": "non-scaling-stroke",
    }));
    svg.appendChild(svgEl("line", {
      id: "grip-playhead", x1: PAD.l, x2: PAD.l, y1: PAD.t, y2: sc.h - PAD.b,
      stroke: "var(--accent)", "stroke-width": 1, "vector-effect": "non-scaling-stroke",
    }));
  }

  function drawJointChart() {
    const svg = $("rv-chart-joints");
    svg.textContent = "";
    if (!S.trace) return;
    const ts = S.trace.t;
    const tMax = ts.length ? ts[ts.length - 1] : 1;
    let lo = Infinity, hi = -Infinity;
    for (let j = 0; j < 5; j++)
      for (const v of S.trace.state[j]) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (!isFinite(lo)) { lo = -1; hi = 1; }
    const padY = (hi - lo) * 0.08 || 1;
    const sc = scales(svg, tMax, lo - padY, hi + padY);
    const mid = Math.round((lo + hi) / 2);
    axis(svg, sc, [Math.round(lo), mid, Math.round(hi)], (v) => v + "°");

    for (let j = 0; j < 5; j++) {
      svg.appendChild(svgEl("path", {
        d: linePath(ts, S.trace.state[j], sc.x, sc.y), fill: "none",
        stroke: JOINT_COLORS[j], "stroke-width": 1.3, opacity: 0.95,
        "vector-effect": "non-scaling-stroke",
      }));
    }
    svg.appendChild(svgEl("line", {
      id: "joint-playhead", x1: PAD.l, x2: PAD.l, y1: PAD.t, y2: sc.h - PAD.b,
      stroke: "var(--accent)", "stroke-width": 1, "vector-effect": "non-scaling-stroke",
    }));
  }

  function drawPlayhead() {
    if (!S.trace) return;
    const { t, dur } = episodeTime();
    const frac = dur ? t / dur : 0;
    for (const [id, svgId] of [["grip-playhead", "rv-chart-grip"], ["joint-playhead", "rv-chart-joints"]]) {
      const line = document.getElementById(id);
      if (!line) continue;
      const vb = $(svgId).getAttribute("viewBox").split(" ").map(Number);
      const x = PAD.l + frac * (vb[2] - PAD.l - PAD.r);
      line.setAttribute("x1", x); line.setAttribute("x2", x);
    }
  }

  // ── review actions ─────────────────────────────────────────────────────
  $("rv-auto").addEventListener("click", async () => {
    try {
      S.ds = await post("/api/dataset/review/bulk", { repo_id: S.ds.repo_id, action: "auto" });
      renderReview();
      toast("FAIL episodes rejected. SUSPECT ones are left for you to watch.", "ok");
    } catch (e) { toast(e.message, "err"); }
  });

  $("rv-clear").addEventListener("click", async () => {
    try {
      S.ds = await post("/api/dataset/review/bulk", { repo_id: S.ds.repo_id, action: "clear" });
      renderReview();
    } catch (e) { toast(e.message, "err"); }
  });

  $("rv-train").addEventListener("click", () => {
    showView("train");
    $("tr-dataset").value = S.ds.repo_id;
    refreshTrainForm();
  });

  $("rv-export").addEventListener("click", () => {
    const d = S.ds;
    const drop = d.episodes.filter((e) => e.status === "reject");
    const suggested = d.repo_id + "_clean";
    const card = $("modal-card");
    card.textContent = "";
    card.appendChild(el("h3", null, "Export pruned copy"));
    if (!drop.length) {
      card.appendChild(el("p", "note",
        "No episodes are rejected, so the copy would be identical to the original. " +
        "Mark the failures first."));
      const row = el("div", "btn-row");
      const close = el("button", "btn", "Close");
      close.addEventListener("click", closeModal);
      row.appendChild(close);
      card.appendChild(row);
      $("modal").classList.add("show");
      return;
    }
    card.appendChild(el("p", "note",
      `Writes a new dataset with ${d.keep_list.length} kept episodes. ` +
      `${drop.length} rejected episode(s) are left out: ` +
      drop.map((e) => "Ep " + e.label).join(", ") + "."));
    const warn = el("div", "warn-box");
    warn.appendChild(document.createTextNode(
      "The original is not modified. In the copy, episodes are renumbered from 0 and the " +
      "video is re-encoded, which takes a few minutes — it runs as a background job. " +
      "The copy starts with no review marks, because the old ones would point at the " +
      "wrong episodes."));
    card.appendChild(warn);
    const field = el("div", "field");
    field.appendChild(el("label", null, "New repo-id"));
    const input = el("input");
    input.type = "text";
    input.value = suggested;
    field.appendChild(input);
    card.appendChild(field);
    const row = el("div", "btn-row");
    const cancel = el("button", "btn", "Cancel");
    cancel.addEventListener("click", closeModal);
    const go = el("button", "btn btn-accent", "Export");
    go.addEventListener("click", async () => {
      go.disabled = true;
      try {
        await post("/api/dataset/export", { repo_id: d.repo_id, new_repo_id: input.value.trim() });
        closeModal();
        toast("Export started — watch it under Train › Runs.", "ok");
        showView("train");
        loadRuns();
      } catch (e) { toast(e.message, "err"); go.disabled = false; }
    });
    row.appendChild(cancel); row.appendChild(go);
    card.appendChild(row);
    $("modal").classList.add("show");
  });

  /* Permanent prune. The export above is the safe route; this one
     rewrites the dataset itself, so the dialog has to make the
     consequences impossible to miss rather than merely mentioning them. */
  $("rv-delete").addEventListener("click", () => {
    const d = S.ds;
    const drop = d.episodes.filter((e) => e.status === "reject");
    const card = $("modal-card");
    card.textContent = "";
    card.appendChild(el("h3", null, "Delete rejected episodes"));

    if (!drop.length || drop.length === d.episodes.length) {
      card.appendChild(el("p", "note", !drop.length
        ? "No episodes are marked reject, so there is nothing to delete. Mark the failures first."
        : "Every episode is marked reject. A dataset cannot be emptied this way — "
          + "delete the whole dataset directory if that is what you want."));
      const row = el("div", "btn-row");
      const close = el("button", "btn", "Close");
      close.addEventListener("click", closeModal);
      row.appendChild(close);
      card.appendChild(row);
      $("modal").classList.add("show");
      return;
    }

    const warn = el("div", "warn-box err");
    warn.appendChild(document.createTextNode(
      `Permanently removes ${drop.length} of ${d.episodes.length} episode(s) from `
      + `${d.repo_id}. This rewrites the dataset:`));
    const ul = el("ul");
    drop.forEach((e) => ul.appendChild(el("li", null,
      `Ep ${e.label} (idx ${e.episode_index}) — ${e.seconds.toFixed(1)}s — ${e.why}`)));
    warn.appendChild(ul);
    card.appendChild(warn);
    card.appendChild(el("p", "note",
      `The ${d.keep_list.length} kept episodes are renumbered 0..${d.keep_list.length - 1} `
      + "and the video is re-encoded, so this runs as a background job. Review marks are "
      + "cleared afterwards, because they would otherwise point at renumbered episodes."));

    const backupWrap = el("label", "toggle");
    backupWrap.style.display = "flex";
    backupWrap.style.gap = ".5rem";
    backupWrap.style.alignItems = "center";
    backupWrap.style.fontSize = ".86rem";
    const backup = el("input");
    backup.type = "checkbox";
    backup.checked = true;
    backupWrap.appendChild(backup);
    backupWrap.appendChild(document.createTextNode(
      `Keep the previous version as ${d.repo_id}_old`));
    card.appendChild(backupWrap);

    // Without a backup there is nothing to undo with, so that is the one
    // path that asks you to type the word.
    const confirmField = el("div", "field");
    confirmField.style.display = "none";
    confirmField.appendChild(el("label", null, 'No backup will be kept. Type DELETE to confirm.'));
    const confirmInput = el("input");
    confirmInput.type = "text";
    confirmInput.placeholder = "DELETE";
    confirmField.appendChild(confirmInput);
    card.appendChild(confirmField);

    const row = el("div", "btn-row");
    const cancel = el("button", "btn", "Cancel");
    cancel.addEventListener("click", closeModal);
    const go = el("button", "btn btn-danger", "Delete permanently");
    const sync = () => {
      const noBackup = !backup.checked;
      confirmField.style.display = noBackup ? "flex" : "none";
      go.disabled = noBackup && confirmInput.value.trim() !== "DELETE";
      go.textContent = noBackup ? "Delete permanently" : "Delete (keep backup)";
    };
    backup.addEventListener("change", sync);
    confirmInput.addEventListener("input", sync);
    go.addEventListener("click", async () => {
      go.disabled = true;
      try {
        await post("/api/dataset/delete_episodes", {
          repo_id: d.repo_id,
          keep_backup: backup.checked,
          // Guard against the dataset changing between the dialog opening
          // and this click: the server refuses if the set no longer matches.
          expect_episodes: drop.map((e) => e.episode_index),
        });
        closeModal();
        toast("Deleting — watch it under Train › Runs.", "ok");
        showView("train");
        loadRuns();
      } catch (e) { toast(e.message, "err"); go.disabled = false; }
    });
    row.appendChild(cancel); row.appendChild(go);
    card.appendChild(row);
    sync();
    $("modal").classList.add("show");
  });

  /* ── dataset lifecycle ────────────────────────────────────────────── */

  function modalShell(title) {
    const card = $("modal-card");
    card.textContent = "";
    card.appendChild(el("h3", null, title));
    $("modal").classList.add("show");
    return card;
  }

  function modalButtons(card, primaryLabel, primaryCls, onGo) {
    const row = el("div", "btn-row");
    const cancel = el("button", "btn", "Cancel");
    cancel.addEventListener("click", closeModal);
    const go = el("button", "btn " + primaryCls, primaryLabel);
    go.addEventListener("click", () => onGo(go));
    row.appendChild(cancel);
    row.appendChild(go);
    card.appendChild(row);
    return go;
  }

  function openRenameModal(d) {
    const card = modalShell("Rename dataset");
    card.appendChild(el("p", "note",
      `${d.repo_id} — ${d.total_episodes} episodes. The directory is moved; nothing `
      + "inside the dataset changes, and any _old backup moves with it."));
    const field = el("div", "field");
    field.appendChild(el("label", null, "New name"));
    const input = el("input");
    input.type = "text";
    input.value = d.repo_id;
    field.appendChild(input);
    field.appendChild(el("span", "hint", "letters, digits, . _ - and at most one /"));
    card.appendChild(field);
    modalButtons(card, "Rename", "btn-accent", async (go) => {
      go.disabled = true;
      try {
        const r = await post("/api/dataset/rename",
          { repo_id: d.repo_id, new_repo_id: input.value.trim() });
        closeModal();
        toast("Renamed to " + r.repo_id, "ok");
        // Any open review of the old name now points at a path that moved.
        if (S.ds && S.ds.repo_id === d.repo_id) { S.ds = null; showView("datasets"); }
        loadDatasets();
      } catch (e) { toast(e.message, "err"); go.disabled = false; }
    });
    input.focus();
    input.select();
  }

  function openDeleteDatasetModal(d) {
    const card = modalShell("Delete dataset");
    const warn = el("div", "warn-box err");
    warn.textContent =
      `Permanently deletes ${d.repo_id} — all ${d.total_episodes} episodes, `
      + `${fmtDur(d.seconds)} of demonstration, ${fmtBytes(d.size_bytes)} on disk. `
      + "There is no undo and nothing is moved to a backup.";
    card.appendChild(warn);
    const field = el("div", "field");
    field.appendChild(el("label", null, "Type the dataset name to confirm"));
    const input = el("input");
    input.type = "text";
    input.placeholder = d.repo_id;
    field.appendChild(input);
    card.appendChild(field);
    const go = modalButtons(card, "Delete permanently", "btn-danger", async (btn) => {
      btn.disabled = true;
      try {
        const r = await post("/api/dataset/delete",
          { repo_id: d.repo_id, confirm: input.value.trim() });
        closeModal();
        toast(`Deleted ${r.repo_id} — ${fmtBytes(r.freed_bytes)} freed.`, "ok");
        if (S.ds && S.ds.repo_id === d.repo_id) { S.ds = null; showView("datasets"); }
        loadDatasets();
      } catch (e) { toast(e.message, "err"); btn.disabled = false; }
    });
    const sync = () => { go.disabled = input.value.trim() !== d.repo_id; };
    input.addEventListener("input", sync);
    sync();
    input.focus();
  }

  /* ── recording ────────────────────────────────────────────────────── */

  let CAMERAS = null;

  async function loadCameras(refresh) {
    try { CAMERAS = await api("/api/cameras" + (refresh ? "?refresh=true" : "")); }
    catch (e) { CAMERAS = { cameras: [], error: e.message, relay_open: [] }; }
    return CAMERAS;
  }

  function cameraRow(name, source, onRemove) {
    const row = el("div", "cam-row");
    const nameInput = el("input", "cam-name");
    nameInput.type = "text";
    nameInput.value = name;
    nameInput.placeholder = "name";
    nameInput.title = "Becomes observation.images.<name>; a rollout must use the same names.";
    row.appendChild(nameInput);

    const sel = el("select");
    for (const cam of (CAMERAS && CAMERAS.cameras) || []) {
      const o = el("option", null, cam.name + (cam.warning ? "  ⚠" : ""));
      o.value = cam.source;
      if (cam.warning) o.title = cam.warning;
      sel.appendChild(o);
    }
    const custom = el("option", null, "custom path…");
    custom.value = "__custom__";
    sel.appendChild(custom);
    const customInput = el("input");
    customInput.type = "text";
    customInput.placeholder = "/dev/videoN";
    customInput.style.display = "none";
    customInput.style.flex = "1";
    if (source && ![...sel.options].some((o) => o.value === source)) {
      sel.value = "__custom__";
      customInput.value = source;
      customInput.style.display = "";
    } else if (source) {
      sel.value = source;
    }
    sel.addEventListener("change", () => {
      customInput.style.display = sel.value === "__custom__" ? "" : "none";
    });
    row.appendChild(sel);
    row.appendChild(customInput);

    const rm = el("button", "btn btn-sm", "×");
    rm.title = "Remove this camera";
    rm.addEventListener("click", () => { row.remove(); onRemove && onRemove(); });
    row.appendChild(rm);

    row.spec = () => {
      const n = nameInput.value.trim();
      const src = sel.value === "__custom__" ? customInput.value.trim() : sel.value;
      return n && src ? `${n}=${src}` : null;
    };
    return row;
  }

  async function openRecordModal() {
    await loadCameras();
    const card = modalShell("Record episodes");

    if (CAMERAS.error) {
      const w = el("div", "warn-box");
      w.textContent = "Camera discovery failed: " + CAMERAS.error;
      card.appendChild(w);
    }
    // A camera the relay has actually opened cannot also be recorded from;
    // the RealSense colour node in particular is single-holder.
    for (const cam of CAMERAS.relay_open || []) {
      const w = el("div", "warn-box err");
      w.textContent =
        `This relay is streaming camera "${cam.id}" (${cam.device}) to the headset. `
        + "Turn camera streaming off in Settings before recording from it — the "
        + "RealSense colour node cannot be shared.";
      card.appendChild(w);
    }

    const known = S.datasets.filter((d) => !d.is_backup);
    const dsField = el("div", "field");
    dsField.appendChild(el("label", null, "Dataset"));
    const dsSel = el("select");
    const newOpt = el("option", null, "＋ new dataset…");
    newOpt.value = "__new__";
    dsSel.appendChild(newOpt);
    known.forEach((d) => {
      const o = el("option", null, `${d.repo_id}  (resume — ${d.total_episodes} eps)`);
      o.value = d.repo_id;
      dsSel.appendChild(o);
    });
    dsField.appendChild(dsSel);
    const nameInput = el("input");
    nameInput.type = "text";
    nameInput.placeholder = "local/so101_pick_cube";
    nameInput.style.marginTop = ".35rem";
    dsField.appendChild(nameInput);
    const dsHint = el("span", "hint", "");
    dsField.appendChild(dsHint);
    card.appendChild(dsField);

    const taskField = el("div", "field");
    taskField.appendChild(el("label", null, "Task"));
    const task = el("input");
    task.type = "text";
    task.placeholder = "Pick up the battery and place it in the box";
    taskField.appendChild(task);
    taskField.appendChild(el("span", "hint",
      "Stored with every episode and used as the policy's language prompt — "
      + "keep it identical across a dataset."));
    card.appendChild(taskField);

    // Cameras
    const camWrap = el("div", "field");
    camWrap.appendChild(el("label", null, "Cameras"));
    const camList = el("div");
    camList.style.display = "flex";
    camList.style.flexDirection = "column";
    camList.style.gap = ".35rem";
    camWrap.appendChild(camList);
    const camBtns = el("div", "cam-row");
    const addCam = el("button", "btn btn-sm", "+ camera");
    addCam.addEventListener("click", () =>
      camList.appendChild(cameraRow("wrist", (CAMERAS.cameras[0] || {}).source || "")));
    const rescan = el("button", "btn btn-sm", "rescan");
    rescan.addEventListener("click", async () => {
      await loadCameras(true);
      closeModal();
      openRecordModal();
    });
    camBtns.appendChild(addCam);
    camBtns.appendChild(rescan);
    camWrap.appendChild(camBtns);
    camWrap.appendChild(el("span", "hint",
      "Each becomes observation.images.<name>. A rollout must use the same names, "
      + "sizes and sources as the recording."));
    card.appendChild(camWrap);
    camList.appendChild(cameraRow("top", (CAMERAS.cameras[0] || {}).source || ""));

    // Settings grid
    const form = el("div", "form");
    const num = (label, value, hint, attrs) => {
      const f = el("div", "field");
      f.appendChild(el("label", null, label));
      const i = el("input");
      i.type = "number";
      i.value = value;
      Object.assign(i, attrs || {});
      f.appendChild(i);
      if (hint) f.appendChild(el("span", "hint", hint));
      form.appendChild(f);
      return i;
    };
    const nEps = num("Episodes", 30, "target for this session", { min: 1 });
    const fps = num("FPS", 30, "dataset + teleop rate", { min: 1, max: 120 });
    const epTime = num("Episode cap (s)", 60, "you normally end with B", { min: 1 });
    const camW = num("Cam width", 640, "", { min: 64 });
    const camH = num("Cam height", 480, "", { min: 64 });
    const camFps = num("Cam FPS", 30, "", { min: 1 });
    card.appendChild(form);

    const advanced = el("details");
    advanced.appendChild(el("summary", null, "Hardware & advanced"));
    const advForm = el("div", "form");
    advForm.style.marginTop = ".5rem";
    const text = (label, value, hint) => {
      const f = el("div", "field");
      f.appendChild(el("label", null, label));
      const i = el("input");
      i.type = "text";
      i.value = value;
      f.appendChild(i);
      if (hint) f.appendChild(el("span", "hint", hint));
      advForm.appendChild(f);
      return i;
    };
    const port = text("Serial port", (S.env && S.env.serial.port) || "/dev/ttyACM0");
    const robotId = text("Robot id", "so101", "must match lerobot-calibrate");
    const handField = el("div", "field");
    handField.appendChild(el("label", null, "Driving hand"));
    const hand = el("select");
    ["right", "left"].forEach((h) => {
      const o = el("option", null, h);
      o.value = h;
      hand.appendChild(o);
    });
    handField.appendChild(hand);
    advForm.appendChild(handField);
    advanced.appendChild(advForm);
    const checks = el("div");
    checks.style.marginTop = ".5rem";
    const check = (label, checked, title) => {
      const l = el("label");
      l.style.display = "flex";
      l.style.gap = ".5rem";
      l.style.alignItems = "center";
      l.style.fontSize = ".84rem";
      l.style.padding = ".15rem 0";
      const i = el("input");
      i.type = "checkbox";
      i.checked = checked;
      if (title) l.title = title;
      l.appendChild(i);
      l.appendChild(document.createTextNode(label));
      checks.appendChild(l);
      return i;
    };
    const startGate = check("Wait for B before each episode", true,
      "Off means episode 0 records the instant the arm finishes its ramp — while "
      + "you are still getting into position.");
    const sounds = check("Speak phase changes aloud", true,
      "How you follow READY / RECORD from inside the headset.");
    const displayData = check("Live rerun view", false);
    advanced.appendChild(checks);
    card.appendChild(advanced);

    const summary = el("p", "note", "");
    card.appendChild(summary);

    const syncDataset = () => {
      const isNew = dsSel.value === "__new__";
      nameInput.style.display = isNew ? "" : "none";
      const repoId = isNew ? nameInput.value.trim() : dsSel.value;
      const existing = known.find((d) => d.repo_id === repoId);
      if (existing) {
        dsHint.textContent =
          `Resuming — ${existing.total_episodes} episodes already recorded. New ones are appended.`;
        if (!task.value && existing.tasks[0]) task.value = existing.tasks[0];
      } else {
        dsHint.textContent = isNew
          ? "A new dataset is created under the lerobot cache."
          : "";
      }
      summary.textContent = repoId
        ? `${existing ? "Resume" : "Create"} ${repoId} · up to ${nEps.value} episodes · `
          + `${fps.value} fps · cap ${epTime.value}s per episode`
        : "";
    };
    dsSel.addEventListener("change", syncDataset);
    nameInput.addEventListener("input", syncDataset);
    [nEps, fps, epTime].forEach((i) => i.addEventListener("input", syncDataset));
    syncDataset();

    modalButtons(card, "Start recording", "btn-accent", async (go) => {
      const isNew = dsSel.value === "__new__";
      const repoId = isNew ? nameInput.value.trim() : dsSel.value;
      const cams = [...camList.children].map((r) => r.spec()).filter(Boolean);
      go.disabled = true;
      try {
        const run = await post("/api/run/record", {
          repo_id: repoId,
          task: task.value.trim(),
          resume: !isNew,
          num_episodes: Number(nEps.value),
          fps: Number(fps.value),
          episode_time_s: Number(epTime.value),
          cameras: cams,
          cam_width: Number(camW.value),
          cam_height: Number(camH.value),
          cam_fps: Number(camFps.value),
          port: port.value,
          robot_id: robotId.value,
          hand: hand.value,
          no_start_gate: !startGate.checked,
          no_sounds: !sounds.checked,
          display_data: displayData.checked,
        });
        closeModal();
        toast("Recording session started — put the headset on.", "ok");
        showView("train");
        await loadRuns();
        openRun(run.id);
      } catch (e) { toast(e.message, "err"); go.disabled = false; }
    });
  }

  $("ds-record").addEventListener("click", openRecordModal);

  /* Session control. The recorder listens for these on the relay, so a
     click here does exactly what pressing B or Y in the headset does —
     which matters because stopping the process instead would lose the
     episode in progress. */
  let controlWs = null;

  function sendRecordControl(action) {
    const url = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
    const send = () => controlWs.send(JSON.stringify({ type: "record_control", action }));
    if (controlWs && controlWs.readyState === WebSocket.OPEN) { send(); return; }
    controlWs = new WebSocket(url);
    controlWs.addEventListener("open", send);
    controlWs.addEventListener("error", () =>
      toast("could not reach the relay to send that control", "err"));
  }

  function closeModal() { $("modal").classList.remove("show"); }
  $("modal").addEventListener("click", (ev) => { if (ev.target === $("modal")) closeModal(); });

  // ── keyboard review shortcuts ──────────────────────────────────────────
  window.addEventListener("keydown", (ev) => {
    if (ev.target.matches("input, select, textarea")) return;
    if (ev.key === "Escape") { closeModal(); return; }
    if (!$("view-review").classList.contains("active") || !S.ds) return;
    const e = currentEpisode();
    if (ev.key === " ") { ev.preventDefault(); togglePlay(); }
    else if (ev.key === "j" || ev.key === "J") selectEpisode(S.ep - 1);
    else if (ev.key === "l" || ev.key === "L") selectEpisode(S.ep + 1);
    else if (ev.key === "k" || ev.key === "K") mark(e.episode_index, e.status === "keep" ? "unset" : "keep");
    else if (ev.key === "r" || ev.key === "R") mark(e.episode_index, e.status === "reject" ? "unset" : "reject");
    else if (ev.key === "[") stepRate(-1);
    else if (ev.key === "]") stepRate(1);
  });

  // ── train ──────────────────────────────────────────────────────────────
  function refreshTrainForm() {
    const sel = $("tr-dataset");
    const previous = sel.value;
    sel.textContent = "";
    for (const d of S.datasets) {
      if (d.is_backup) continue;
      const o = el("option", null, `${d.repo_id}  (${d.total_episodes} eps)`);
      o.value = d.repo_id;
      sel.appendChild(o);
    }
    if (previous) sel.value = previous;
    else if (S.ds) sel.value = S.ds.repo_id;
    updateTrainSummary();
  }

  async function updateTrainSummary() {
    const repoId = $("tr-dataset").value;
    const hint = $("tr-dataset-hint");
    const summary = $("tr-summary");
    if (!repoId) { summary.textContent = ""; return; }
    let d = S.ds && S.ds.repo_id === repoId ? S.ds : null;
    if (!d) {
      try { d = await api("/api/dataset?repo_id=" + encodeURIComponent(repoId)); }
      catch (e) { hint.textContent = e.message; summary.textContent = ""; return; }
    }
    const { split, mode } = splitParams();
    const plan = await refreshSplit(repoId);
    const nTrain = d.keep_list.length - plan.evalSet.size;
    hint.textContent =
      `${d.keep_list.length} of ${d.total_episodes} episodes kept · ${d.tasks[0] || "no task"}`;
    const evalLabels = d.episodes
      .filter((e) => plan.evalSet.has(e.episode_index)).map((e) => "Ep " + e.label);
    summary.textContent =
      `Trains on ${nTrain} episode(s)` +
      (plan.evalSet.size
        ? `; holds out ${plan.evalSet.size} for eval loss — ` +
          (mode === "random" ? "a random sample: " : "the most recent: ") +
          evalLabels.join(", ") + "."
        : "; no held-out episodes — eval loss will not be plotted.");
    if (!$("tr-job").value) $("tr-job").placeholder = ($("tr-policy").value + "_" +
      repoId.split("/").pop()).slice(0, 60);
  }
  $("tr-reshuffle").addEventListener("click", () => {
    // A different draw of the same size — useful when a small dataset
    // happens to put every hard example in the held-out set.
    $("tr-eval-seed").value = String(Math.floor(Math.random() * 10000));
    updateTrainSummary();
    if (S.ds && $("view-review").classList.contains("active")) renderReview();
  });
  ["tr-dataset", "tr-eval-split", "tr-policy", "tr-eval-mode", "tr-eval-seed"].forEach((id) =>
    $(id).addEventListener("change", () => {
      updateTrainSummary();
      if (S.ds && $("view-review").classList.contains("active")) renderReview();
    }));

  $("tr-launch").addEventListener("click", async () => {
    const btn = $("tr-launch");
    btn.disabled = true;
    try {
      const run = await post("/api/run/train", {
        repo_id: $("tr-dataset").value,
        policy_type: $("tr-policy").value,
        steps: Number($("tr-steps").value),
        batch_size: Number($("tr-batch").value),
        eval_split: Number($("tr-eval-split").value),
        eval_mode: $("tr-eval-mode").value,
        eval_seed: Number($("tr-eval-seed").value),
        eval_steps: Number($("tr-eval-steps").value),
        max_eval_samples: Number($("tr-max-eval").value),
        save_freq: Number($("tr-save-freq").value),
        num_workers: Number($("tr-workers").value),
        device: $("tr-device").value,
        job_name: $("tr-job").value || $("tr-job").placeholder,
      });
      toast("Training started: " + run.id, "ok");
      await loadRuns();
      openRun(run.id);
    } catch (e) { toast(e.message, "err"); }
    btn.disabled = false;
  });

  async function loadRuns() {
    let data;
    try { data = await api("/api/runs"); } catch (e) { return; }
    S.runs = data.runs;
    const box = $("run-list");
    box.textContent = "";
    if (!S.runs.length) {
      box.appendChild(el("p", "empty", "No runs yet."));
      return;
    }
    for (const r of S.runs) {
      const row = el("div", "run" + (S.run && S.run.id === r.id ? " sel" : ""));
      row.appendChild(el("span", "st " + r.status, r.status));
      row.appendChild(el("span", "run-name", r.name || r.id));
      row.appendChild(el("span", "run-meta", r.kind + " · " + r.id.slice(r.kind.length + 1, r.kind.length + 16)));
      row.addEventListener("click", () => openRun(r.id));
      box.appendChild(row);
    }
  }

  async function openRun(id) {
    S.metrics = []; S.metricsOffset = 0; S.logOffset = 0; S.logText = "";
    try { S.run = await api("/api/run?id=" + encodeURIComponent(id)); }
    catch (e) { toast(e.message, "err"); return; }
    renderRunDetail();
    await pollRun();
    loadRuns();
  }

  function renderRunDetail() {
    const box = $("run-detail");
    box.textContent = "";
    const r = S.run;
    if (!r) { box.appendChild(el("p", "empty", "Select a run.")); return; }

    const head = el("div", "head");
    head.style.padding = "0 0 .6rem";
    head.style.border = "none";
    head.appendChild(el("h2", null, r.name || r.id));
    head.appendChild(el("span", "st " + r.status, r.status));
    head.appendChild(el("span", "spacer"));
    const actions = el("div", "head-actions");
    if (r.status === "running") {
      const stop = el("button", "btn btn-sm btn-danger", "Stop");
      stop.addEventListener("click", async () => {
        stop.disabled = true;
        try { S.run = await post("/api/run/stop", { id: r.id }); renderRunDetail(); loadRuns(); }
        catch (e) { toast(e.message, "err"); stop.disabled = false; }
      });
      actions.appendChild(stop);
    }
    if (r.kind === "train") {
      const roll = el("button", "btn btn-sm btn-accent", "Run on robot…");
      roll.addEventListener("click", () => openRolloutModal(r));
      actions.appendChild(roll);
    }
    head.appendChild(actions);
    box.appendChild(head);

    const spec = r.spec || {};
    const meta = el("p", "note");
    meta.style.margin = "0";
    meta.style.color = "var(--muted)";
    meta.style.fontSize = ".8rem";
    if (r.kind === "train") {
      meta.textContent =
        `${spec.policy_type} · ${spec.repo_id} · ${(spec.episodes || []).length} episodes · ` +
        `${spec.steps} steps · batch ${spec.batch_size} · ${spec.device}` +
        (spec.eval_split ? ` · eval_split ${spec.eval_split}` : "");
    } else if (r.kind === "export") {
      meta.textContent = `${spec.repo_id} → ${spec.new_repo_id}, dropping ${(spec.delete_episodes || []).length} episode(s)`;
    } else if (r.kind === "rollout") {
      meta.textContent = `${spec.policy_path} · ${spec.duration_s}s · ${spec.port}`;
    } else if (r.kind === "record") {
      meta.textContent =
        `${spec.repo_id} · ${spec.resume ? "resume" : "new"} · up to ${spec.num_episodes} episodes · `
        + `${spec.fps} fps · ${(spec.cameras || []).join(", ")} · "${spec.task}"`;
    }
    box.appendChild(meta);

    if (r.kind === "record" && r.status === "running") {
      const help = el("p", "note", "");
      help.style.margin = "0";
      help.style.fontSize = ".82rem";
      help.textContent =
        "Drive the session from the headset (B ends an episode, Y discards and re-records) "
        + "or from here. This page has no keyboard access to the recorder, so use Stop "
        + "session rather than killing the run — an episode is only written on a clean end.";
      box.appendChild(help);

      const ctl = el("div", "session-controls");
      const ctlBtn = (label, cls, action, title) => {
        const b = el("button", "btn btn-sm " + cls, label);
        b.title = title;
        b.addEventListener("click", () => {
          sendRecordControl(action);
          toast("sent: " + label.toLowerCase(), "ok");
          // Ending an episode twice in quick succession saves one with zero
          // frames, and lerobot raises inside save_episode — taking the
          // whole session down. Rate-limit rather than trust the mouse.
          b.disabled = true;
          setTimeout(() => { b.disabled = false; }, 1500);
        });
        ctl.appendChild(b);
      };
      ctlBtn("End episode", "btn-accent", "end_episode",
        "Same as B on the right controller: save this episode and move on.");
      ctlBtn("Re-record", "", "rerecord",
        "Same as Y on the left controller: throw this episode away and redo it.");
      ctlBtn("Stop session", "btn-danger", "stop",
        "Same as Esc: finish cleanly, saving the current episode.");
      box.appendChild(ctl);
    }

    if (r.kind === "train") {
      const chart = el("div", "chart");
      const ch = el("div", "chart-head");
      ch.appendChild(el("b", null, "Loss"));
      ch.appendChild(el("span", null, "log scale"));
      const legend = el("span", "legend");
      const lg = (color, label) => {
        const s = el("span");
        const i = el("i"); i.style.background = color;
        s.appendChild(i); s.appendChild(document.createTextNode(label));
        legend.appendChild(s);
      };
      lg("var(--accent)", "train");
      lg("var(--warn)", "eval (held out)");
      ch.appendChild(legend);
      chart.appendChild(ch);
      const svg = svgEl("svg", { id: "loss-chart", viewBox: "0 0 800 220" });
      chart.appendChild(svg);
      box.appendChild(chart);

      const stats = el("div", "card-stats");
      stats.id = "run-stats";
      stats.style.padding = ".1rem 0";
      box.appendChild(stats);
    }

    const logPre = el("pre");
    logPre.id = "run-log";
    if (r.kind !== "train") logPre.style.height = "26rem";
    logPre.textContent = S.logText;
    box.appendChild(logPre);
    drawLossChart();
  }

  function drawLossChart() {
    const svg = $("loss-chart");
    if (!svg) return;
    svg.textContent = "";
    const train = S.metrics.filter((m) => m.kind === "train" && m.loss > 0);
    const evals = S.metrics.filter((m) => m.kind === "eval" && m.eval_loss > 0);
    if (!train.length) {
      const t = svgEl("text", { x: 400, y: 110, "text-anchor": "middle", fill: "#5c6270", "font-size": 12 });
      t.textContent = "waiting for the first logged step…";
      svg.appendChild(t);
      return;
    }
    const all = train.map((m) => m.loss).concat(evals.map((m) => m.eval_loss));
    // Loss spans orders of magnitude early on (ACT starts in the tens and
    // lands near 1); a linear axis would flatten everything after step 50.
    const lo = Math.log10(Math.max(1e-6, Math.min(...all)));
    const hi = Math.log10(Math.max(...all));
    const stepMax = Math.max(...train.map((m) => m.steps), 1);
    const sc = scales(svg, stepMax, lo - 0.05 * (hi - lo || 1), hi + 0.05 * (hi - lo || 1));
    const ticks = [];
    for (let p = Math.floor(lo); p <= Math.ceil(hi); p++) ticks.push(p);
    for (const p of ticks) {
      const yy = sc.y(p);
      svg.appendChild(svgEl("line", {
        x1: PAD.l, x2: sc.w - PAD.r, y1: yy, y2: yy, stroke: "var(--border)",
        "stroke-width": 1, "vector-effect": "non-scaling-stroke",
      }));
      const label = svgEl("text", {
        x: PAD.l - 5, y: yy + 3, "text-anchor": "end", fill: "#5c6270",
        "font-size": 9, "font-family": "ui-monospace, monospace",
      });
      label.textContent = Math.pow(10, p) >= 1 ? String(Math.pow(10, p)) : Math.pow(10, p).toExponential(0);
      svg.appendChild(label);
    }
    const mkPath = (rows, key, color, width) => {
      if (!rows.length) return;
      svg.appendChild(svgEl("path", {
        d: linePath(rows.map((m) => m.steps), rows.map((m) => Math.log10(Math.max(1e-6, m[key]))), sc.x, sc.y),
        fill: "none", stroke: color, "stroke-width": width, "vector-effect": "non-scaling-stroke",
      }));
    };
    mkPath(train, "loss", "var(--accent)", 1.6);
    mkPath(evals, "eval_loss", "var(--warn)", 1.6);
    // x-axis label: the final step, so the axis is readable without ticks.
    const xl = svgEl("text", {
      x: sc.w - PAD.r, y: sc.h - 3, "text-anchor": "end", fill: "#5c6270",
      "font-size": 9, "font-family": "ui-monospace, monospace",
    });
    xl.textContent = "step " + stepMax;
    svg.appendChild(xl);

    const stats = $("run-stats");
    if (stats) {
      stats.textContent = "";
      const last = train[train.length - 1];
      const lastEval = evals[evals.length - 1];
      const stat = (label, value) => {
        const s = el("span");
        s.appendChild(el("b", null, value));
        s.appendChild(document.createTextNode(" " + label));
        stats.appendChild(s);
      };
      stat("step", `${last.steps} / ${(S.run.spec || {}).steps || "?"}`);
      stat("loss", fmtNum(last.loss));
      if (lastEval) stat("eval loss", fmtNum(lastEval.eval_loss));
      stat("epochs", fmtNum(last.epochs, 2));
      stat("lr", fmtNum(last.lr, 6));
      if (last.samples_per_s) stat("smp/s", fmtNum(last.samples_per_s, 0));
      if (last.gpu_mem_gb) stat("GB", fmtNum(last.gpu_mem_gb, 1));
    }
  }

  async function pollRun() {
    if (!S.run) return;
    const id = S.run.id;
    try {
      const [m, l, r] = await Promise.all([
        api(`/api/run/metrics?id=${encodeURIComponent(id)}&offset=${S.metricsOffset}`),
        api(`/api/run/log?id=${encodeURIComponent(id)}&offset=${S.logOffset}`),
        api(`/api/run?id=${encodeURIComponent(id)}`),
      ]);
      if (m.rows.length) { S.metrics = S.metrics.concat(m.rows); }
      S.metricsOffset = m.offset;
      if (l.text) { S.logText += l.text; }
      S.logOffset = l.offset;
      const statusChanged = S.run.status !== r.status;
      S.run = r;
      if (statusChanged) { renderRunDetail(); loadRuns(); }
      const pre = $("run-log");
      if (pre && l.text) {
        const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 30;
        pre.textContent = S.logText;
        if (atBottom) pre.scrollTop = pre.scrollHeight;
      }
      if (m.rows.length) drawLossChart();
    } catch (e) { /* transient; the next tick retries */ }
  }

  function startPolling(on) {
    if (S.timer) { clearInterval(S.timer); S.timer = null; }
    if (!on) return;
    S.timer = setInterval(() => {
      if (S.run && S.run.status === "running") pollRun();
      // A recording session appends episodes as it goes; keep the dataset
      // list honest while one is running.
      else if (S.runs.some((r) => r.kind === "record" && r.status === "running")) loadRuns();
    }, POLL_MS);
  }

  // ── rollout ────────────────────────────────────────────────────────────
  async function openRolloutModal(run) {
    let cps = [];
    try { cps = (await api("/api/run/checkpoints?id=" + encodeURIComponent(run.id))).checkpoints; }
    catch (e) { /* handled below */ }
    await loadEnv();

    const card = $("modal-card");
    card.textContent = "";
    card.appendChild(el("h3", null, "Run this policy on the arm"));

    if (!cps.length) {
      card.appendChild(el("p", "note", "This run has no saved checkpoint yet."));
      const row = el("div", "btn-row");
      const close = el("button", "btn", "Close");
      close.addEventListener("click", closeModal);
      row.appendChild(close);
      card.appendChild(row);
      $("modal").classList.add("show");
      return;
    }

    // Preflight. Each of these has cost a real session before: the relay
    // holding the RealSense the policy needs to see through, something
    // else holding the servo bus, or a camera list that does not match
    // what the policy was trained on.
    const spec = run.spec || {};
    const problems = [];
    const datasetCams = (spec.camera_keys || []).length ? spec.camera_keys : null;
    if (S.env) {
      if (!S.env.serial.exists)
        problems.push(`No arm at ${S.env.serial.port}.`);
      for (const h of (S.env.serial.holders || []))
        problems.push(`${S.env.serial.port} is held by pid ${h.pid} (${h.cmd.slice(0, 80)}). Stop it first — concurrent access corrupts Feetech packets.`);
      for (const c of S.env.relay_cameras)
        problems.push(`This relay holds camera "${c.id}" (${c.device}). If the policy needs that camera, restart the relay without it.`);
    }

    if (problems.length) {
      const warn = el("div", "warn-box err");
      warn.appendChild(document.createTextNode("Before you start:"));
      const ul = el("ul");
      problems.forEach((p) => ul.appendChild(el("li", null, p)));
      warn.appendChild(ul);
      card.appendChild(warn);
    }

    const safety = el("div", "warn-box");
    safety.textContent =
      "The arm will move on its own. It ramps to the rest pose first (goal-position presync, " +
      "calibration and limit gates), then runs the policy for the duration below and returns " +
      "to where it started. Keep the workspace clear and stay near the power switch.";
    card.appendChild(safety);

    const form = el("div", "form");
    const field = (label, node, hint) => {
      const f = el("div", "field");
      f.appendChild(el("label", null, label));
      f.appendChild(node);
      if (hint) f.appendChild(el("span", "hint", hint));
      form.appendChild(f);
      return node;
    };
    const cpSel = el("select");
    cps.forEach((c) => {
      const o = el("option", null, c.name + (c.is_link ? " (latest)" : ""));
      o.value = c.path;
      cpSel.appendChild(o);
    });
    field("Checkpoint", cpSel);
    const dur = el("input"); dur.type = "number"; dur.min = "1"; dur.value = "30";
    field("Duration (s)", dur, "required — never open-ended");
    const port = el("input"); port.type = "text";
    port.value = (S.env && S.env.serial.port) || "/dev/ttyACM0";
    field("Serial port", port);
    const robotId = el("input"); robotId.type = "text"; robotId.value = "so101";
    field("Robot id", robotId, "must match lerobot-calibrate");
    const cams = el("input"); cams.type = "text";
    cams.value = (datasetCams || ["top"]).map((k) => k.replace("observation.images.", "") + "=auto").join(" ");
    field("Cameras", cams, "NAME=SOURCE, space separated — must match recording");
    card.appendChild(form);

    const taskField = el("div", "field");
    taskField.appendChild(el("label", null, "Task"));
    const task = el("input"); task.type = "text";
    task.value = spec.task_text || "";
    task.placeholder = "same wording as the recorded task";
    taskField.appendChild(task);
    card.appendChild(taskField);

    const row = el("div", "btn-row");
    const cancel = el("button", "btn", "Cancel");
    cancel.addEventListener("click", closeModal);
    const dry = el("button", "btn", "Dry run");
    dry.title = "Print the exact command and run the preflight checks without touching the arm.";
    const go = el("button", "btn btn-danger", "Start rollout");
    const launch = async (dryRun) => {
      go.disabled = dry.disabled = true;
      try {
        const r = await post("/api/run/rollout", {
          policy_path: cpSel.value,
          duration_s: Number(dur.value),
          port: port.value,
          robot_id: robotId.value,
          cameras: cams.value.trim().split(/\s+/).filter(Boolean),
          task: task.value,
          source_run: run.id,
          dry_run: dryRun,
          name: (run.name || run.id) + (dryRun ? "-dryrun" : "-rollout"),
        });
        closeModal();
        toast(dryRun ? "Dry run started." : "Rollout started — watch the arm.", "ok");
        await loadRuns();
        openRun(r.id);
      } catch (e) { toast(e.message, "err"); go.disabled = dry.disabled = false; }
    };
    dry.addEventListener("click", () => launch(true));
    go.addEventListener("click", () => launch(false));
    row.appendChild(cancel); row.appendChild(dry); row.appendChild(go);
    card.appendChild(row);
    $("modal").classList.add("show");
  }

  // ── boot ───────────────────────────────────────────────────────────────
  (async () => {
    let saved = 1;
    try { saved = parseFloat(localStorage.getItem("vrteleop.rate")) || 1; } catch (_) { /* private mode */ }
    S.rate = SPEEDS.includes(saved) ? saved : 1;
    applyRate();
    await loadEnv();
    await loadDatasets();
    refreshTrainForm();
    const params = new URL(location.href).searchParams;
    if (params.get("repo_id")) openDataset(params.get("repo_id"));
  })();
})();

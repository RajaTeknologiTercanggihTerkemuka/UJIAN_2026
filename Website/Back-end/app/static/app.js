(() => {
  const esc = (v) =>
    String(v ?? '—').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmtNum = (v, d = 3) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '—');
  const fmtPct = (v, d = 2) => (typeof v === 'number' && Number.isFinite(v) ? (v * 100).toFixed(d) + '%' : '—');
  const fmtTime = (iso) => {
    if (!iso) return '—';
    const dt = new Date(iso);
    return Number.isNaN(dt.getTime()) ? iso : dt.toLocaleString('id-ID', { hour12: false });
  };
  const NIM_REGEX = /^\d{8,9}$/;

  const state = {
    activeTab: 'register',
    adminKey: sessionStorage.getItem('ta_v2_admin_key') || '',
    streams: { reg: null, ver: null },
    busy: false,
  };

  const appShell = document.getElementById('appShell');
  const workCanvas = document.getElementById('workCanvas');
  const workCtx = workCanvas.getContext('2d', { willReadFrequently: true });

  // ==================== HEALTH CHECK ====================
  async function checkHealth() {
    const dot = document.getElementById('healthDot');
    const txt = document.getElementById('healthText');
    try {
      const r = await fetch('/api/health');
      const d = await r.json();
      if (d.model_ready) {
        dot.className = 'health-dot ok';
        txt.textContent = `Model siap (${d.model_version})`;
      } else {
        dot.className = 'health-dot err';
        txt.textContent = 'Model belum siap: ' + (d.startup_error || 'cek .env');
      }
    } catch {
      dot.className = 'health-dot err';
      txt.textContent = 'API tidak terhubung';
    }
  }

  // ==================== INPUT VALIDATION (NAMA & NIM 8-9 DIGIT) ====================
  function validateIdentityFields(scope) {
    const name = document.getElementById(scope + 'Name').value.trim();
    const nim = document.getElementById(scope + 'Nim').value.trim();
    if (!name) {
      showResult(scope, 'VALIDASI GAGAL', 'Nama wajib diisi terlebih dahulu sebelum mengaktifkan kamera.', '');
      return null;
    }
    if (!NIM_REGEX.test(nim)) {
      showResult(scope, 'VALIDASI GAGAL', 'NIM harus terdiri dari 8 hingga 9 digit angka saja (contoh: 04221001).', '');
      return null;
    }
    return { name, nim };
  }

  // ==================== TAB NAVIGATION & THEME ====================
  function stopAllStreams() {
    ['reg', 'ver'].forEach((scope) => {
      if (state.streams[scope]) {
        state.streams[scope].getTracks().forEach((t) => t.stop());
        state.streams[scope] = null;
        const vid = document.getElementById(scope + 'Video');
        if (vid) vid.srcObject = null;
        const submitBtn = document.getElementById(scope + 'SubmitCam');
        if (submitBtn) submitBtn.disabled = true;
        const banner = document.getElementById(scope + 'Banner');
        if (banner) banner.textContent = 'Isi Nama & NIM (8–9 digit), lalu klik "Aktifkan Kamera"';
      }
    });
  }

  function switchTab(tabName) {
    state.activeTab = tabName;
    document.body.setAttribute('data-theme', tabName);
    document.querySelectorAll('.tab-btn').forEach((b) => b.classList.toggle('active', b.dataset.tab === tabName));
    document.querySelectorAll('.tab-panel').forEach((p) => p.classList.toggle('active', p.id === 'panel-' + tabName));
    stopAllStreams();

    if (tabName === 'admin' && state.adminKey) {
      appShell.classList.add('full-width');
      loadAdminDashboard();
    } else {
      appShell.classList.remove('full-width');
    }
  }

  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // ==================== CAMERA & 1-SECOND BLINK + HEAD ACTIVE LIVENESS ====================
  async function startCamera(scope) {
    if (!validateIdentityFields(scope)) return;
    stopAllStreams();
    const video = document.getElementById(scope + 'Video');
    const banner = document.getElementById(scope + 'Banner');
    const submitBtn = document.getElementById(scope + 'SubmitCam');
    banner.textContent = 'Meminta izin kamera...';
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      state.streams[scope] = stream;
      video.srcObject = stream;
      await video.play();
      submitBtn.disabled = false;
      banner.textContent = 'Kamera aktif — posisikan wajah di dalam bingkai oval';
    } catch (e) {
      banner.textContent = 'Gagal mengakses kamera: ' + e.message;
    }
  }

  function sampleRegionsGray(video) {
    const w = 128;
    const h = 96;
    workCanvas.width = w;
    workCanvas.height = h;
    workCtx.drawImage(video, 0, 0, w, h);
    const full = workCtx.getImageData(0, 0, w, h).data;
    const toGray = (x0, y0, rw, rh) => {
      const arr = new Float32Array(rw * rh);
      let idx = 0;
      for (let y = y0; y < y0 + rh; y++) {
        for (let x = x0; x < x0 + rw; x++) {
          const p = (y * w + x) * 4;
          arr[idx++] = 0.299 * full[p] + 0.587 * full[p + 1] + 0.114 * full[p + 2];
        }
      }
      return arr;
    };
    return {
      // Upper-middle band inside oval where eyes sit (sensitive to genuine eye blink)
      eyeBand: toGray(42, 26, 44, 18),
      // Full face oval (for deliberate head turn / nod)
      faceOval: toGray(36, 14, 56, 66),
      // Outer left strip as background camera noise reference
      bgNoise: toGray(2, 10, 14, 70),
    };
  }

  function meanAbsDiff(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
    return sum / a.length;
  }

  function meanVal(a) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += a[i];
    return sum / a.length;
  }

  async function runActiveChallenge(scope) {
    const video = document.getElementById(scope + 'Video');
    const oval = document.getElementById(scope + 'Oval');
    const banner = document.getElementById(scope + 'Banner');
    oval.className = 'face-oval scanning';
    banner.textContent = 'Tes Liveness (1 detik): Kedipkan mata atau gerakkan kepala...';

    let prev = sampleRegionsGray(video);
    const baseEyeMean = meanVal(prev.eyeBand);
    let eyeSpikeFrames = 0;
    let eyeReturned = false;
    let headMotionFrames = 0;
    let passed = false;

    const durationMs = 1150;
    const stepMs = 75;
    const deadline = performance.now() + durationMs;

    while (performance.now() < deadline) {
      await new Promise((r) => setTimeout(r, stepMs));
      const cur = sampleRegionsGray(video);
      const bgFloor = Math.max(1.8, meanAbsDiff(prev.bgNoise, cur.bgNoise));
      const eyeDiff = meanAbsDiff(prev.eyeBand, cur.eyeBand);
      const faceDiff = meanAbsDiff(prev.faceOval, cur.faceOval);
      const eyeShift = Math.abs(meanVal(cur.eyeBand) - baseEyeMean);

      // 1) Localized eye blink detection: eye-band spikes noticeably above noise floor, then changes/returns
      if (eyeDiff >= Math.max(6.2, bgFloor * 2.6) || eyeShift >= Math.max(4.5, bgFloor * 2.0)) {
        eyeSpikeFrames += 1;
      } else if (eyeSpikeFrames >= 1) {
        eyeReturned = true;
      }

      // 2) Deliberate head movement (must exceed stricter threshold across >=3 frames so tiny tremor won't pass)
      if (faceDiff >= Math.max(8.8, bgFloor * 2.8)) {
        headMotionFrames += 1;
      }

      prev = cur;
      if ((eyeSpikeFrames >= 2 && eyeReturned) || eyeSpikeFrames >= 3 || headMotionFrames >= 3) {
        passed = true;
        break;
      }
    }

    if (passed) {
      oval.className = 'face-oval passed';
      banner.textContent = 'Liveness terdeteksi! Tahan wajah lurus sejenak...';
      await new Promise((r) => setTimeout(r, 320));
    } else {
      oval.className = 'face-oval';
      banner.textContent = 'Memeriksa keberadaan wajah di kamera...';
    }
    oval.className = 'face-oval';
    return passed;
  }

  async function captureFrameBlob(video) {
    workCanvas.width = video.videoWidth || 1280;
    workCanvas.height = video.videoHeight || 720;
    workCtx.drawImage(video, 0, 0, workCanvas.width, workCanvas.height);
    return new Promise((resolve) => workCanvas.toBlob(resolve, 'image/jpeg', 0.92));
  }

  // ==================== RENDER PARTICIPANT RESULT ====================
  function showResult(scope, decision, message, metaText) {
    const box = document.getElementById(scope + 'Result');
    const titleEl = document.getElementById(scope + 'ResultTitle');
    const msgEl = document.getElementById(scope + 'ResultMsg');
    const metaEl = document.getElementById(scope + 'ResultMeta');

    const map = {
      REGISTERED_NEW: { cls: 'success', label: 'LOLOS — Registrasi Peserta Baru Berhasil' },
      RE_REGISTERED: { cls: 'success', label: 'LOLOS — Daftar Ulang Berhasil' },
      MATCH: { cls: 'success', label: 'LOLOS — Wajah Terverifikasi Cocok' },
      REJECTED_DUPLICATE_FACE: { cls: 'danger', label: 'DITOLAK — Indikasi Joki (Duplikasi Wajah 1:N)' },
      REJECTED_NIM_FACE_MISMATCH: { cls: 'danger', label: 'DITOLAK — NIM Terdaftar pada Wajah Berbeda' },
      REJECTED_SPOOF: { cls: 'danger', label: 'DITOLAK — Indikasi Spoofing / Foto / Layar' },
      MISMATCH: { cls: 'danger', label: 'DITOLAK — Wajah Tidak Cocok' },
      ACCOUNT_FLAGGED: { cls: 'danger', label: 'DITOLAK — Akun Ditandai / Tidak Aktif' },
      REVIEW_REQUIRED: { cls: 'warning', label: 'PERLU PEMERIKSAAN OPERATOR' },
      VALIDASI_GAGAL: { cls: 'warning', label: 'Data Belum Valid' },
      'VALIDASI GAGAL': { cls: 'warning', label: 'Data Belum Valid' },
    };
    const info = map[decision] || { cls: 'warning', label: decision || 'Status' };
    box.className = `result-box show ${info.cls}`;
    titleEl.textContent = info.label;
    msgEl.textContent = message || '';
    metaEl.textContent = metaText || '';
  }

  // ==================== SUBMIT REGISTRASI & VERIFIKASI ====================
  async function handleSubmit(scope) {
    if (state.busy) return;
    const fields = validateIdentityFields(scope);
    if (!fields) return;

    const video = document.getElementById(scope + 'Video');
    if (!state.streams[scope] || !video.videoWidth) {
      showResult(scope, 'VALIDASI GAGAL', 'Aktifkan kamera terlebih dahulu.', '');
      return;
    }

    const isReg = scope === 'reg';
    state.busy = true;
    try {
      const challengeOn = document.getElementById(scope + 'ChallengeToggle').checked;
      let activePassed = true;
      if (challengeOn) {
        activePassed = await runActiveChallenge(scope);
      }

      // Always capture and send to server so MTCNN checks face presence FIRST
      // (empty camera -> "Wajah tidak terdeteksi", static face photo -> REJECTED_SPOOF).
      const imageBlob = await captureFrameBlob(video);
      showResult(scope, 'MEMPROSES', 'Menjalankan deteksi wajah MTCNN, Anti-Spoofing PAD, dan pencocokan embedding...', '');

      const form = new FormData();
      form.append('nim', fields.nim);
      form.append('full_name', fields.name);
      form.append('active_liveness_passed', activePassed ? 'true' : 'false');
      form.append('image', imageBlob, 'capture.jpg');

      const url = isReg ? '/api/registration/register' : '/api/verification/verify';
      const resp = await fetch(url, { method: 'POST', body: form });
      const data = await resp.json();

      if (!resp.ok) {
        const detail = typeof data.detail === 'object' ? (data.detail.message || JSON.stringify(data.detail)) : (data.detail || `HTTP ${resp.status}`);
        const code = typeof data.detail === 'object' ? data.detail.code : '';
        const heading = code === 'no_face' ? 'WAJAH TIDAK TERDETEKSI' : code === 'multiple_faces' ? 'TERDETEKSI LEBIH DARI 1 WAJAH' : 'GAGAL PROSES';
        showResult(scope, heading, detail, `HTTP ${resp.status}`);
        document.getElementById(scope + 'Banner').textContent = detail;
        return;
      }

      showResult(
        scope,
        data.decision_status,
        data.message,
        `NIM: ${data.nim_masked || '—'} • Request ID: ${data.request_id || '—'}`
      );
      document.getElementById(scope + 'Banner').textContent = 'Selesai — siap untuk pemindaian berikutnya';
      checkHealth();
    } catch (err) {
      showResult(scope, 'ERROR', 'Terjadi kesalahan jaringan/server: ' + err.message, '');
    } finally {
      state.busy = false;
    }
  }

  document.getElementById('regStartCam').addEventListener('click', () => startCamera('reg'));
  document.getElementById('verStartCam').addEventListener('click', () => startCamera('ver'));
  document.getElementById('regSubmitCam').addEventListener('click', () => handleSubmit('reg'));
  document.getElementById('verSubmitCam').addEventListener('click', () => handleSubmit('ver'));

  // ==================== ADMIN / OPERATOR & MLOPS PANEL ====================
  async function adminFetch(path, options = {}) {
    const headers = { 'X-Admin-Key': state.adminKey, ...(options.headers || {}) };
    const r = await fetch(path, { ...options, headers });
    const data = await r.json();
    if (!r.ok) {
      const err = new Error(data.detail || `HTTP ${r.status}`);
      err.status = r.status;
      throw err;
    }
    return data;
  }

  async function tryAdminLogin() {
    const key = document.getElementById('adminKeyInput').value.trim();
    const errEl = document.getElementById('adminLoginErr');
    errEl.textContent = '';
    if (!key) {
      errEl.textContent = 'Masukkan password operator.';
      return;
    }
    state.adminKey = key;
    try {
      await adminFetch('/api/admin/stats');
      sessionStorage.setItem('ta_v2_admin_key', key);
      document.getElementById('adminLoginView').style.display = 'none';
      document.getElementById('adminDashView').style.display = 'block';
      appShell.classList.add('full-width');
      loadAdminDashboard();
    } catch (e) {
      state.adminKey = '';
      sessionStorage.removeItem('ta_v2_admin_key');
      errEl.textContent = e.status === 401 ? 'Password salah. Coba lagi.' : ('Gagal masuk: ' + e.message);
    }
  }

  document.getElementById('adminLoginBtn').addEventListener('click', tryAdminLogin);
  document.getElementById('adminKeyInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') tryAdminLogin();
  });

  document.getElementById('btnAdminLogout').addEventListener('click', () => {
    state.adminKey = '';
    sessionStorage.removeItem('ta_v2_admin_key');
    document.getElementById('adminKeyInput').value = '';
    document.getElementById('adminDashView').style.display = 'none';
    document.getElementById('adminLoginView').style.display = 'block';
    appShell.classList.remove('full-width');
  });

  function decisionBadge(status) {
    if (['MATCH', 'REGISTERED_NEW', 'RE_REGISTERED'].includes(status)) {
      return `<span class="badge badge-green">${esc(status)}</span>`;
    }
    if (['REJECTED_DUPLICATE_FACE', 'REJECTED_NIM_FACE_MISMATCH', 'REJECTED_SPOOF', 'MISMATCH', 'ACCOUNT_FLAGGED'].includes(status)) {
      return `<span class="badge badge-red">${esc(status)}</span>`;
    }
    return `<span class="badge badge-amber">${esc(status)}</span>`;
  }

  function alertTypeLabel(t) {
    const m = {
      duplicate_face_different_nim: 'Joki Tipe B (Duplikat 1:N)',
      nim_takeover_attempt: 'Takeover NIM Terdaftar',
      spoof_on_register: 'Spoof saat Registrasi',
      spoof_on_verify: 'Spoof saat Verifikasi',
    };
    return m[t] || t;
  }

  async function loadAdminDashboard() {
    if (!state.adminKey) return;
    document.getElementById('adminLoginView').style.display = 'none';
    document.getElementById('adminDashView').style.display = 'block';
    appShell.classList.add('full-width');

    const alertStatus = document.getElementById('alertFilter').value;
    const evMode = document.getElementById('eventModeFilter').value;

    try {
      const [stats, mlops, alerts, demoUsers, events] = await Promise.all([
        adminFetch('/api/admin/stats'),
        adminFetch('/api/admin/mlops'),
        adminFetch('/api/admin/alerts?limit=100' + (alertStatus ? '&review_status=' + encodeURIComponent(alertStatus) : '')),
        adminFetch('/api/admin/demo-users'),
        adminFetch('/api/admin/events?limit=150' + (evMode ? '&mode=' + encodeURIComponent(evMode) : '')),
      ]);

      renderKPIs(stats);
      renderMLOps(mlops, stats);
      renderAlerts(alerts);
      renderDemoUsers(demoUsers);
      renderEvents(events);
      document.getElementById('adminLastUpdated').textContent =
        'Terakhir diperbarui: ' + new Date().toLocaleTimeString('id-ID', { hour12: false });
    } catch (e) {
      if (e.status === 401) {
        document.getElementById('btnAdminLogout').click();
      } else {
        document.getElementById('adminLastUpdated').textContent = 'Gagal memuat dashboard: ' + e.message;
      }
    }
  }

  function renderKPIs(stats) {
    const tel = stats.telemetry_averages || {};
    const items = [
      { label: 'Total Peserta DB', value: Number(stats.users.total).toLocaleString('id-ID'), sub: `Aktif: ${stats.users.active} • Ditandai: ${stats.users.flagged_joki}` },
      { label: 'Vektor 1:N Terindeks', value: Number(stats.vector_index.indexed_vectors).toLocaleString('id-ID'), sub: `Backend: ${stats.vector_index.backend || '—'}` },
      { label: 'Total Event Tercatat', value: Number(stats.events.total).toLocaleString('id-ID'), sub: `Match: ${stats.events.by_decision.MATCH || 0} • Reg Baru: ${stats.events.by_decision.REGISTERED_NEW || 0}` },
      { label: 'Alert Anti-Joki Pending', value: stats.events.pending_alerts, sub: `Joki Diblokir: ${(stats.events.by_decision.REJECTED_DUPLICATE_FACE || 0) + (stats.events.by_decision.REJECTED_NIM_FACE_MISMATCH || 0)}` },
      { label: 'Avg Similarity', value: fmtNum(tel.avg_similarity_score, 4), sub: `Th 1:1 ${fmtNum(stats.model.verify_threshold, 4)} • 1:N ${fmtNum(stats.model.dedup_threshold, 2)}` },
      { label: 'Avg Search / Total Latency', value: tel.avg_total_latency_ms ? tel.avg_total_latency_ms.toFixed(0) + ' ms' : '—', sub: `Pencarian: ${tel.avg_comparison_latency_ms ? tel.avg_comparison_latency_ms.toFixed(2) + ' ms' : '—'}` },
    ];
    document.getElementById('kpiGrid').innerHTML = items
      .map(
        (it) => `<div class="kpi-card">
          <div class="kpi-label">${esc(it.label)}</div>
          <div class="kpi-value">${esc(it.value)}</div>
          <div class="kpi-sub">${esc(it.sub)}</div>
        </div>`
      )
      .join('');
  }

  function renderMLOps(mlops, stats) {
    // 1. Drift Monitor (without retraining policy note box)
    const dm = mlops.drift_monitor || {};
    const dStatus = dm.status || 'HEALTHY';
    const dBadge = document.getElementById('driftBadge');
    const statusMap = {
      HEALTHY: { cls: 'badge-green', txt: 'HEALTHY — Stabil' },
      DRIFT_WARNING: { cls: 'badge-amber', txt: 'DRIFT WARNING' },
      RETRAINING_RECOMMENDED: { cls: 'badge-red', txt: 'PERLU EVALUASI MODEL' },
    };
    const sm = statusMap[dStatus] || statusMap.HEALTHY;
    dBadge.className = 'badge ' + sm.cls;
    dBadge.textContent = sm.txt;

    const bar = document.getElementById('driftStatusBar');
    bar.className = 'drift-status-bar ' + dStatus;
    document.getElementById('driftHeadline').textContent =
      dStatus === 'HEALTHY'
        ? 'Model V1 Beroperasi Normal & Stabil'
        : dStatus === 'DRIFT_WARNING'
        ? 'Peringatan Pergeseran Performa — Evaluasi Operator Disarankan'
        : 'Degradasi Performa Terdeteksi — Perlu Evaluasi Model V2';
    document.getElementById('driftWindow').textContent = `Window: ${dm.window_events || 0} event terakhir`;

    document.getElementById('driftKvGrid').innerHTML = [
      ['Avg Similarity (MATCH)', dm.avg_match_similarity !== null ? fmtNum(dm.avg_match_similarity, 4) : 'Belum ada'],
      ['Rasio Kegagalan / Mismatch', fmtPct(dm.failure_rate, 1)],
      ['Rasio Spoof Ditolak', fmtPct(dm.spoof_rate, 1)],
      ['Indikasi Joki Diblokir', `${dm.joki_blocked_count || 0} percobaan`],
      ['Avg PAD Live Score', dm.avg_pad_score !== null ? fmtNum(dm.avg_pad_score, 4) : '—'],
      ['Avg Total Latency', dm.avg_total_latency_ms !== null ? dm.avg_total_latency_ms + ' ms' : '—'],
    ]
      .map(([k, v]) => `<div class="kv-item"><span>${esc(k)}</span><b>${esc(v)}</b></div>`)
      .join('');

    document.getElementById('driftReasons').innerHTML = (dm.reasons || []).map((r) => `<li>${esc(r)}</li>`).join('');

    // 2. MLflow, Model V1, PAD & DVC
    const mlf = mlops.mlflow || {};
    const mlfBadge = document.getElementById('mlflowStatusBadge');
    mlfBadge.className = 'badge ' + (mlf.online ? 'badge-green' : 'badge-amber');
    mlfBadge.textContent = mlf.online ? 'MLflow: ONLINE' : 'MLflow: OFFLINE (Artefak Lokal Aktif)';
    if (mlf.tracking_uri) document.getElementById('mlflowLink').href = mlf.tracking_uri;

    const tm = mlops.verification_test_metrics || {};
    const ft = mlops.face_threshold_calibration || {};
    const pt = mlops.pad_threshold_calibration || {};
    const ptTest = pt.test_metrics_at_frozen_validation_threshold || {};
    const audit = mlops.audit_summary || {};

    document.getElementById('mlopsSummaryGrid').innerHTML = [
      ['Model Aktif', `${stats.model.model_version || '—'} (512-d ArcFace)`],
      ['Test Accuracy / ROC-AUC (IMSFD)', `${fmtPct(tm.test_accuracy, 2)} / ${fmtPct(tm.test_ROC_AUC, 2)}`],
      ['Test FAR / FRR (26.460 pasang)', `${fmtPct(tm.test_FAR, 2)} / ${fmtPct(tm.test_FRR, 2)}`],
      ['Kalibrasi MTCNN (1:1 / 1:N)', `${fmtNum(ft.threshold, 4)} (EER ${fmtPct(ft.eer_estimate, 2)}) / ${fmtNum(stats.model.dedup_threshold, 2)}`],
      ['Kalibrasi PAD MiniFASNetV2', `Th ${fmtNum(pt.threshold, 4)} • Test ROC-AUC ${fmtPct(ptTest.ROC_AUC, 2)}`],
      ['PAD BPCER / APCER (389 uji)', `BPCER ${fmtPct(ptTest.BPCER, 2)} • APCER ${fmtPct(ptTest.APCER, 2)}`],
    ]
      .map(([k, v]) => `<div class="kv-item"><span>${esc(k)}</span><b>${esc(v)}</b></div>`)
      .join('');

    const hist = mlops.training_history || [];
    document.getElementById('epochChart').innerHTML = hist
      .map((ep) => {
        const pct = Math.max(8, Math.min(100, Math.round((ep.val_top1 - 0.9) * 1000)));
        return `<div class="epoch-bar-wrap" title="Epoch ${ep.epoch}: Val Top-1 ${(ep.val_top1 * 100).toFixed(2)}% | Loss ${ep.val_loss}">
          <div class="epoch-bar" style="height:${pct}%"></div>
          <span class="epoch-lbl">${ep.epoch}</span>
        </div>`;
      })
      .join('');

    const dvcLines = (mlops.dvc_datasets || [])
      .map((d) => `${d.path}: ${Number(d.nfiles || 0).toLocaleString('id-ID')} file (${((d.size_bytes || 0) / (1024 ** 3)).toFixed(2)} GB)`)
      .join(' • ');
    document.getElementById('dvcBox').innerHTML =
      `<strong>Dataset &amp; DVC Versioning:</strong> IMSFD Audit ${Number(audit.valid_images || 0).toLocaleString('id-ID')} gambar (${audit.unique_identities || 68} identitas, 0 korup). ` +
      `DVC Snapshot: ${esc(dvcLines || 'Belum ada')}. MLflow Experiment: <code>${esc(mlf.experiment_name)}</code>.`;
  }

  function renderAlerts(alerts) {
    const wrap = document.getElementById('alertsTableWrap');
    if (!alerts.length) {
      wrap.innerHTML = '<p class="muted" style="padding:14px">Belum ada peringatan indikasi joki atau spoofing.</p>';
      return;
    }
    const rows = alerts
      .map(
        (a) => `<tr>
          <td>${esc(fmtTime(a.created_at))}</td>
          <td><span class="badge badge-red">${esc(alertTypeLabel(a.alert_type))}</span></td>
          <td>${esc(a.claimed_nim_masked)}</td>
          <td>${esc(a.claimed_name || '—')}</td>
          <td>${esc(a.matched_nim_masked || '—')}</td>
          <td><strong>${fmtNum(a.similarity_score, 4)}</strong> / ${fmtNum(a.threshold_used, 2)}</td>
          <td><span class="badge ${a.review_status === 'pending' ? 'badge-amber' : a.review_status === 'confirmed_joki' ? 'badge-red' : 'badge-green'}">${esc(a.review_status)}</span></td>
          <td>
            <button class="btn btn-danger-soft btn-sm" onclick="window.__reviewAlert(${a.id}, 'confirmed_joki', true)">Konfirmasi Joki</button>
            <button class="btn btn-outline btn-sm" onclick="window.__reviewAlert(${a.id}, 'false_positive', false)">False Positive</button>
          </td>
        </tr>`
      )
      .join('');
    wrap.innerHTML = `<table>
      <thead><tr><th>Waktu</th><th>Jenis Peringatan</th><th>NIM Klaim</th><th>Nama Klaim</th><th>Cocok dengan NIM</th><th>Skor / Th</th><th>Status</th><th>Aksi Operator</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  }

  window.__reviewAlert = async (alertId, reviewStatus, flagUser) => {
    try {
      await adminFetch(`/api/admin/alerts/${alertId}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          review_status: reviewStatus,
          reviewer_notes: `Ditinjau operator via dashboard (${reviewStatus})`,
          flag_matched_user: flagUser,
        }),
      });
      loadAdminDashboard();
    } catch (e) {
      alert('Gagal memperbarui alert: ' + e.message);
    }
  };

  function renderDemoUsers(users) {
    const wrap = document.getElementById('demoUsersWrap');
    if (!users.length) {
      wrap.innerHTML = '<p class="muted" style="padding:14px">Belum ada peserta terdaftar.</p>';
      return;
    }
    const rows = users
      .map(
        (u) => `<tr>
          <td><code>${esc(u.nim)}</code></td>
          <td>${esc(u.full_name)}</td>
          <td>${esc(u.exam_year)}</td>
          <td><span class="badge ${u.status === 'active' ? 'badge-green' : 'badge-red'}">${esc(u.status)}</span></td>
          <td><span class="badge badge-blue">${esc(u.source_type || '—')}</span></td>
          <td>${esc(fmtTime(u.created_at))}</td>
          <td><button class="btn btn-danger-soft btn-sm" onclick="window.__deleteNim('${esc(u.nim)}')">Hapus</button></td>
        </tr>`
      )
      .join('');
    wrap.innerHTML = `<table>
      <thead><tr><th>NIM</th><th>Nama</th><th>Tahun</th><th>Status</th><th>Sumber Template</th><th>Terdaftar</th><th>Aksi</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  }

  window.__deleteNim = async (nim) => {
    if (!confirm(`Hapus peserta dengan NIM ${nim} beserta template wajahnya?`)) return;
    try {
      await adminFetch(`/api/admin/users/${encodeURIComponent(nim)}`, { method: 'DELETE' });
      loadAdminDashboard();
      checkHealth();
    } catch (e) {
      alert('Gagal menghapus NIM: ' + e.message);
    }
  };

  document.getElementById('btnDeleteNim').addEventListener('click', () => {
    const nim = document.getElementById('deleteNimInput').value.trim();
    if (!nim) return alert('Ketik NIM terlebih dahulu.');
    window.__deleteNim(nim);
    document.getElementById('deleteNimInput').value = '';
  });

  function renderEvents(events) {
    const wrap = document.getElementById('eventsTableWrap');
    if (!events.length) {
      wrap.innerHTML = '<p class="muted" style="padding:14px">Belum ada riwayat event.</p>';
      return;
    }
    const rows = events
      .map(
        (e) => `<tr>
          <td>${esc(fmtTime(e.created_at))}</td>
          <td><span class="badge badge-blue">${esc(e.mode)}</span></td>
          <td>${esc(e.nim_masked)}</td>
          <td>${decisionBadge(e.decision_status)}</td>
          <td>${esc(e.matched_nim_masked || '—')}</td>
          <td>${fmtNum(e.similarity_score, 4)} / ${fmtNum(e.threshold_used, 2)}</td>
          <td>${esc(e.antispoof_status)} (${fmtNum(e.antispoof_score, 3)})</td>
          <td>${esc(e.candidate_count ?? '—')} (${esc(e.search_backend || '—')})</td>
          <td>${e.comparison_latency_ms !== null ? e.comparison_latency_ms.toFixed(2) + ' ms' : '—'}</td>
          <td>${e.total_latency_ms !== null ? e.total_latency_ms.toFixed(0) + ' ms' : '—'}</td>
          <td>${esc(e.failure_reason || '—')}</td>
        </tr>`
      )
      .join('');
    wrap.innerHTML = `<table>
      <thead><tr>
        <th>Waktu</th><th>Mode</th><th>NIM</th><th>Keputusan</th><th>Match NIM</th>
        <th>Similarity / Th</th><th>PAD (Skor)</th><th>Kandidat (Backend)</th>
        <th>Search Latency</th><th>Total Latency</th><th>Catatan</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  }

  document.getElementById('btnRefreshAdmin').addEventListener('click', loadAdminDashboard);
  document.getElementById('btnResetEvents').addEventListener('click', async () => {
    if (!confirm('Reset seluruh riwayat event registrasi/verifikasi dan daftar peringatan (alerts)? Data peserta & template wajah tetap aman.')) return;
    try {
      const res = await adminFetch('/api/admin/events', { method: 'DELETE' });
      alert(`Riwayat berhasil direset (${res.deleted_events} event dan ${res.deleted_alerts} alert dihapus).`);
      loadAdminDashboard();
    } catch (e) {
      alert('Gagal mereset riwayat: ' + e.message);
    }
  });
  document.getElementById('alertFilter').addEventListener('change', loadAdminDashboard);
  document.getElementById('eventModeFilter').addEventListener('change', loadAdminDashboard);
  document.getElementById('btnReloadIndex').addEventListener('click', async () => {
    try {
      const res = await adminFetch('/api/admin/index/reload', { method: 'POST' });
      alert(`Vector index dimuat ulang: ${res.indexed_vectors} vektor (${res.backend})`);
      loadAdminDashboard();
      checkHealth();
    } catch (e) {
      alert('Gagal reload index: ' + e.message);
    }
  });

  // ==================== INIT ====================
  checkHealth();
  const params = new URLSearchParams(window.location.search);
  const initialTab = params.get('tab');
  if (initialTab && ['register', 'verify', 'admin'].includes(initialTab)) {
    switchTab(initialTab);
  }
  window.addEventListener('beforeunload', stopAllStreams);
})();

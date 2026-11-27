// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

// Browser-only client for the Qwen3-Omni-Duplex deployment. It speaks the
// duplex window protocol directly to vLLM: 200 ms pcm_f32le appends at
// 16 kHz in, pcm16 at 24 kHz out; no proxy sits in between. Every window the
// Thinker decides listen or speak by itself; this page only moves audio, shows
// what the session reported, and (optionally) answers the System 2 calls the
// model makes.
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const ui = Object.fromEntries(
    ['url', 'model', 'voice', 'autoAnswer', 'answerText', 'probe', 'start', 'stop', 'mute',
      'status', 'playback', 'sid', 'inTime', 'outTime', 'userText', 'assistantText', 'log', 'download']
      .map((id) => [id, $(id)]),
  );
  const INPUT_RATE = 16000;
  const OUTPUT_RATE = 24000;
  const FLUSH_MS = 200;
  const UNIT_MS = 480; // one duplex decision window
  const INITIAL_BUFFER_MS = 400;
  const WORKLET_DIR = '.'; // pcm_worklet.js / playback_worklet.js sit next to this page

  let socket = null;
  let stream = null;
  let captureContext = null;
  let captureNode = null;
  let playbackContext = null;
  let playbackNode = null;
  let timer = null;
  let pending = [];
  let sentSamples = 0;
  let receivedSamples = 0;
  let muted = false;
  let stopping = false;
  let sessionReady = false;
  let responses = new Map();
  let playbackResponse = null;
  let logLines = 0;
  let recorded = [];
  let currentTurnText = '';

  // System 2 stand-in: the launch with ``result_index`` N is answered with the
  // N-th step of the "delay|content|delay|content" script, ``delay`` windows
  // after the launch was seen; a cancel is acknowledged at once with "".
  let dueAnswers = [];

  function parseScript(text) {
    const parts = text.split('|').map((part) => part.trim());
    if (parts.length < 2 || parts.length % 2) throw new Error('script must be <windows>|<content> pairs');
    const steps = [];
    for (let i = 0; i < parts.length; i += 2) {
      if (!/^\d+$/.test(parts[i])) throw new Error(`step ${i / 2 + 1}: windows "${parts[i]}" is not a non-negative integer`);
      steps.push({ delay: Number(parts[i]), content: parts[i + 1] });
    }
    return steps;
  }

  function log(text) {
    const line = document.createElement('div');
    line.textContent = text;
    ui.log.appendChild(line);
    if (++logLines > 300) { ui.log.firstChild.remove(); logLines--; }
    ui.log.scrollTop = ui.log.scrollHeight;
  }

  function tick() {
    ui.inTime.textContent = `${(sentSamples / INPUT_RATE).toFixed(1)}s`;
    ui.outTime.textContent = `${(receivedSamples / OUTPUT_RATE).toFixed(1)}s`;
  }

  function realtimeUrl(address, model) {
    const url = new URL(address);
    if (!['ws:', 'wss:'].includes(url.protocol) || !url.host || url.username || url.password) {
      throw new Error('url must be ws:// or wss:// without credentials');
    }
    if (location.protocol === 'https:' && url.protocol !== 'wss:') throw new Error('https page requires wss://');
    if (url.pathname === '/') url.pathname = '/v1/realtime';
    url.searchParams.set('duplex', '1');
    url.searchParams.set('model', model);
    url.searchParams.set('autostart', '0');
    return url.toString();
  }

  function sessionUpdate(model) {
    return {
      type: 'session.update',
      event_id: `evt_${crypto.randomUUID()}`,
      session: {
        model,
        modalities: ui.voice.checked ? ['text', 'audio'] : ['text'],
        input_audio_format: 'pcm_f32le',
        output_audio_format: 'pcm16',
        sample_rate_hz: INPUT_RATE,
        audio: { input: { sample_rate_hz: INPUT_RATE }, output: { sample_rate_hz: OUTPUT_RATE } },
        turn_detection: null,
        extra_body: { auto_response: true },
      },
    };
  }

  function send(payload) {
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(payload));
  }

  function toBase64(bytes) {
    let result = '';
    for (let offset = 0; offset < bytes.length; offset += 8192) {
      result += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
    }
    return btoa(result);
  }

  function fromBase64(encoded) {
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  function answerDue() {
    const sentMs = sentSamples * 1000 / INPUT_RATE;
    for (const due of dueAnswers.filter((entry) => entry.at <= sentMs)) {
      dueAnswers = dueAnswers.filter((entry) => entry !== due);
      log(`[s2] send output call=${due.callId.slice(-8)} payload=${due.output}`);
      send({ type: 'conversation.item.create', item: { type: 'function_call_output', call_id: due.callId, output: due.output } });
    }
  }

  function flush() {
    if (socket?.readyState !== WebSocket.OPEN) return;
    answerDue();
    if (!pending.length) return;
    const count = pending.reduce((total, part) => total + part.length, 0);
    const chunk = new Int16Array(count);
    let offset = 0;
    for (const part of pending) { chunk.set(part, offset); offset += part.length; }
    pending = [];
    const bytes = new Uint8Array(chunk.length * 4);
    const view = new DataView(bytes.buffer);
    for (let i = 0; i < chunk.length; i++) view.setFloat32(i * 4, chunk[i] / 32768, true);
    sentSamples += chunk.length;
    send({
      type: 'input_audio_buffer.append', event_id: `evt_${crypto.randomUUID()}`,
      audio: toBase64(bytes), format: 'pcm_f32le', sample_rate_hz: INPUT_RATE,
      duration_ms: Math.round(chunk.length * 1000 / INPUT_RATE),
      audio_end_ms: Math.round(sentSamples * 1000 / INPUT_RATE),
    });
    tick();
  }

  function responseId(event) { return event.response_id || event.response?.id || null; }

  function acknowledge(responseIdValue, playedMs) {
    if (!responseIdValue || !(playedMs > 0) || socket?.readyState !== WebSocket.OPEN) return;
    send({
      type: 'playback.ack', event_id: `evt_${crypto.randomUUID()}`, response_id: responseIdValue,
      item_id: `item_${responseIdValue}`, played_ms: playedMs, committed_ms: playedMs,
    });
  }

  function handleCall(item) {
    const name = String(item.name || '');
    const callId = String(item.call_id || '');
    log(`[s2] recv ${name} call=${callId.slice(-8)} arguments=${item.arguments || '{}'}`);
    if (!ui.autoAnswer.checked) return;
    if (name === 'system2_cancel') {
      log(`[s2] send cancel_ack call=${callId.slice(-8)}`);
      send({ type: 'conversation.item.create', item: { type: 'function_call_output', call_id: callId, output: '' } });
      return;
    }
    if (name !== 'system2_launch') return;
    let arguments_ = {};
    try { arguments_ = JSON.parse(String(item.arguments || '{}')); } catch { log(`[s2] skip launch call=${callId.slice(-8)}: arguments is not JSON`); return; }
    let steps;
    try { steps = parseScript(ui.answerText.value); } catch (error) { log(`[s2] skip launch call=${callId.slice(-8)}: ${error.message}`); return; }
    const index = Math.max(0, Number(arguments_.result_index) || 0);
    const last = index >= steps.length - 1;
    const { delay, content } = steps[Math.min(index, steps.length - 1)];
    const output = JSON.stringify({ content: last ? content + '[SYSTEM2_END]' : content, final: last });
    dueAnswers.push({ at: sentSamples * 1000 / INPUT_RATE + delay * UNIT_MS, callId, output });
    log(`[s2] schedule gen=${String(arguments_.generation_id || '').slice(-8)} result_index=${index} call=${callId.slice(-8)} delay=${delay}w final=${last}`);
  }

  function handleEvent(event) {
    const raw = { ...event };
    if (raw.type === 'response.output_audio.delta') raw.delta = `<${String(raw.delta || '').length} b64 chars>`;
    log(`[recv] ${JSON.stringify(raw)}`);

    const type = event.type;
    const id = responseId(event);
    if (type === 'session.created') {
      ui.sid.textContent = event.session?.session_id || event.session?.id || event.session_id || '—';
      return;
    }
    if (type === 'error') {
      const message = event.error?.message || event.message || JSON.stringify(event.error || event);
      ui.status.textContent = `Error: ${message}`;
      if (!sessionReady) void stop();
      return;
    }
    if (type === 'response.created') {
      if (id) responses.set(id, { audioSamples: 0 });
      currentTurnText = '';
      ui.assistantText.textContent = '';
    } else if (type === 'response.output_audio_transcript.delta' || type === 'response.output_text.delta') {
      currentTurnText += event.delta || '';
      ui.assistantText.textContent = currentTurnText;
    } else if (type === 'response.output_audio_transcript.done') {
      if (typeof event.transcript === 'string') ui.assistantText.textContent = currentTurnText = event.transcript;
    } else if (type === 'response.output_audio.delta') {
      if (event.format && event.format !== 'pcm16') { log(`[audio] drop delta: unsupported format=${event.format}`); return; }
      if (event.sample_rate_hz && event.sample_rate_hz !== OUTPUT_RATE) { log(`[audio] drop delta: unexpected sample_rate_hz=${event.sample_rate_hz}`); return; }
      const bytes = fromBase64(event.delta || '');
      if (bytes.length % 2) { log('[audio] drop delta: pcm16 payload has odd byte length'); return; }
      const view = new DataView(bytes.buffer);
      const pcm = new Int16Array(bytes.length / 2);
      for (let i = 0; i < pcm.length; i++) pcm[i] = view.getInt16(i * 2, true);
      const response = responses.get(id);
      if (response) response.audioSamples += pcm.length;
      receivedSamples += pcm.length;
      // Copy: postMessage below transfers the buffer away from this view.
      recorded.push(new Int16Array(pcm));
      playbackResponse = id;
      playbackNode?.port.postMessage({ type: 'audio', pcm, responseId: id, initialBufferMs: INITIAL_BUFFER_MS }, [pcm.buffer]);
      tick();
    } else if (type === 'response.done') {
      // Drain only when the model finished the turn, never at the end of an
      // input chunk: the Talker can lag several windows behind the Thinker.
      if (id && id === playbackResponse) playbackNode?.port.postMessage({ type: 'drain', responseId: id });
    } else if (type === 'conversation.item.input_audio_transcription.delta') {
      ui.userText.textContent += event.delta || '';
    } else if (type === 'conversation.item.input_audio_transcription.completed') {
      ui.userText.textContent = event.transcript || '';
    } else if (type === 'input_audio_buffer.speech_started') {
      ui.userText.textContent = '';
    } else if (type === 'response.output_item.done') {
      const item = event.item || {};
      if (item.type === 'function_call') handleCall(item);
    } else if (type === 'session.closed') {
      void stop();
    }
  }

  async function openAudio() {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    playbackContext = new AudioContext({ sampleRate: OUTPUT_RATE });
    if (playbackContext.sampleRate !== OUTPUT_RATE) throw new Error('playback AudioContext did not grant 24 kHz');
    await playbackContext.audioWorklet.addModule(`${WORKLET_DIR}/playback_worklet.js`);
    playbackNode = new AudioWorkletNode(playbackContext, 'fullduplex-pcm-playback');
    playbackNode.port.onmessage = ({ data }) => {
      if (data.type === 'playback-started') ui.playback.textContent = 'Playing';
      else if (data.type === 'playback-stopped') acknowledge(data.responseId, data.playedMs);
      else if (data.type === 'playback-drained') {
        ui.playback.textContent = 'Off';
        acknowledge(data.responseId, data.playedMs);
        if (data.responseId) responses.delete(data.responseId);
        if (data.responseId === playbackResponse) playbackResponse = null;
      } else if (data.type === 'playback-underrun') log(`[audio] playback underrun ms=${data.underrunMs || 0}`);
    };
    playbackNode.connect(playbackContext.destination);
    await playbackContext.resume();

    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true, sampleRate: { ideal: INPUT_RATE } },
    });
    captureContext = new AudioContext({ sampleRate: INPUT_RATE });
    if (captureContext.sampleRate !== INPUT_RATE) throw new Error('capture AudioContext did not grant 16 kHz');
    await captureContext.audioWorklet.addModule(`${WORKLET_DIR}/pcm_worklet.js`);
    captureNode = new AudioWorkletNode(captureContext, 'fullduplex-pcm-capture', {
      channelCount: 1, channelCountMode: 'explicit', channelInterpretation: 'speakers',
    });
    captureNode.port.onmessage = ({ data }) => pending.push(new Int16Array(data));
    captureContext.createMediaStreamSource(stream).connect(captureNode);
    const silent = captureContext.createGain();
    silent.gain.value = 0;
    captureNode.connect(silent).connect(captureContext.destination);
    await captureContext.resume();
    timer = setInterval(flush, FLUSH_MS);
  }

  function wavBlob() {
    const samples = recorded.reduce((total, part) => total + part.length, 0);
    const buffer = new ArrayBuffer(44 + samples * 2);
    const view = new DataView(buffer);
    const ascii = (offset, text) => { for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i)); };
    ascii(0, 'RIFF'); view.setUint32(4, 36 + samples * 2, true); ascii(8, 'WAVEfmt ');
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, OUTPUT_RATE, true); view.setUint32(28, OUTPUT_RATE * 2, true);
    view.setUint16(32, 2, true); view.setUint16(34, 16, true); ascii(36, 'data');
    view.setUint32(40, samples * 2, true);
    const body = new Int16Array(buffer, 44, samples); // 44 is even, so the view is aligned
    let offset = 0;
    for (const part of recorded) {
      body.set(part, offset);
      offset += part.length;
    }
    return samples ? new Blob([buffer], { type: 'audio/wav' }) : null;
  }

  async function stop() {
    if (stopping) return;
    stopping = true;
    if (timer !== null) clearInterval(timer);
    timer = null;
    if (socket?.readyState === WebSocket.OPEN) {
      if (sessionReady) flush();
      send({ type: 'session.close', event_id: `evt_${crypto.randomUUID()}` });
    }
    pending = [];
    if (captureNode) captureNode.port.onmessage = null;
    if (playbackNode) { playbackNode.port.onmessage = null; playbackNode.port.postMessage({ type: 'clear' }); }
    stream?.getTracks().forEach((track) => track.stop());
    stream = null;
    const oldSocket = socket;
    socket = null;
    oldSocket?.close();
    await Promise.all([captureContext?.close(), playbackContext?.close()]);
    captureContext = captureNode = playbackContext = playbackNode = null;
    responses = new Map();
    dueAnswers = [];
    playbackResponse = null;
    sessionReady = false;
    muted = false;
    const wav = wavBlob();
    if (wav) {
      ui.download.href = URL.createObjectURL(wav);
      ui.download.download = 'duplex_session.wav';
      ui.download.hidden = false;
    }
    ui.mute.textContent = '闭麦';
    ui.start.disabled = false;
    ui.stop.disabled = ui.mute.disabled = true;
    ui.url.disabled = ui.model.disabled = ui.voice.disabled = false;
    ui.status.textContent = 'Offline';
    ui.playback.textContent = 'Off';
    stopping = false;
  }

  async function start() {
    if (ui.start.disabled) return;
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('getUserMedia unavailable: requires localhost or HTTPS');
      const model = ui.model.value.trim();
      if (!model) throw new Error('model is empty');
      const address = realtimeUrl(ui.url.value, model);
      ui.start.disabled = true;
      ui.url.disabled = ui.model.disabled = ui.voice.disabled = true;
      ui.status.textContent = 'Connecting';
      ui.sid.textContent = '—';
      ui.userText.textContent = ui.assistantText.textContent = currentTurnText = '';
      sentSamples = receivedSamples = 0;
      pending = [];
      responses = new Map();
      dueAnswers = [];
      playbackResponse = null;
      recorded = [];
      ui.download.hidden = true;
      tick();
      const connected = new WebSocket(address);
      socket = connected;
      connected.onopen = () => {
        if (socket === connected) send(sessionUpdate(model));
      };
      connected.onmessage = async ({ data }) => {
        if (socket !== connected) return;
        try {
          const event = JSON.parse(data);
          handleEvent(event);
          if (event.type === 'session.created' && !sessionReady) {
            sessionReady = true;
            await openAudio();
            if (socket !== connected) return;
            ui.status.textContent = 'Connected';
            ui.stop.disabled = ui.mute.disabled = false;
          }
        } catch (error) {
          log(`[session] handle message failed: ${error.message || error}`);
          await stop();
        }
      };
      connected.onerror = () => log(
        `[ws] connect failed url=${address} (host is resolved on the browser machine; ` +
        'forward the vLLM port to it when debugging remotely and check the port is the actual server port)',
      );
      connected.onclose = () => {
        if (socket === connected) { log('[ws] closed'); void stop(); }
      };
      log(`[ws] connecting url=${address}`);
    } catch (error) {
      log(`[session] start failed: ${error.message || error}`);
      await stop();
    }
  }

  // Reachability probe: a no-cors GET needs no CORS headers, so success means
  // the address answered at all (port forwarded and service listening) and a
  // failure means the browser never reached it.
  async function probe() {
    let base;
    try {
      const url = new URL(ui.url.value.trim());
      if (!['ws:', 'wss:'].includes(url.protocol)) throw new Error('url must be ws:// or wss://');
      base = `${url.protocol === 'wss:' ? 'https:' : 'http:'}//${url.host}/v1/models`;
    } catch (error) {
      log(`[probe] invalid url: ${error.message}`);
      return;
    }
    log(`[probe] GET ${base}`);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    try {
      await fetch(base, { mode: 'no-cors', cache: 'no-store', signal: controller.signal });
      log(`[probe] reachable url=${base} (HTTP answered; a WebSocket failure now means the server rejected the upgrade)`);
    } catch (error) {
      log(`[probe] unreachable url=${base} reason=${error.name === 'AbortError' ? 'timeout(5s)' : 'refused-or-unresolvable'}` +
        ' (check port forwarding and that the server is running)');
    } finally {
      clearTimeout(timeout);
    }
  }

  ui.probe.addEventListener('click', () => { void probe(); });
  ui.start.addEventListener('click', start);
  ui.stop.addEventListener('click', () => { void stop(); });
  ui.mute.addEventListener('click', () => {
    muted = !muted;
    stream?.getAudioTracks().forEach((track) => { track.enabled = !muted; });
    ui.mute.textContent = muted ? '开麦' : '闭麦';
  });
  window.addEventListener('beforeunload', () => { void stop(); });
})();

/**
 * AI Support Assistant - Industrial Operations Console Client
 * Native WebSocket integration with FastAPI /ws/live backend.
 */

(function () {
  'use strict';

  // --- DOM Elements ---
  // Header & System Telemetry
  const statusDot = document.getElementById('status-dot');
  const statusLabel = document.getElementById('status-label');
  const sessionTimer = document.getElementById('session-timer');
  const turnCountText = document.getElementById('turn-count');
  const deviceLabel = document.getElementById('device-label');
  const themeToggleBtn = document.getElementById('theme-toggle-btn');
  const agentNameEl = document.getElementById('agent-name');
  const startBtn = document.getElementById('start-btn');
  const stopBtn = document.getElementById('stop-btn');
  const logoutBtn = document.getElementById('logout-btn');
  const autoscrollBtn = document.getElementById('autoscroll-btn');
  const streamIndicator = document.getElementById('stream-indicator');

  // Subsystems Telemetry
  const dotMic = document.getElementById('dot-mic');
  const valMic = document.getElementById('val-mic');
  const dotStt = document.getElementById('dot-stt');
  const valStt = document.getElementById('val-stt');
  const dotRole = document.getElementById('dot-role');
  const valRole = document.getElementById('val-role');
  const dotGemini = document.getElementById('dot-gemini');
  const valGemini = document.getElementById('val-gemini');
  const dotRag = document.getElementById('dot-rag');
  const valRag = document.getElementById('val-rag');

  // Error Banner
  const errorBanner = document.getElementById('error-banner');
  const errorMessage = document.getElementById('error-message');
  const errorClose = document.getElementById('error-close');

  // Conversation Timeline
  const transcriptScroll = document.getElementById('transcript-scroll');
  const transcriptEmpty = document.getElementById('transcript-empty');
  const conversationFeed = document.getElementById('conversation-feed');
  const partialContainer = document.getElementById('partial-container');
  const partialSpeaker = document.getElementById('partial-speaker');
  const partialText = document.getElementById('partial-text');
  const partialTime = document.getElementById('partial-time');

  // AI Support Recommendation
  const groundingBadge = document.getElementById('grounding-badge');
  const suggestionConfidence = document.getElementById('suggestion-confidence');
  const suggestionLatency = document.getElementById('suggestion-latency');
  const suggestionLatencyVal = document.getElementById('suggestion-latency-val');
  const suggestionsEmpty = document.getElementById('suggestions-empty');
  const ungroundedAlert = document.getElementById('ungrounded-alert');
  const suggestionContent = document.getElementById('suggestion-content');
  const sectionDiagnosis = document.getElementById('section-diagnosis');
  const sectionActionsHeader = document.getElementById('section-actions-header');
  const suggestionSummaryText = document.getElementById('suggestion-summary-text');
  const suggestionStepsList = document.getElementById('suggestion-steps-list');
  const suggestionSafetyBox = document.getElementById('suggestion-safety-box');
  const suggestionSafetyText = document.getElementById('suggestion-safety-text');

  // Suggested Questions
  const questionsPanel = document.getElementById('questions-panel');
  const questionsEmpty = document.getElementById('questions-empty');
  const questionsList = document.getElementById('questions-list');
  const questionsCount = document.getElementById('questions-count');

  // Sources Provenance
  const sourcesEmpty = document.getElementById('sources-empty');
  const sourcesContainer = document.getElementById('sources-container');
  const sourcesList = document.getElementById('sources-list');
  const sourcesCount = document.getElementById('sources-count');

  // Customer Analysis
  const sentimentBadge = document.getElementById('sentiment-badge');
  const sentimentConfidence = document.getElementById('sentiment-confidence');
  const sentimentRationale = document.getElementById('sentiment-rationale');
  const sentimentDot = document.getElementById('sentiment-dot');

  const categoryBadge = document.getElementById('category-badge');
  const categoryConfidence = document.getElementById('category-confidence');
  const categoryRationale = document.getElementById('category-rationale');
  const categoryDot = document.getElementById('category-dot');

  // --- State Variables ---
  let ws = null;
  let reconnectTimeout = null;
  let reconnectDelay = 1000;
  const MAX_RECONNECT_DELAY = 10000;
  let isIntentionallyClosed = false;
  let turnCount = 0;
  let isRunning = false;
  let autoScrollEnabled = true;
  let sessionStartTime = null;
  let timerInterval = null;
  let lastCustomerTurnTimestamp = null;
  let lastCustomerStatement = null;
  let confirmedIssueDiagnosis = null;

  // --- Helper: Format Timestamp ---
  function formatTime(isoString) {
    if (!isoString) {
      return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }
    try {
      const d = new Date(isoString);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return new Date().toLocaleTimeString();
    }
  }

  // --- Helper: Format Timer Seconds into HH:MM:SS ---
  function formatDuration(totalSeconds) {
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    const pad = (n) => String(n).padStart(2, '0');
    return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
  }

  // --- Timer Controls ---
  function startSessionTimer() {
    stopSessionTimer();
    sessionStartTime = Date.now();
    timerInterval = setInterval(function () {
      if (sessionStartTime && sessionTimer) {
        const elapsedSec = Math.floor((Date.now() - sessionStartTime) / 1000);
        sessionTimer.textContent = formatDuration(elapsedSec);
      }
    }, 1000);
  }

  function stopSessionTimer() {
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }
  }

  function resetSessionTimer() {
    stopSessionTimer();
    sessionStartTime = null;
    if (sessionTimer) {
      sessionTimer.textContent = '00:00:00';
    }
  }

  // --- Scroll Management ---
  function scrollTranscriptToBottom() {
    if (autoScrollEnabled && transcriptScroll) {
      transcriptScroll.scrollTop = transcriptScroll.scrollHeight;
    }
  }

  if (autoscrollBtn) {
    autoscrollBtn.addEventListener('click', function () {
      autoScrollEnabled = !autoScrollEnabled;
      if (autoScrollEnabled) {
        autoscrollBtn.classList.add('active');
        scrollTranscriptToBottom();
      } else {
        autoscrollBtn.classList.remove('active');
      }
    });
  }

  if (transcriptScroll) {
    transcriptScroll.addEventListener('scroll', function () {
      const distFromBottom = transcriptScroll.scrollHeight - transcriptScroll.scrollTop - transcriptScroll.clientHeight;
      if (distFromBottom > 80 && autoScrollEnabled) {
        autoScrollEnabled = false;
        if (autoscrollBtn) autoscrollBtn.classList.remove('active');
      } else if (distFromBottom <= 20 && !autoScrollEnabled) {
        autoScrollEnabled = true;
        if (autoscrollBtn) autoscrollBtn.classList.add('active');
      }
    });
  }

  // --- Error Banner Management ---
  function showError(msg) {
    if (errorMessage && errorBanner) {
      errorMessage.textContent = msg;
      errorBanner.classList.remove('hidden');
    }
  }

  function hideError() {
    if (errorBanner) {
      errorBanner.classList.add('hidden');
    }
  }

  if (errorClose) {
    errorClose.addEventListener('click', hideError);
  }

  // --- Reset Dashboard State for New Session ---
  function resetDashboardState() {
    turnCount = 0;
    lastCustomerTurnTimestamp = null;

    if (turnCountText) {
      turnCountText.textContent = '0 turns';
    }

    // Reset session timer
    resetSessionTimer();

    // Clear dialogue feed & show empty state
    if (conversationFeed) {
      conversationFeed.innerHTML = '';
    }
    if (transcriptEmpty) {
      transcriptEmpty.classList.remove('hidden');
    }

    // Clear partial transcript container
    if (partialContainer) {
      partialContainer.classList.add('hidden');
    }
    if (partialText) {
      partialText.textContent = '';
    }
    if (partialSpeaker) {
      partialSpeaker.textContent = 'Customer';
    }

    // Reset recommendation hero card
    if (suggestionsEmpty) {
      suggestionsEmpty.classList.remove('hidden');
    }
    if (suggestionContent) {
      suggestionContent.classList.add('hidden');
    }
    if (ungroundedAlert) {
      ungroundedAlert.classList.add('hidden');
    }
    if (sectionDiagnosis) {
      sectionDiagnosis.classList.remove('hidden');
    }
    if (sectionActionsHeader) {
      sectionActionsHeader.classList.remove('hidden');
    }
    if (suggestionSummaryText) {
      suggestionSummaryText.innerHTML = '';
    }
    if (suggestionStepsList) {
      suggestionStepsList.innerHTML = '';
    }
    if (suggestionSafetyBox) {
      suggestionSafetyBox.classList.add('hidden');
    }
    if (suggestionSafetyText) {
      suggestionSafetyText.textContent = '';
    }
    if (groundingBadge) {
      groundingBadge.className = 'grounding-pill empty';
      groundingBadge.textContent = 'Awaiting Context';
    }
    if (suggestionConfidence) {
      suggestionConfidence.textContent = '';
    }
    if (suggestionLatency) {
      suggestionLatency.classList.add('hidden');
    }

    // Reset suggested questions (kept hidden unless questions arrive)
    if (questionsPanel) {
      questionsPanel.classList.add('hidden');
    }
    if (questionsEmpty) {
      questionsEmpty.classList.remove('hidden');
    }
    if (questionsList) {
      questionsList.classList.add('hidden');
      questionsList.innerHTML = '';
    }
    if (questionsCount) {
      questionsCount.textContent = 'Agent Assistance';
    }

    // Reset sources citations
    if (sourcesEmpty) {
      sourcesEmpty.classList.remove('hidden');
    }
    if (sourcesContainer) {
      sourcesContainer.classList.add('hidden');
    }
    if (sourcesList) {
      sourcesList.innerHTML = '';
    }
    if (sourcesCount) {
      sourcesCount.textContent = '0 citations';
    }

    // Reset sentiment card
    if (sentimentBadge) {
      sentimentBadge.className = 'sentiment-badge empty';
      sentimentBadge.textContent = 'No sentiment detected';
    }
    if (sentimentConfidence) {
      sentimentConfidence.textContent = '';
    }
    if (sentimentRationale) {
      sentimentRationale.textContent = '';
      sentimentRationale.classList.add('hidden');
    }
    if (sentimentDot) {
      sentimentDot.className = 'mini-status-dot sentiment-dot';
    }

    // Reset category card
    if (categoryBadge) {
      categoryBadge.className = 'category-badge empty';
      categoryBadge.textContent = 'No category detected';
    }
    if (categoryConfidence) {
      categoryConfidence.textContent = '';
    }
    if (categoryRationale) {
      categoryRationale.textContent = '';
      categoryRationale.classList.add('hidden');
    }
    if (categoryDot) {
      categoryDot.className = 'mini-status-dot category-dot';
    }

    // Reset Subsystem statuses
    updateSubsystemsState({
      mic: 'Standby',
      stt: 'Idle',
      role: 'Ready',
      gemini: 'Standby',
      rag: 'Indexed'
    });

    hideError();
  }

  // Expose reset state globally for test harnesses
  if (typeof window !== 'undefined') {
    window.resetDashboardState = resetDashboardState;
  }

  // --- Update Subsystem Telemetry Indicators ---
  function updateSubsystemsState(states) {
    if (states.mic && valMic && dotMic) {
      valMic.textContent = states.mic;
      dotMic.className = 'subsystem-dot ' + (states.mic === 'Active' ? 'active' : 'standby');
    }
    if (states.stt && valStt && dotStt) {
      valStt.textContent = states.stt;
      dotStt.className = 'subsystem-dot ' + (states.stt === 'Streaming' ? 'busy' : (states.stt === 'Listening' ? 'active' : 'standby'));
    }
    if (states.role && valRole && dotRole) {
      valRole.textContent = states.role;
      dotRole.className = 'subsystem-dot ' + (states.role === 'Active' ? 'active' : 'standby');
    }
    if (states.gemini && valGemini && dotGemini) {
      valGemini.textContent = states.gemini;
      dotGemini.className = 'subsystem-dot ' + (states.gemini === 'Synthesizing...' ? 'busy' : (states.gemini === 'Ready' ? 'active' : 'standby'));
    }
    if (states.rag && valRag && dotRag) {
      valRag.textContent = states.rag;
      dotRag.className = 'subsystem-dot ' + (states.rag.includes('Retrieved') ? 'active' : 'standby');
    }
  }

  // --- Update System Status State ---
  function updateSystemStatus(status, running) {
    if (typeof running === 'boolean') {
      isRunning = running;
    }

    const norm = (status || 'unknown').toLowerCase();

    if (statusLabel) {
      if (norm === 'running' || norm === 'started') {
        statusLabel.textContent = 'Listening';
      } else if (norm === 'processing') {
        statusLabel.textContent = 'Processing';
      } else {
        statusLabel.textContent = status.charAt(0).toUpperCase() + status.slice(1);
      }
    }

    if (statusDot) {
      statusDot.className = 'status-beacon';
      if (norm === 'connected') {
        statusDot.classList.add(isRunning ? 'running' : 'connected');
      } else if (norm === 'running' || norm === 'started') {
        statusDot.classList.add('running');
        isRunning = true;
      } else if (norm === 'processing') {
        statusDot.classList.add('processing');
      } else if (norm === 'stopped') {
        statusDot.classList.add('stopped');
        isRunning = false;
      } else if (norm === 'disconnected') {
        statusDot.classList.add('disconnected');
        isRunning = false;
      } else if (norm === 'error') {
        statusDot.classList.add('error');
      }
    }

    // Controls state
    if (startBtn && stopBtn) {
      if (norm === 'disconnected') {
        startBtn.disabled = true;
        stopBtn.disabled = true;
      } else {
        startBtn.disabled = isRunning;
        stopBtn.disabled = !isRunning;
      }
    }

    // Stream beacon
    if (streamIndicator) {
      const textSpan = streamIndicator.querySelector('.beacon-text');
      if (isRunning) {
        streamIndicator.classList.add('active');
        if (textSpan) textSpan.textContent = 'LIVE';
      } else {
        streamIndicator.classList.remove('active');
        if (textSpan) textSpan.textContent = 'IDLE';
      }
    }

    // Subsystems adjustment
    if (isRunning) {
      updateSubsystemsState({ mic: 'Active', stt: 'Listening' });
    } else {
      updateSubsystemsState({ mic: 'Standby', stt: 'Idle' });
    }
  }

  // --- WebSocket Setup ---
  function getWebSocketUrl() {
    const loc = window.location;
    const protocol = loc.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = loc.host || 'localhost:5000';
    return `${protocol}//${host}/ws/live`;
  }

  function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const wsUrl = getWebSocketUrl();
    console.log('[Dashboard] Connecting to WebSocket:', wsUrl);

    try {
      ws = new WebSocket(wsUrl);
    } catch (err) {
      console.error('[Dashboard] WebSocket instantiation error:', err);
      showError('WebSocket creation error: ' + err.message);
      scheduleReconnect();
      return;
    }

    ws.onopen = function () {
      console.log('[Dashboard] Connected to /ws/live');
      reconnectDelay = 1000;
      updateSystemStatus('connected', isRunning);
      hideError();
    };

    ws.onmessage = function (event) {
      try {
        const payload = JSON.parse(event.data);
        handlePipelineEvent(payload);
      } catch (err) {
        console.error('[Dashboard] JSON parsing error for payload:', event.data, err);
      }
    };

    ws.onerror = function (event) {
      console.error('[Dashboard] WebSocket error:', event);
      updateSystemStatus('error');
    };

    ws.onclose = function (event) {
      console.warn('[Dashboard] WebSocket closed:', event.code, event.reason);
      updateSystemStatus('disconnected', false);
      stopSessionTimer();
      if (event.code === 1008) {
        console.warn('[Dashboard] Unauthorized WebSocket connection. Redirecting to login.');
        window.location.href = '/login';
        return;
      }
      if (!isIntentionallyClosed) {
        scheduleReconnect();
      }
    };
  }

  function scheduleReconnect() {
    if (reconnectTimeout) clearTimeout(reconnectTimeout);
    reconnectTimeout = setTimeout(function () {
      reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY);
      connectWebSocket();
    }, reconnectDelay);
  }

  // --- Pipeline Event Dispatcher ---
  function handlePipelineEvent(payload) {
    if (!payload || !payload.type) return;

    switch (payload.type) {
      case 'transcript_partial':
        handleTranscriptPartial(payload);
        break;

      case 'transcript_final':
        handleTranscriptFinal(payload);
        break;

      case 'sentiment_update':
        handleSentimentUpdate(payload);
        break;

      case 'category_update':
        handleCategoryUpdate(payload);
        break;

      case 'suggestion_update':
        handleSuggestionUpdate(payload);
        break;

      case 'issue_resolved':
        handleIssueResolved(payload);
        break;

      case 'system_status':
        handleSystemStatus(payload);
        break;

      case 'system_error':
        handleSystemError(payload);
        break;

      default:
        console.debug('[Dashboard] Unhandled pipeline event:', payload.type);
    }
  }

  // ============================================================================
  // Event Handlers
  // ============================================================================

  // 1. Partial Utterance Stream
  function handleTranscriptPartial(data) {
    if (!data.text || !data.text.trim()) {
      if (partialContainer) partialContainer.classList.add('hidden');
      return;
    }

    if (partialContainer && partialText && partialSpeaker) {
      let spk = (data.speaker || 'Customer').trim();
      if (spk.toLowerCase() === 'speaker' || spk.toLowerCase() === 'speaker a' || spk.toUpperCase() === 'A') {
        spk = 'Customer';
      } else if (spk.toLowerCase() === 'speaker b' || spk.toUpperCase() === 'B') {
        spk = 'Agent';
      }

      partialSpeaker.textContent = spk;
      partialText.textContent = data.text;
      if (partialTime) {
        partialTime.textContent = formatTime();
      }

      partialContainer.classList.remove('hidden');

      updateSubsystemsState({ stt: 'Streaming' });
      scrollTranscriptToBottom();
    }
  }

  // 2. Finalized Dialogue Turn
  function handleTranscriptFinal(data) {
    // Hide partial utterance box
    if (partialContainer) {
      partialContainer.classList.add('hidden');
      if (partialText) partialText.textContent = '';
    }

    if (!data.text || !data.text.trim()) return;

    // Hide empty state
    if (transcriptEmpty) {
      transcriptEmpty.classList.add('hidden');
    }

    // Increment and update turn counter
    turnCount += 1;
    if (turnCountText) {
      turnCountText.textContent = `${turnCount} turn${turnCount === 1 ? '' : 's'}`;
    }

    // Map speaker role
    const assignedRole = (data.role || (data.speaker && data.speaker.toLowerCase().includes('agent') ? 'AGENT' : 'CUSTOMER')).toUpperCase();
    const isCustomer = assignedRole === 'CUSTOMER';
    const isAgent = assignedRole === 'AGENT';

    const cardClass = isCustomer ? 'customer-turn' : (isAgent ? 'agent-turn' : 'unknown-turn');
    const badgeClass = isCustomer ? 'customer' : (isAgent ? 'agent' : 'unknown');
    const displayRole = isCustomer ? 'Customer' : (isAgent ? 'Agent' : 'Unknown');
    const avatarLetter = isCustomer ? 'C' : (isAgent ? 'A' : '?');

    // Acoustic label
    const acousticRaw = data.raw_speaker || (isCustomer ? 'A' : 'B');
    const timeStr = formatTime(data.timestamp);

    // Diarization confidence & latency
    let metaText = '';
    if (typeof data.role_confidence === 'number') {
      const confPct = Math.round(data.role_confidence * 100);
      metaText += `Diarization: ${confPct}%`;
    }
    if (typeof data.role_latency_ms === 'number' && data.role_latency_ms > 0) {
      metaText += ` (${Math.round(data.role_latency_ms)}ms)`;
    }

    // Build timeline card with avatar and content
    const turnCard = document.createElement('div');
    turnCard.className = `turn-card ${cardClass}`;
    turnCard.innerHTML = `
      <div class="turn-avatar ${badgeClass}">${avatarLetter}</div>
      <div class="turn-content">
        <div class="turn-header">
          <div class="speaker-cluster">
            <span class="role-badge ${badgeClass}">${escapeHtml(displayRole)}</span>
          </div>
          <span class="turn-time">${escapeHtml(timeStr)}</span>
        </div>
        <div class="turn-body">${escapeHtml(data.text)}</div>
      </div>
    `;

    if (conversationFeed) {
      conversationFeed.appendChild(turnCard);
      scrollTranscriptToBottom();
    }

    // If customer turn, record timestamp and set state to Processing
    if (isCustomer) {
      lastCustomerTurnTimestamp = performance.now();
      updateSystemStatus('processing', isRunning);
      updateSubsystemsState({ gemini: 'Synthesizing...' });
    } else {
      updateSubsystemsState({ stt: 'Listening' });
    }
  }


  // 3. Customer Sentiment Update
  function handleSentimentUpdate(data) {
    if (!sentimentBadge) return;

    const sentiment = (data.sentiment || '').trim();
    if (!sentiment) return;

    sentimentBadge.textContent = sentiment;
    sentimentBadge.className = 'sentiment-badge';

    const norm = sentiment.toLowerCase();
    if (norm === 'positive') {
      sentimentBadge.classList.add('positive');
      if (sentimentDot) sentimentDot.style.backgroundColor = '#10b981';
    } else if (norm === 'neutral') {
      sentimentBadge.classList.add('neutral');
      if (sentimentDot) sentimentDot.style.backgroundColor = '#38bdf8';
    } else if (norm === 'negative') {
      sentimentBadge.classList.add('negative');
      if (sentimentDot) sentimentDot.style.backgroundColor = '#f59e0b';
    } else if (norm === 'agitated') {
      sentimentBadge.classList.add('agitated');
      if (sentimentDot) sentimentDot.style.backgroundColor = '#ef4444';
    }

    if (sentimentConfidence) {
      if (typeof data.confidence === 'number') {
        const pct = Math.round(data.confidence <= 1.0 ? data.confidence * 100 : data.confidence);
        sentimentConfidence.textContent = `${pct}% conf`;
      } else {
        sentimentConfidence.textContent = '';
      }
    }

    if (sentimentRationale) {
      if (data.rationale) {
        sentimentRationale.textContent = data.rationale;
        sentimentRationale.classList.remove('hidden');
      } else {
        sentimentRationale.classList.add('hidden');
      }
    }
  }

  // 4. Query Category Update
  function handleCategoryUpdate(data) {
    if (!categoryBadge) return;

    const category = (data.category || '').trim();
    if (!category) return;

    categoryBadge.textContent = category;
    categoryBadge.className = 'category-badge';

    const norm = category.toLowerCase();
    if (norm.includes('operation')) {
      categoryBadge.classList.add('machine-operation');
      if (categoryDot) categoryDot.style.backgroundColor = '#3b82f6';
    } else if (norm.includes('maintenance') || norm.includes('part')) {
      categoryBadge.classList.add('maintenance-parts');
      if (categoryDot) categoryDot.style.backgroundColor = '#a855f7';
    } else if (norm.includes('troubleshooting') || norm.includes('technical')) {
      categoryBadge.classList.add('technical-troubleshooting');
      if (categoryDot) categoryDot.style.backgroundColor = '#f59e0b';
    }

    if (categoryConfidence) {
      if (typeof data.confidence === 'number') {
        const pct = Math.round(data.confidence <= 1.0 ? data.confidence * 100 : data.confidence);
        categoryConfidence.textContent = `${pct}% conf`;
      } else {
        categoryConfidence.textContent = '';
      }
    }

    if (categoryRationale) {
      if (data.rationale) {
        categoryRationale.textContent = data.rationale;
        categoryRationale.classList.remove('hidden');
      } else {
        categoryRationale.classList.add('hidden');
      }
    }
  }

  // 5. Support Suggestion & Grounding Update (Primary Focus)
  function handleIssueResolved(data) {
    // 1. Clear previous suggestion action cards and render resolved confirmation
    const cardsContainer = document.getElementById('suggestions-cards');
    if (cardsContainer) {
      cardsContainer.innerHTML = `
        <div class="resolved-banner">
          <svg class="w-6 h-6 text-emerald-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
          <div>
            <h4 class="text-sm font-semibold text-emerald-400">Customer Issue Resolved</h4>
            <p class="text-xs text-slate-300 mt-1">${(data && data.message) || (data && data.suggestion) || "The customer confirmed the problem has been resolved. Previous technical recommendations cleared."}</p>
          </div>
        </div>
      `;
    }

    // 2. Clear previous source citations and active highlight state
    renderSourceCitations([], []);

    // 3. Clear summary, safety warning, and unverified details notice
    const summaryEl = document.getElementById('suggestion-summary');
    if (summaryEl) summaryEl.textContent = 'Issue confirmed resolved by customer.';

    const safetyEl = document.getElementById('suggestion-safety');
    if (safetyEl) safetyEl.classList.add('hidden');

    const unverifiedNotice = document.getElementById('suggestion-unverified-notice');
    if (unverifiedNotice) unverifiedNotice.classList.add('hidden');

    if (ungroundedAlert) ungroundedAlert.classList.add('hidden');
    if (suggestionsEmpty) suggestionsEmpty.classList.add('hidden');
    if (suggestionContent) suggestionContent.classList.remove('hidden');

    // 4. Update Grounding status badge to Issue Resolved
    if (groundingBadge) {
      groundingBadge.className = 'grounding-pill resolved';
      groundingBadge.textContent = 'Issue Resolved';
    }

    // 5. Update confidence badge
    if (suggestionConfidence) {
      suggestionConfidence.textContent = '100% Resolved';
      suggestionConfidence.classList.remove('hidden');
    }

    // 6. Update Subsystem statuses
    updateSubsystemsState({
      gemini: 'Ready',
      rag: 'Idle'
    });
  }

  function handleSuggestionUpdate(data) {
    if (!data) return;

    // Check if customer issue has been resolved
    if (data.is_resolved === true || (data.has_issue === false && data.status === 'resolved')) {
      handleIssueResolved(data);
      return;
    }

    // Check if an issue has been identified and suggestions are present
    const hasIssue = data.has_issue !== false;
    const hasSuggestions = (Array.isArray(data.suggestions) && data.suggestions.length > 0) ||
      (typeof data.suggestion === 'string' && data.suggestion.trim().length > 0);

    if (!hasIssue || !hasSuggestions) {
      // Maintain awaiting-context / empty state when no customer issue has been identified
      if (suggestionsEmpty) suggestionsEmpty.classList.remove('hidden');
      if (suggestionContent) suggestionContent.classList.add('hidden');
      if (groundingBadge) {
        groundingBadge.className = 'grounding-pill empty';
        groundingBadge.textContent = 'Awaiting Context';
      }
      return;
    }

    // Hide empty placeholder
    if (suggestionsEmpty) {
      suggestionsEmpty.classList.add('hidden');
    }

    // Calculate & display synthesis latency if customer turn timestamp exists
    if (lastCustomerTurnTimestamp && suggestionLatency && suggestionLatencyVal) {
      const elapsedSec = ((performance.now() - lastCustomerTurnTimestamp) / 1000.0).toFixed(2);
      suggestionLatencyVal.textContent = `${elapsedSec}s`;
      suggestionLatency.classList.remove('hidden');
    }

    // Grounding Status Badge
    if (groundingBadge) {
      groundingBadge.className = 'grounding-pill';
      if (data.is_grounded === false || data.evidence_status === 'insufficient') {
        groundingBadge.textContent = 'Insufficient Documentation';
        groundingBadge.classList.add('ungrounded');
      } else if (data.evidence_status === 'partial') {
        groundingBadge.textContent = 'Partially Grounded';
        groundingBadge.classList.add('partial');
      } else {
        groundingBadge.textContent = 'Grounded in Knowledge Base';
        groundingBadge.classList.add('grounded');
      }
    }

    // Confidence badge
    if (suggestionConfidence) {
      if (typeof data.confidence === 'number' && data.is_grounded !== false && data.evidence_status !== 'insufficient') {
        const pct = Math.round(data.confidence <= 1.0 ? data.confidence * 100 : data.confidence);
        suggestionConfidence.textContent = `${pct}% Grounding Conf`;
        suggestionConfidence.classList.remove('hidden');
      } else {
        suggestionConfidence.textContent = '';
        suggestionConfidence.classList.add('hidden');
      }
    }

    // Ungrounded Warning Banner
    if (ungroundedAlert) {
      if (data.is_grounded === false || data.evidence_status === 'insufficient') {
        ungroundedAlert.classList.remove('hidden');
        const alertP = ungroundedAlert.querySelector('p');
        if (alertP) {
          alertP.textContent = data.suggestion || "The available documentation does not contain verified resolution steps for this specific issue. Escalate to senior technical engineering.";
        }
      } else {
        ungroundedAlert.classList.add('hidden');
      }
    }

    if (suggestionContent) {
      suggestionContent.classList.remove('hidden');
      parseAndRenderSuggestion(data, data.sources);
    }

    // Knowledge Base Citations
    renderSourceCitations(data.sources, data.suggestion_sources);

    // Update Subsystem statuses back to Ready
    updateSubsystemsState({
      gemini: 'Ready',
      rag: `Retrieved (${Array.isArray(data.sources) ? data.sources.length : 0} Chunks)`
    });

    // Reset processing state back to listening
    if (isRunning) {
      updateSystemStatus('running', true);
    }
  }

  // --- Dedicated Frontend Parser for Support Suggestions ---
  function parseSuggestionContent(rawText) {
    let text = (rawText || '').trim();
    if (!text) {
      return { likelyIssue: '', actions: [], safety: '', questions: [] };
    }

    let safety = '';
    let likelyIssue = '';
    let actions = [];
    let questions = [];

    // 0. Extract questions if any
    const lines = text.split(/[\r\n]+/).map(l => l.trim()).filter(Boolean);
    lines.forEach(l => {
      if ((l.endsWith('?') || l.includes('?')) && l.length > 10 && !l.toLowerCase().includes('likely')) {
        questions.push(l);
      }
    });

    // 1. Explicit Safety Header: "Safety Reminder: ...", "Safety: ...", "Safety Warning: ..."
    const safetyHeaderMatch = text.match(/(?:safety(?:\s+reminder|\s+warning|\s+precaution|\s+note)?\s*[:\-]\s*)([^\n\r]+)/i);
    if (safetyHeaderMatch) {
      safety = safetyHeaderMatch[1].trim();
      text = text.replace(safetyHeaderMatch[0], '').trim();
    }

    // 2. Identify numbered steps anywhere in text
    // Matches "1. ", "1) ", "(1) ", "Step 1: " at start of line OR preceded by punctuation/space
    const globalStepRegex = /(?:^|[\n\r]+|(?<=[.!?:\-;])\s+)(?:(?:\(?(\d+)\)?[.:]\s+)|\b(?:step\s+(\d+)[:.]\s+))/gi;
    const matches = [...text.matchAll(globalStepRegex)];

    if (matches.length > 0) {
      let preText = text.substring(0, matches[0].index).trim();
      // Clean trailing transition phrases from preText
      preText = preText.replace(/(?:to resolve(?: this issue| the fault)?|resolution steps|recommended actions|action steps|troubleshooting steps|follow these steps)\s*[:\-]?\s*$/i, '').trim();
      likelyIssue = preText;

      for (let i = 0; i < matches.length; i++) {
        const startPos = matches[i].index + matches[i][0].length;
        const endPos = (i + 1 < matches.length) ? matches[i + 1].index : text.length;
        let stepText = text.substring(startPos, endPos).trim();
        stepText = stepText.replace(/^[\s\-–—]+/, '').trim();
        if (stepText) {
          actions.push(stepText);
        }
      }
    } else {
      // Check for bullet points
      const bulletLines = lines.filter(l => /^[\-\*\•]\s+/.test(l));
      if (bulletLines.length > 0) {
        actions = bulletLines.map(l => l.replace(/^[\-\*\•]\s+/, '').trim());
        likelyIssue = lines.filter(l => !/^[\-\*\•]\s+/.test(l)).join(' ').trim();
      } else {
        actions = [text];
      }
    }

    // 3. Extract Safety from likelyIssue or actions if not yet found
    const safetyPatterns = [
      /(?:^|[\n\.\!\?]\s*)([^.\n!?]*(?:never bypass|lock-?out|tag-?out|safety interlock|disconnect power before servicing)[^.\n!?]*[\.\!\?]?)/i
    ];
    if (!safety) {
      for (const pat of safetyPatterns) {
        if (likelyIssue) {
          const m = likelyIssue.match(pat);
          if (m) {
            safety = m[1].trim();
            likelyIssue = likelyIssue.replace(m[0], '').trim();
            break;
          }
        }
      }
    }

    // Check if any action is purely safety advice
    if (!safety) {
      const safetyActionIdx = actions.findIndex(a => /\b(never bypass|lock-?out|tag-?out)\b/i.test(a));
      if (safetyActionIdx !== -1) {
        safety = actions[safetyActionIdx];
        actions.splice(safetyActionIdx, 1);
      }
    }

    // Clean trailing punctuation or separators from actions
    actions = actions.map(a => a.replace(/[\s,;]+$/, '').trim()).filter(Boolean);

    // Limit actions up to 4
    if (actions.length > 4) {
      actions = actions.slice(0, 4);
    }

    return { likelyIssue, actions, safety, questions };
  }

  // --- Render Support Suggestions: Exactly 4 Action Cards, Likely Issue, Safety & Questions ---
  function parseAndRenderSuggestion(dataOrText, sources) {
    let actions = [];
    let likelyIssue = '';
    let safety = '';
    let questions = [];
    let suggestionSources = [];

    if (dataOrText && typeof dataOrText === 'object' && !Array.isArray(dataOrText)) {
      likelyIssue = dataOrText.likely_issue || '';
      safety = dataOrText.safety || '';
      if (Array.isArray(dataOrText.suggestion_sources)) {
        suggestionSources = dataOrText.suggestion_sources;
      }
      if (Array.isArray(dataOrText.suggestions) && dataOrText.suggestions.length > 0) {
        actions = dataOrText.suggestions.map(s => String(s).trim()).filter(Boolean);
      } else if (typeof dataOrText.suggestion === 'string') {
        const parsed = parseSuggestionContent(dataOrText.suggestion);
        actions = parsed.actions;
        if (!likelyIssue) likelyIssue = parsed.likelyIssue;
        if (!safety) safety = parsed.safety;
        questions = parsed.questions;
      }
    } else {
      let rawText = '';
      if (typeof dataOrText === 'string') {
        rawText = dataOrText;
      } else if (Array.isArray(dataOrText)) {
        rawText = dataOrText.join('\n');
      }
      const parsed = parseSuggestionContent(rawText);
      actions = parsed.actions;
      likelyIssue = parsed.likelyIssue;
      safety = parsed.safety;
      questions = parsed.questions;
    }

    // Deduplicate actions while preserving order
    const seen = new Set();
    const uniqueActions = [];
    actions.forEach(act => {
      const norm = act.replace(/^[\d+.)\-\*]+\s*/, '').trim().toLowerCase();
      if (norm && !seen.has(norm)) {
        seen.add(norm);
        uniqueActions.push(act);
      }
    });

    // Enforce exactly 4 suggestion cards ONLY when sufficient supporting documentation exists
    const isInsufficient = dataOrText && (dataOrText.is_grounded === false || dataOrText.evidence_status === 'insufficient');
    const isPartial = dataOrText && dataOrText.evidence_status === 'partial';

    if (isInsufficient) {
      // Strict grounding: Never generate technical cards or generic fallbacks when documentation is insufficient
      uniqueActions.length = 0;
    } else if (isPartial) {
      // Partial documentation: preserve only explicitly supported actions, do NOT force 4 suggestions
    } else {
      // Sufficient documentation available: enforce exactly 4 distinct suggestions
      const fallbackProceduralSteps = [
        "Verify operating parameters: Confirm power supply, line voltages, and pneumatic pressure gauges match machine specifications.",
        "Inspect documented safety status: Ensure all emergency stop circuits and guard interlocks are in a safe, normal operating state.",
        "Check documentation & diagnostics: Review error history logs and cross-reference equipment schematics in the technical manual.",
        "Escalate to technical engineering: Contact an authorized factory service technician if the fault persists after initial checks."
      ];

      while (uniqueActions.length < 4 && uniqueActions.length > 0) {
        const candidate = fallbackProceduralSteps.find(step => {
          const normStep = step.toLowerCase().substring(0, 20);
          return !uniqueActions.some(existing => existing.toLowerCase().includes(normStep));
        }) || fallbackProceduralSteps[uniqueActions.length % fallbackProceduralSteps.length];
        uniqueActions.push(candidate);
      }

      if (uniqueActions.length > 4) {
        uniqueActions.length = 4;
      }
    }

    const sourceList = Array.isArray(sources) ? sources : [];

    // 1. Likely Issue Summary Header
    if (sectionDiagnosis && suggestionSummaryText) {
      if (likelyIssue) {
        suggestionSummaryText.textContent = likelyIssue;
        sectionDiagnosis.classList.remove('hidden');
      } else {
        sectionDiagnosis.classList.add('hidden');
      }
    }

    // 2. Recommended Actions Header & Cards
    if (sectionActionsHeader) {
      if (uniqueActions.length > 0) {
        sectionActionsHeader.classList.remove('hidden');
      } else {
        sectionActionsHeader.classList.add('hidden');
      }
    }

    if (suggestionStepsList) {
      suggestionStepsList.innerHTML = '';

      uniqueActions.forEach((act, idx) => {
        const badgeNum = String(idx + 1).padStart(2, '0');
        const card = document.createElement('div');
        card.className = 'suggestion-card';
        card.setAttribute('data-card-idx', String(idx + 1));

        // Determine specific document citations for this suggestion
        let rawCitations = [];
        if (Array.isArray(suggestionSources) && suggestionSources[idx]) {
          const rawItem = suggestionSources[idx];
          if (Array.isArray(rawItem)) {
            rawCitations = rawItem;
          } else if (typeof rawItem === 'string' || typeof rawItem === 'object') {
            rawCitations = [rawItem];
          }
        }

        // Normalize raw citations into array of objects with document_reference, page, title
        const citations = [];
        const seenRefs = new Set();

        rawCitations.forEach(c => {
          let ref = '';
          let page = -1;
          let title = '';
          if (typeof c === 'string') {
            ref = c.trim();
          } else if (c && typeof c === 'object') {
            ref = (c.document_reference || c.doc_id || '').trim();
            page = (c.page !== undefined && c.page !== null) ? c.page : -1;
            title = c.title || c.section || '';
          }

          if (ref && !seenRefs.has(ref)) {
            seenRefs.add(ref);
            const matchingSource = sourceList.find(s => s.document_reference === ref || s.doc_id === ref);
            if (matchingSource) {
              if (page === -1 && matchingSource.page !== undefined && matchingSource.page !== null) {
                page = matchingSource.page;
              }
              if (!title) {
                title = matchingSource.title || matchingSource.section || ref;
              }
            }
            citations.push({
              document_reference: ref,
              page: page,
              title: title || ref
            });
          }
        });

        // If no citations provided via suggestionSources, check if action text cites any known source
        if (citations.length === 0 && sourceList.length > 0) {
          for (const s of sourceList) {
            const ref = s.document_reference || s.doc_id;
            if (ref && act.toLowerCase().includes(ref.toLowerCase()) && !seenRefs.has(ref)) {
              seenRefs.add(ref);
              citations.push({
                document_reference: ref,
                page: (s.page !== undefined && s.page !== null) ? s.page : -1,
                title: s.title || s.section || ref
              });
            }
          }
        }

        // Store comma-separated document references on the card for accurate multi-reference matching
        if (citations.length > 0) {
          const docRefsStr = citations.map(c => c.document_reference).join(',');
          card.setAttribute('data-doc-refs', docRefsStr);
          card.setAttribute('data-doc-ref', citations[0].document_reference);
        }

        // Check if action has a title prefix before a colon or spaced dash
        const colonMatch = act.match(/^([^:\n\r]{3,45})[:]\s*(.+)/) || act.match(/^([^\n\r—–]{3,45})\s+[—–\-]\s+(.+)/);
        let contentHtml = '';
        if (colonMatch) {
          contentHtml = `
            <div class="suggestion-card-main">
              <span class="suggestion-number-badge">${badgeNum}</span>
              <div class="suggestion-card-body">
                <strong>${escapeHtml(colonMatch[1].trim())}:</strong> ${escapeHtml(colonMatch[2].trim())}
              </div>
            </div>
          `;
        } else {
          contentHtml = `
            <div class="suggestion-card-main">
              <span class="suggestion-number-badge">${badgeNum}</span>
              <div class="suggestion-card-body">${escapeHtml(act)}</div>
            </div>
          `;
        }

        let citationHtml = '';
        if (citations.length > 0) {
          const badgesHtml = citations.map(c => {
            const ref = c.document_reference;
            const pageText = (c.page !== undefined && c.page !== null && c.page !== -1) ? `PAGE ${c.page}` : '';
            return `
              <div class="suggestion-citation-chip" data-doc-ref="${escapeHtml(ref)}" data-card-idx="${idx + 1}" title="Click to inspect ${escapeHtml(ref)} in Knowledge Base">
                <svg class="chip-doc-icon" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                  <polyline points="14 2 14 8 20 8"></polyline>
                </svg>
                <span class="chip-doc-id">${escapeHtml(ref)}</span>
                ${pageText ? `<span class="chip-divider">|</span><span class="chip-page">${escapeHtml(pageText)}</span>` : ''}
              </div>
            `;
          }).join('');

          citationHtml = `
            <div class="suggestion-card-citation-row">
              ${badgesHtml}
            </div>
          `;
        } else {
          // Standard support protocol: clearly labeled, no fabricated document ID or page
          citationHtml = `
            <div class="suggestion-card-citation-row">
              <div class="suggestion-citation-chip protocol-chip" title="Standard procedural verification (not derived from machine manual)">
                <svg class="chip-doc-icon" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                </svg>
                <span class="chip-protocol-text">STANDARD SUPPORT PROTOCOL</span>
              </div>
            </div>
          `;
        }

        card.innerHTML = contentHtml + citationHtml;
        suggestionStepsList.appendChild(card);
      });
    }

    // 3. Safety Warning Banner
    if (suggestionSafetyBox && suggestionSafetyText) {
      if (safety) {
        suggestionSafetyText.textContent = safety;
        suggestionSafetyBox.classList.remove('hidden');
      } else {
        suggestionSafetyBox.classList.add('hidden');
      }
    }

    // 4. Render Dynamic Suggested Questions (if any present)
    renderSuggestedQuestions(questions);

    // 5. Bind Interactive Bidirectional Highlighting
    bindCitationHighlightEvents();
  }

  // --- Render Suggested Questions ---
  function renderSuggestedQuestions(questions) {
    if (!questionsList) return;

    if (questions && questions.length > 0) {
      questionsList.innerHTML = '';
      questions.forEach(q => {
        const li = document.createElement('li');
        li.className = 'question-item';
        li.innerHTML = `
          <span class="question-arrow">→</span>
          <span class="question-text-content">${escapeHtml(q)}</span>
          <span class="question-chevron">›</span>
        `;
        questionsList.appendChild(li);
      });

      if (questionsEmpty) questionsEmpty.classList.add('hidden');
      questionsList.classList.remove('hidden');
      if (questionsPanel) questionsPanel.classList.remove('hidden');
      if (questionsCount) {
        questionsCount.textContent = `${questions.length} Question${questions.length === 1 ? '' : 's'}`;
      }
    } else {
      questionsList.innerHTML = '';
      questionsList.classList.add('hidden');
      if (questionsEmpty) questionsEmpty.classList.remove('hidden');
      if (questionsPanel) questionsPanel.classList.add('hidden');
      if (questionsCount) {
        questionsCount.textContent = 'Agent Assistance';
      }
    }
  }

  // --- Render Knowledge Base Citations ---
  function renderSourceCitations(sources, suggestionSources) {
    if (!sourcesContainer || !sourcesList || !sourcesEmpty) return;

    const sourceList = Array.isArray(sources) ? sources : [];
    if (sourceList.length === 0) {
      sourcesContainer.classList.add('hidden');
      sourcesEmpty.classList.remove('hidden');
      if (sourcesCount) sourcesCount.textContent = '0 citations';
      return;
    }

    sourcesEmpty.classList.add('hidden');
    sourcesContainer.classList.remove('hidden');
    if (sourcesCount) {
      sourcesCount.textContent = `${sourceList.length} citation${sourceList.length === 1 ? '' : 's'}`;
    }

    sourcesList.innerHTML = '';
    sourceList.forEach(src => {
      const card = document.createElement('div');
      card.className = 'source-card';

      const docName = src.source_file || src.doc_id || 'Technical Manual';
      const docRef = src.document_reference || docName;
      card.setAttribute('data-doc-ref', docRef);

      const pageStr = (src.page !== undefined && src.page !== null && src.page !== -1) ? `PAGE ${src.page}` : '';
      const docTitle = src.title || src.section || 'Technical Documentation';
      const sourceFile = src.source_file || '';
      const rawExcerpt = src.excerpt || (src.text ? src.text.substring(0, 200) + '...' : '');
      const excerpt = rawExcerpt.replace(/\s+/g, ' ').trim();

      // Find which suggestions reference this document
      let supported = Array.isArray(src.supported_suggestions) ? [...src.supported_suggestions] : [];
      if (supported.length === 0 && Array.isArray(suggestionSources)) {
        suggestionSources.forEach((rawItem, sIdx) => {
          if (Array.isArray(rawItem)) {
            if (rawItem.some(c => (typeof c === 'string' ? c === docRef : (c && (c.document_reference === docRef || c.doc_id === docRef))))) {
              supported.push(sIdx + 1);
            }
          } else if (typeof rawItem === 'string' && rawItem === docRef) {
            supported.push(sIdx + 1);
          } else if (rawItem && typeof rawItem === 'object' && (rawItem.document_reference === docRef || rawItem.doc_id === docRef)) {
            supported.push(sIdx + 1);
          }
        });
      }
      supported = Array.from(new Set(supported)).sort((a, b) => a - b);

      let supportedBadge = '';
      if (supported.length > 0) {
        const numbers = supported.map(n => `#${String(n).padStart(2, '0')}`).join(', ');
        supportedBadge = `<span class="source-supported-badge" title="Referenced by recommendation ${escapeHtml(numbers)}">Supports Suggestions ${escapeHtml(numbers)}</span>`;
      } else {
        supportedBadge = `<span class="source-context-badge" title="Retrieved for contextual background">Retrieved Context</span>`;
      }

      card.setAttribute('data-supported-cards', supported.join(','));

      card.innerHTML = `
        <div class="source-card-header">
          <div class="source-ref-badge-group">
            <svg class="source-icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
              <polyline points="14 2 14 8 20 8"></polyline>
            </svg>
            <span class="source-ref-code">${escapeHtml(docRef)}</span>
            ${pageStr ? `<span class="source-page-divider">|</span><span class="source-page-badge">${escapeHtml(pageStr)}</span>` : ''}
          </div>
          ${supportedBadge}
        </div>
        <div class="source-card-title-row">
          <span class="source-doc-title" title="${escapeHtml(docTitle)}">${escapeHtml(docTitle)}</span>
          ${sourceFile ? `<span class="source-file-badge" title="${escapeHtml(sourceFile)}">${escapeHtml(sourceFile)}</span>` : ''}
        </div>
        ${excerpt ? `<div class="source-excerpt" title="Retrieved knowledge excerpt">"${escapeHtml(excerpt)}"</div>` : ''}
      `;
      sourcesList.appendChild(card);
    });

    bindCitationHighlightEvents();
  }

  // --- Bind Bidirectional Citation Highlighting ---
  function bindCitationHighlightEvents() {
    // 1. Suggestion citation chip interactions
    const chips = document.querySelectorAll('.suggestion-citation-chip[data-doc-ref]');
    chips.forEach(chip => {
      const docRef = chip.getAttribute('data-doc-ref');
      if (!docRef) return;

      chip.addEventListener('mouseenter', () => {
        document.querySelectorAll(`.source-card[data-doc-ref="${CSS.escape(docRef)}"]`).forEach(c => c.classList.add('highlight-connected'));
      });
      chip.addEventListener('mouseleave', () => {
        document.querySelectorAll(`.source-card[data-doc-ref="${CSS.escape(docRef)}"]`).forEach(c => c.classList.remove('highlight-connected'));
      });
      chip.addEventListener('click', (e) => {
        e.stopPropagation();
        const targetSource = document.querySelector(`.source-card[data-doc-ref="${CSS.escape(docRef)}"]`);
        if (targetSource) {
          targetSource.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          targetSource.classList.add('pulse-focus');
          setTimeout(() => targetSource.classList.remove('pulse-focus'), 1200);
        }
      });
    });

    // 2. Source card interactions (highlights every suggestion card supported by this document)
    const sourceCards = document.querySelectorAll('.source-card[data-doc-ref]');
    sourceCards.forEach(card => {
      const docRef = card.getAttribute('data-doc-ref');
      if (!docRef) return;

      card.addEventListener('mouseenter', () => {
        document.querySelectorAll('.suggestion-card').forEach(sc => {
          const refs = (sc.getAttribute('data-doc-refs') || sc.getAttribute('data-doc-ref') || '').split(',').map(r => r.trim());
          if (refs.includes(docRef) || sc.querySelector(`.suggestion-citation-chip[data-doc-ref="${CSS.escape(docRef)}"]`)) {
            sc.classList.add('highlight-connected');
          }
        });
      });
      card.addEventListener('mouseleave', () => {
        document.querySelectorAll('.suggestion-card.highlight-connected').forEach(sc => sc.classList.remove('highlight-connected'));
      });
      card.addEventListener('click', () => {
        const matchingCards = Array.from(document.querySelectorAll('.suggestion-card')).filter(sc => {
          const refs = (sc.getAttribute('data-doc-refs') || sc.getAttribute('data-doc-ref') || '').split(',').map(r => r.trim());
          return refs.includes(docRef) || sc.querySelector(`.suggestion-citation-chip[data-doc-ref="${CSS.escape(docRef)}"]`);
        });
        if (matchingCards.length > 0) {
          matchingCards[0].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          matchingCards.forEach(mc => {
            mc.classList.add('pulse-focus');
            setTimeout(() => mc.classList.remove('pulse-focus'), 1200);
          });
        }
      });
    });
  }

  // 6. System Status Event
  function handleSystemStatus(data) {
    const status = data.status || 'unknown';

    if (data.device && deviceLabel) {
      deviceLabel.textContent = data.device;
    }

    if (typeof data.is_running === 'boolean') {
      isRunning = data.is_running;
    } else if (status === 'started') {
      isRunning = true;
      resetDashboardState();
      startSessionTimer();
    } else if (status === 'stopped') {
      isRunning = false;
      stopSessionTimer();
    }

    updateSystemStatus(status, isRunning);
  }

  // 7. System Error Event
  function handleSystemError(data) {
    const stage = data.stage ? `[${data.stage.toUpperCase()}] ` : '';
    const msg = stage + (data.message || 'An unexpected pipeline error occurred.');
    showError(msg);
    updateSystemStatus('error');
  }

  // --- Command Dispatcher ---
  function sendCommand(cmd) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      showError('Cannot send command: WebSocket is disconnected.');
      return;
    }
    try {
      ws.send(JSON.stringify({ command: cmd }));
      console.log(`[Dashboard] Command sent: ${cmd}`);
    } catch (err) {
      console.error('[Dashboard] Error sending command:', err);
      showError('Failed to send command: ' + err.message);
    }
  }

  if (startBtn) {
    startBtn.addEventListener('click', function () {
      resetDashboardState();
      startSessionTimer();
      sendCommand('start');
    });
  }

  if (stopBtn) {
    stopBtn.addEventListener('click', function () {
      stopSessionTimer();
      sendCommand('stop');
    });
  }

  // --- Safe Logout Handler ---
  async function handleLogout() {
    console.log('[Dashboard] Initiating safe logout...');
    isIntentionallyClosed = true;

    // 1. Stop active audio capture / listening session safely
    if (isRunning) {
      try {
        sendCommand('stop');
      } catch (err) {
        console.warn('[Dashboard] Error sending stop command during logout:', err);
      }
      stopSessionTimer();
      isRunning = false;
    }

    // 2. Safely close WebSocket
    if (ws) {
      try {
        ws.close(1000, 'Agent logged out');
      } catch (err) {
        console.warn('[Dashboard] Error closing WebSocket on logout:', err);
      }
      ws = null;
    }

    // 3. Invalidate server-side session and cookie
    try {
      await fetch('/api/auth/logout', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        }
      });
    } catch (err) {
      console.warn('[Dashboard] Logout request error:', err);
    }

    // 4. Redirect to login page
    window.location.href = '/login';
  }

  if (logoutBtn) {
    logoutBtn.addEventListener('click', handleLogout);
  }

  // --- Authenticated Identity Initialization ---
  async function initAgentIdentity() {
    try {
      const res = await fetch('/api/auth/me');
      if (!res.ok) {
        window.location.href = '/login';
        return false;
      }
      const data = await res.json();
      if (data.authenticated && data.agent_name) {
        if (agentNameEl) {
          agentNameEl.textContent = data.agent_name;
        }
        return true;
      } else {
        window.location.href = '/login';
        return false;
      }
    } catch (err) {
      console.warn('[Dashboard] Auth check failed:', err);
      window.location.href = '/login';
      return false;
    }
  }

  // --- Theme Management (Light / Dark Switching) ---
  function getEffectiveTheme() {
    return document.documentElement.getAttribute('data-theme') || 'dark';
  }

  function updateThemeButtonA11y(theme) {
    if (!themeToggleBtn) return;
    if (theme === 'light') {
      themeToggleBtn.setAttribute('aria-label', 'Switch to dark theme');
      themeToggleBtn.setAttribute('title', 'Switch to dark theme');
    } else {
      themeToggleBtn.setAttribute('aria-label', 'Switch to light theme');
      themeToggleBtn.setAttribute('title', 'Switch to light theme');
    }
  }

  function applyTheme(targetTheme) {
    const theme = targetTheme === 'light' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('theme', theme);
    } catch (e) {
      console.warn('[Dashboard] Could not persist theme preference to localStorage:', e);
    }
    updateThemeButtonA11y(theme);
  }

  function toggleTheme() {
    const currentTheme = getEffectiveTheme();
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    applyTheme(newTheme);
  }

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', toggleTheme);
    // Sync accessibility label/tooltip with current theme state
    updateThemeButtonA11y(getEffectiveTheme());
  }

  // --- Helper: HTML Sanitization ---
  function escapeHtml(str) {
    if (typeof str !== 'string') return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // --- Initial Launch ---
  document.addEventListener('DOMContentLoaded', async function () {
    const isAuthed = await initAgentIdentity();
    if (isAuthed) {
      connectWebSocket();
    }
  });

  // Expose for testing & verification
  window.handleSuggestionUpdate = handleSuggestionUpdate;
})();

(function () {
  function createUid() {
    if (window.crypto && window.crypto.randomUUID) return window.crypto.randomUUID();
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  window.RTSessionState = {
    createSession() {
      return {
        sessionId: createUid(),
        startedAt: new Date().toISOString(),
        status: "active",
      };
    },
    buildConfig(base, session, resume) {
      return {
        ...base,
        session_id: session.sessionId,
        session_started_at: session.startedAt,
        resume_session: Boolean(resume),
      };
    },
  };
})();

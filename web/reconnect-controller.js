(function () {
  class ReconnectController {
    constructor(options) {
      this.delays = options.delays || [1000, 2000, 4000, 8000, 15000];
      this.onStatus = options.onStatus || function () {};
      this.onReconnect = options.onReconnect || function () {};
      this.onFailed = options.onFailed || function () {};
      this.timer = null;
      this.attempt = 0;
      this.active = false;
    }

    reset() {
      if (this.timer) window.clearTimeout(this.timer);
      this.timer = null;
      this.attempt = 0;
      this.active = false;
    }

    schedule() {
      if (this.active) return;
      this.active = true;
      this._next();
    }

    _next() {
      if (this.attempt >= this.delays.length) {
        this.reset();
        this.onFailed();
        return;
      }
      const delay = this.delays[this.attempt];
      this.attempt += 1;
      this.onStatus(this.attempt, this.delays.length, delay);
      this.timer = window.setTimeout(() => {
        this.timer = null;
        Promise.resolve(this.onReconnect(this.attempt, this.delays.length))
          .catch(() => {})
          .finally(() => {
            if (this.active) this._next();
          });
      }, delay);
    }

    markConnected() {
      this.reset();
    }
  }

  window.RTReconnectController = ReconnectController;
})();

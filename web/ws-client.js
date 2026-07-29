(function () {
  window.RTWsClient = {
    open(url, handlers) {
      const ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";
      ws.addEventListener("open", handlers.open);
      ws.addEventListener("message", handlers.message);
      ws.addEventListener("error", handlers.error);
      ws.addEventListener("close", handlers.close);
      return ws;
    },
  };
})();

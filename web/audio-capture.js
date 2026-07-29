(function () {
  window.RTAudioCapture = {
    async requestMicrophoneStream() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("当前页面无法访问麦克风。请使用 HTTPS，或在 client 本机通过 http://localhost 打开页面。");
      }
      return navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
      });
    },
    stopTracks(stream) {
      if (stream) stream.getTracks().forEach((track) => track.stop());
    },
  };
})();

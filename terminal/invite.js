/* 传统版主页右下角的「终端版」提示卡片。
 *
 * 这段代码必须放在独立文件里，不能内联进 _includes/footer/terminal-invite.html：
 * Jekyll 的 compress_html 会把整个页面压成一行，内联脚本里的 // 单行注释
 * 会把它之后的所有代码一起注释掉，脚本直接失效。外部 js 不经过 HTML 压缩。
 */
(function () {
  var KEY = "lyz.invite.dismissed";
  var HIDE_DAYS = 30;
  var DELAY = 1800;

  function init() {
    var el = document.getElementById("lyz-invite");
    if (!el) return;
    /* 已经在终端版里就不必再邀请 */
    if (location.pathname.indexOf("/terminal") === 0) return;
    try {
      var until = parseInt(localStorage.getItem(KEY) || "0", 10);
      if (until && Date.now() < until) return;
    } catch (e) {}

    function remember() {
      try { localStorage.setItem(KEY, String(Date.now() + HIDE_DAYS * 864e5)); } catch (e) {}
    }
    function dismiss() {
      el.classList.remove("lyz-show");
      setTimeout(function () { el.hidden = true; }, 320);
      remember();
    }

    var x = el.querySelector(".lyz-invite-x");
    var later = el.querySelector(".lyz-invite-later");
    var go = el.querySelector(".lyz-invite-go");
    if (x) x.addEventListener("click", dismiss);
    if (later) later.addEventListener("click", dismiss);
    if (go) go.addEventListener("click", remember);

    setTimeout(function () {
      el.hidden = false;
      requestAnimationFrame(function () { el.classList.add("lyz-show"); });
    }, DELAY);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

/**
 * WILD-ID — Motion & Interaction System
 *
 * Requirements:
 * 1. Transform and opacity only.
 * 2. Fully supports prefers-reduced-motion (plain 150ms fade, no wordmark letter animation).
 * 3. Tab switching restarts @keyframes on every activation (300ms, translateY(14px)).
 * 4. Sliding underline (280ms ease) repositioned on resize and after fonts load.
 * 5. WILD-ID wordmark letter-by-letter reveal on first page load only (<800ms total).
 * 6. Section headings inside active tab fade up with short stagger (capped at 4 elements).
 * 7. Honesty notes NEVER animate, fade, or delay (immediate, readable display).
 * 8. Webcam safeguard depends on active MediaStream, not mere video presence.
 * 9. Real-outcome action feedback on success/failure, Delete confirmation attention.
 * 10. Zero external network requests or third-party dependencies.
 */
(function () {
  "use strict";

  const isReducedMotion = () =>
    Boolean(
      document.body.classList.contains("wid-reduced-motion") ||
        window.__wid_force_reduced_motion ||
        (window.matchMedia &&
          window.matchMedia("(prefers-reduced-motion: reduce)").matches)
    );

  /**
   * 1. Wordmark Letter-by-Letter Reveal (First Page Load Only)
   */
  function initWordmarkReveal() {
    const wordmark = document.querySelector(".wid-wordmark");
    if (!wordmark) {
      setTimeout(initWordmarkReveal, 60);
      return;
    }

    if (wordmark.__wid_revealed) return;

    // Check if already revealed in this browser session
    const alreadyRevealed =
      window.__wid_session_wordmark ||
      sessionStorage.getItem("wid_wordmark_revealed") === "true";

    const rawText = (wordmark.textContent || "").trim() || "WILD-ID";
    wordmark.setAttribute("aria-label", rawText);

    if (alreadyRevealed || isReducedMotion()) {
      wordmark.__wid_revealed = true;
      return;
    }

    wordmark.__wid_revealed = true;
    window.__wid_session_wordmark = true;
    try {
      sessionStorage.setItem("wid_wordmark_revealed", "true");
    } catch (_) {}

    wordmark.innerHTML = "";
    const letters = Array.from(rawText);
    letters.forEach((char, idx) => {
      const span = document.createElement("span");
      span.className = "wid-wordmark-char";
      span.setAttribute("aria-hidden", "true");
      span.textContent = char;
      // Stagger: 35ms per character (total for 7 chars: 6 * 35 = 210ms + 220ms anim = 430ms < 800ms)
      span.style.animationDelay = `${idx * 35}ms`;
      wordmark.appendChild(span);
    });
  }

  /**
   * 2. Visible Tab Switching & Sliding Underline
   */
  function initTabSystem() {
    const tabNav =
      document.querySelector(".tab-nav") ||
      document.querySelector('[role="tablist"]');

    if (!tabNav) {
      setTimeout(initTabSystem, 100);
      return;
    }

    // ── Create or get sliding underline (not a pill) ──
    let slider = tabNav.querySelector(".wid-tab-slider");
    if (!slider) {
      slider = document.createElement("span");
      slider.className = "wid-tab-slider";
      slider.setAttribute("aria-hidden", "true");
      tabNav.appendChild(slider);
    }

    function repositionSlider() {
      const activeBtn =
        tabNav.querySelector("button.selected") ||
        tabNav.querySelector('[role="tab"][aria-selected="true"]') ||
        tabNav.querySelector("button");

      if (!activeBtn) {
        slider.style.opacity = "0";
        return;
      }

      const navRect = tabNav.getBoundingClientRect();
      const btnRect = activeBtn.getBoundingClientRect();
      const left = btnRect.left - navRect.left + tabNav.scrollLeft;
      const width = btnRect.width;

      if (isReducedMotion()) {
        slider.style.transition = "opacity 150ms ease";
        slider.style.transform = `translateX(${left}px)`;
      } else {
        slider.style.transition =
          "transform 280ms cubic-bezier(0.2, 0, 0.2, 1), width 280ms cubic-bezier(0.2, 0, 0.2, 1), opacity 150ms ease";
        slider.style.transform = `translateX(${left}px)`;
      }

      slider.style.width = width + "px";
      slider.style.opacity = "1";
    }

    // ── Animate section headings inside active tab panel ──
    function animateHeadingsInPanel(panel) {
      if (!panel) return;

      const candidates = Array.from(
        panel.querySelectorAll("h4, .prose h4, p, .gr-row, .gr-button")
      );

      // Strict filter: EXCLUDE any honesty notes, alerts, warnings, or result metrics
      const eligible = candidates.filter((el) => {
        if (
          el.closest(".wid-note") ||
          el.closest(".wid-threshold-note") ||
          el.closest(".wid-summary-detail") ||
          el.closest(".wid-destructive-zone") ||
          el.closest(".wid-results-table") ||
          el.closest("table")
        ) {
          return false;
        }
        if (
          el.classList.contains("wid-note") ||
          el.classList.contains("wid-threshold-note") ||
          el.classList.contains("wid-summary-detail")
        ) {
          return false;
        }
        if (!el.textContent.trim()) return false;
        return true;
      });

      // Cap at 4 elements max
      const toAnimate = eligible.slice(0, 4);
      toAnimate.forEach((el, idx) => {
        el.classList.remove(
          "wid-heading-animate",
          "wid-stagger-1",
          "wid-stagger-2",
          "wid-stagger-3",
          "wid-stagger-4"
        );
        void el.offsetWidth; // Force reflow
        el.classList.add("wid-heading-animate", `wid-stagger-${idx + 1}`);
      });
    }

    // ── Panel activation trigger ──
    let activePanelElement = null;

    function activatePanel(panel) {
      if (!panel || panel === activePanelElement) return;
      activePanelElement = panel;

      // Restart tab panel animation on every activation
      panel.classList.remove("wid-tab-entering");
      void panel.offsetWidth; // Force reflow
      panel.classList.add("wid-tab-entering");

      // Trigger headings fade-up
      animateHeadingsInPanel(panel);
    }

    function checkActivePanel() {
      const panels = Array.from(
        document.querySelectorAll('.tabitem, [role="tabpanel"]')
      );
      for (const p of panels) {
        const style = window.getComputedStyle(p);
        if (style.display !== "none" && p.offsetWidth > 0) {
          activatePanel(p);
          break;
        }
      }
    }

    // ── Listeners ──
    // 1. Tab buttons click
    tabNav.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", () => {
        setTimeout(() => {
          checkActivePanel();
          repositionSlider();
        }, 20);
        setTimeout(() => {
          checkActivePanel();
          repositionSlider();
        }, 80);
      });
    });

    // 2. Observe tab navigation mutations (aria-selected, class changes)
    const navObserver = new MutationObserver(() => {
      repositionSlider();
      checkActivePanel();
    });
    navObserver.observe(tabNav, {
      attributes: true,
      subtree: true,
      attributeFilter: ["class", "aria-selected"],
    });

    // 3. Observe panel display changes
    const panelObserver = new MutationObserver(() => {
      checkActivePanel();
    });
    document
      .querySelectorAll('.tabitem, [role="tabpanel"]')
      .forEach((panel) => {
        panelObserver.observe(panel, {
          attributes: true,
          attributeFilter: ["style", "class", "aria-hidden"],
        });
      });

    // 4. Window resize
    window.addEventListener("resize", repositionSlider);

    // 5. Document fonts ready (handles font load shifts)
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => {
        repositionSlider();
      });
    }

    // Initial triggers
    setTimeout(() => {
      repositionSlider();
      checkActivePanel();
    }, 50);
    setTimeout(() => {
      repositionSlider();
      checkActivePanel();
    }, 250);
  }

  /**
   * 3. Inspection State Safeguard
   * Freeze hover zoom and tilt whenever an image is being inspected in a preview modal.
   */
  function initInspectionSafeguard() {
    function checkInspection() {
      const isModal =
        document.querySelector(".modal") ||
        document.querySelector(".preview") ||
        document.querySelector(".fullscreen") ||
        document.querySelector('[role="dialog"]') ||
        document.querySelector(".gallery-item.selected");

      if (isModal) {
        document.body.setAttribute("data-inspecting", "true");
      } else {
        document.body.removeAttribute("data-inspecting");
      }
    }

    document.addEventListener("click", () => setTimeout(checkInspection, 60));
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") setTimeout(checkInspection, 60);
    });
  }

  /**
   * 4. Webcam Safeguard
   * Checks for an active MediaStream, NOT merely the existence of a video element.
   * Animations resume immediately when camera stops or a photo is captured.
   */
  function initWebcamSafeguard() {
    let throttledSafetyTimer = null;

    function isStreamActuallyLive(video) {
      if (!video) return false;
      const stream = video.srcObject;
      if (!stream || !stream.active) return false;
      if (video.paused || video.ended) return false;

      const tracks = stream.getVideoTracks ? stream.getVideoTracks() : [];
      return tracks.some((t) => t.readyState === "live" && t.enabled);
    }

    function updateWebcamState() {
      const videos = document.querySelectorAll("video");
      let active = false;

      for (const v of videos) {
        if (isStreamActuallyLive(v)) {
          active = true;
          break;
        }
      }

      if (active) {
        document.body.setAttribute("data-webcam-active", "true");
        if (!throttledSafetyTimer) {
          throttledSafetyTimer = setInterval(updateWebcamState, 2500);
        }
      } else {
        document.body.removeAttribute("data-webcam-active");
        if (throttledSafetyTimer) {
          clearInterval(throttledSafetyTimer);
          throttledSafetyTimer = null;
        }
      }
    }

    function bindVideoEvents(v) {
      if (!v || v.__wid_bound) return;
      v.__wid_bound = true;

      v.addEventListener("play", updateWebcamState);
      v.addEventListener("playing", updateWebcamState);
      v.addEventListener("pause", updateWebcamState);
      v.addEventListener("ended", updateWebcamState);
      v.addEventListener("emptied", updateWebcamState);
      v.addEventListener("suspend", updateWebcamState);

      if (v.srcObject && v.srcObject.getTracks) {
        v.srcObject.getTracks().forEach((track) => {
          track.addEventListener("ended", updateWebcamState);
          track.addEventListener("mute", updateWebcamState);
          track.addEventListener("unmute", updateWebcamState);
        });
      }
    }

    const videoObserver = new MutationObserver(() => {
      document.querySelectorAll("video").forEach(bindVideoEvents);
      updateWebcamState();
    });

    videoObserver.observe(document.body, { childList: true, subtree: true });
    document.querySelectorAll("video").forEach(bindVideoEvents);

    document.addEventListener("click", () => {
      setTimeout(() => {
        document.querySelectorAll("video").forEach(bindVideoEvents);
        updateWebcamState();
      }, 100);
    });

    updateWebcamState();
  }

  /**
   * 5. Real-Outcome Action Feedback
   * Confirmation pulse (green) only after real success is rendered in output box.
   * On error or aborted actions, triggers distinct amber/red warning state.
   */
  function initActionFeedback() {
    const actionConfigs = [
      {
        btnId: "wid-approve-btn",
        outId: "wid-review-status",
        successTerms: ["approved"],
        errorTerms: ["not found", "invalid", "error", "please enter"],
      },
      {
        btnId: "wid-reject-btn",
        outId: "wid-review-status",
        successTerms: ["rejected and moved to trash", "rejected"],
        errorTerms: ["not found", "invalid", "error", "please enter"],
      },
      {
        btnId: "wid-delete-btn",
        outId: "wid-delete-output",
        confirmBoxId: "wid-delete-confirm",
        successTerms: ["successfully removed"],
        errorTerms: [
          "deletion aborted",
          "explicit confirmation required",
          "not found",
          "invalid",
          "error",
          "please enter",
        ],
      },
      {
        btnId: "wid-submit-btn",
        outId: "wid-submit-status",
        successTerms: ["submitted to pending queue"],
        errorTerms: ["failed", "invalid", "please provide", "please upload"],
      },
    ];

    actionConfigs.forEach((config) => {
      const btn = document.getElementById(config.btnId);
      if (!btn) return;

      btn.addEventListener("click", () => {
        const outContainer = document.getElementById(config.outId);
        if (!outContainer) return;

        const textarea =
          outContainer.querySelector("textarea") ||
          outContainer.querySelector("input") ||
          outContainer;
        const initialVal = textarea.value || textarea.innerText || "";

        outContainer.classList.remove(
          "wid-confirmed-feedback",
          "wid-error-feedback"
        );
        if (config.confirmBoxId) {
          const cBox = document.getElementById(config.confirmBoxId);
          if (cBox) cBox.classList.remove("wid-confirm-needed");
        }

        let attempts = 0;
        const maxAttempts = 50;
        const pollResult = setInterval(() => {
          attempts++;
          const currentVal = textarea.value || textarea.innerText || "";

          if (currentVal && currentVal !== initialVal) {
            clearInterval(pollResult);
            const lower = currentVal.toLowerCase();

            const isSuccess = config.successTerms.some((t) => lower.includes(t));
            const isError = config.errorTerms.some((t) => lower.includes(t));

            if (isSuccess && !isError) {
              outContainer.classList.remove("wid-error-feedback");
              void outContainer.offsetWidth;
              outContainer.classList.add("wid-confirmed-feedback");
              setTimeout(() => {
                outContainer.classList.remove("wid-confirmed-feedback");
              }, 2200);
            } else if (isError) {
              outContainer.classList.remove("wid-confirmed-feedback");
              void outContainer.offsetWidth;
              outContainer.classList.add("wid-error-feedback");

              if (
                config.confirmBoxId &&
                lower.includes("confirmation required")
              ) {
                const cBox = document.getElementById(config.confirmBoxId);
                if (cBox) {
                  cBox.classList.add("wid-confirm-needed");
                  setTimeout(
                    () => cBox.classList.remove("wid-confirm-needed"),
                    2600
                  );
                }
              }

              setTimeout(() => {
                outContainer.classList.remove("wid-error-feedback");
              }, 2600);
            }
          }

          if (attempts >= maxAttempts) {
            clearInterval(pollResult);
          }
        }, 100);
      });
    });
  }

  /**
   * 6. Left Sidebar Navigation System
   * - Controls page navigation across all 9 pages.
   * - Positions sliding vertical edge bar indicator (no pills!).
   * - Animates page transition (280-320ms @keyframes widPageEnter, translateY(14px) and opacity).
   * - Synchronizes with Gradio's internal tab switching.
   * - Handles mobile drawer toggling and hamburger button.
   * - Fully respects prefers-reduced-motion (150ms opacity only, transform: none !important).
   */
  function initSidebarNavigation() {
    const sidebar = document.getElementById("wid-sidebar");
    const navItems = document.querySelectorAll(".wid-nav-item");
    const indicator = document.querySelector(".wid-nav-indicator");
    const mobileTitle = document.getElementById("wid-mobile-page-title");
    const backdrop = document.getElementById("wid-drawer-backdrop");

    if (!navItems.length) {
      setTimeout(initSidebarNavigation, 60);
      return;
    }

    function repositionNavIndicator(activeItem) {
      if (!indicator || !activeItem) return;
      const navList = activeItem.closest(".wid-nav-list");
      if (!navList) return;

      const top = activeItem.offsetTop;
      const height = activeItem.offsetHeight;

      if (isReducedMotion()) {
        indicator.style.transition = "opacity 150ms ease";
        indicator.style.transform = "none";
        indicator.style.top = top + "px";
      } else {
        indicator.style.transition =
          "transform var(--wid-duration-slider) var(--wid-ease), height var(--wid-duration-fast) ease, opacity var(--wid-duration-fast) ease";
        indicator.style.top = "0px";
        indicator.style.transform = `translateY(${top}px)`;
      }

      indicator.style.height = height + "px";
      indicator.style.opacity = "1";
    }

    function switchPage(index, skipGradioClick) {
      const targetItem = document.querySelector(`.wid-nav-item[data-tab-index="${index}"]`);
      if (!targetItem) return;

      // Update active classes on nav items
      navItems.forEach((btn) => btn.classList.remove("active"));
      targetItem.classList.add("active");

      // Position sliding vertical indicator
      repositionNavIndicator(targetItem);

      // Update mobile header title
      if (mobileTitle) {
        const label = targetItem.querySelector(".wid-nav-label");
        if (label) mobileTitle.textContent = label.textContent.trim();
      }

      // Close mobile drawer if open
      if (sidebar) sidebar.classList.remove("wid-drawer-open");
      if (backdrop) backdrop.classList.remove("wid-active");

      // Click Gradio's internal tab button
      if (!skipGradioClick) {
        const gradioTabBtns = document.querySelectorAll(
          "#wid-main-tabs > .tab-nav > button, #wid-main-tabs [role='tablist'] > button, #wid-main-tabs .tab-nav button"
        );
        if (gradioTabBtns[index]) {
          gradioTabBtns[index].click();
        }
      }

      // Trigger entrance animation on the active tab panel with timer fallback.
      // The class is only added here (on real page switch), never from a
      // MutationObserver or any other generic hook.
      setTimeout(() => {
        const panels = document.querySelectorAll(
          "#wid-main-tabs > .tabitem, #wid-main-tabs [role='tabpanel']"
        );
        panels.forEach((p) => {
          const style = window.getComputedStyle(p);
          if (style.display !== "none" && p.offsetWidth > 0) {
            // Remove first to allow restart
            p.classList.remove("wid-page-entering", "wid-tab-entering");
            void p.offsetWidth; // force reflow
            p.classList.add("wid-page-entering");

            // Timer fallback: duration (ms) from CSS custom prop + 100 ms safety
            const durStr = window.getComputedStyle(p)
              .getPropertyValue("--wid-duration-page").trim();
            const durMs = durStr.endsWith("ms")
              ? parseFloat(durStr)
              : parseFloat(durStr) * 1000;
            const safeMs = (isNaN(durMs) ? 300 : durMs) + 100;

            const timer = setTimeout(() => {
              p.classList.remove("wid-page-entering");
            }, safeMs);

            p.addEventListener("animationend", function onEnd() {
              clearTimeout(timer);
              p.classList.remove("wid-page-entering");
              p.removeEventListener("animationend", onEnd);
            });
          }
        });
      }, 30);

      // Sync URL hash and body dataset
      const hash = "#page-" + index;
      if (window.location.hash !== hash) {
        history.replaceState(null, "", hash);
      }
      document.body.dataset.page = index;
    }

    // Attach click events to nav items
    navItems.forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        const idx = parseInt(btn.getAttribute("data-tab-index") || "0", 10);
        switchPage(idx, false);
      });
    });

    // Mobile Hamburger button
    const hamburger = document.getElementById("wid-hamburger-btn");
    if (hamburger && sidebar && backdrop) {
      hamburger.addEventListener("click", () => {
        const isOpen = sidebar.classList.toggle("wid-drawer-open");
        backdrop.classList.toggle("wid-active", isOpen);
      });

      backdrop.addEventListener("click", () => {
        sidebar.classList.remove("wid-drawer-open");
        backdrop.classList.remove("wid-active");
      });
    }

    // Window resize: reposition indicator
    window.addEventListener("resize", () => {
      const active = document.querySelector(".wid-nav-item.active");
      if (active) repositionNavIndicator(active);
    });

    // Initial positioning
    const initialActive = document.querySelector(".wid-nav-item.active") || navItems[0];
    if (initialActive) {
      setTimeout(() => repositionNavIndicator(initialActive), 80);
      setTimeout(() => repositionNavIndicator(initialActive), 300);
    }

    // Expose helpers globally for testing/automation
    window.__wid_switch_page = switchPage;
  }

  // Helpers for testing and automation
  window.__wid_toggle_reduced_motion = function (force) {
    if (typeof force === "boolean") {
      window.__wid_force_reduced_motion = force;
    } else {
      window.__wid_force_reduced_motion = !window.__wid_force_reduced_motion;
    }
    if (window.__wid_force_reduced_motion) {
      document.body.classList.add("wid-reduced-motion");
    } else {
      document.body.classList.remove("wid-reduced-motion");
    }
    return window.__wid_force_reduced_motion;
  };

  /**
   * 7. Dynamic status line — reads real runtime state.
   * Never shows hard-coded text. "System ready" only when both model and DB
   * are actually loaded. Populates #wid-status-text in the sidebar.
   */
  function updateStatus() {
    const statusEl = document.getElementById("wid-status-text");
    if (!statusEl) return;

    // Python objects are NOT accessible from JS. The status is provided
    // by the server via the #wid-header HTML element (updated by Gradio callbacks).
    // We mirror it into the sidebar's status span.
    const headerStatusEl = document.querySelector("#wid-header .wid-status");
    if (headerStatusEl) {
      // Collapse the HTML to a short plain-text summary for the sidebar badge.
      const raw = headerStatusEl.textContent || "";
      const loaded = raw.includes("Database loaded") && raw.includes("Model loaded");
      const dot = document.querySelector(".wid-status-dot");
      if (dot) {
        dot.style.background = loaded
          ? "var(--wid-green-text)"
          : "var(--wid-amber)";
      }
      // Compact format: show key facts only
      statusEl.textContent = loaded ? "System ready" : "Initialising…";
      // Store full text on the element for smoke-test assertions
      statusEl.dataset.statusFull = raw.trim();
    } else {
      // Header not yet rendered; retry
      setTimeout(updateStatus, 500);
    }
  }

  // Initialize once DOM is ready
  function initAll() {
    initWordmarkReveal();
    initTabSystem();
    initSidebarNavigation();
    initInspectionSafeguard();
    initWebcamSafeguard();
    initActionFeedback();
    // Restore page from URL hash on load
    const m = window.location.hash.match(/#page-(\d+)/);
    if (m) {
      const idx = parseInt(m[1], 10);
      if (!isNaN(idx) && window.__wid_switch_page) {
        setTimeout(() => window.__wid_switch_page(idx, false), 200);
      }
    }
    // Update status after a short wait for Gradio to finish rendering
    setTimeout(updateStatus, 800);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAll);
  } else {
    initAll();
  }
})();

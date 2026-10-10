/**
 * static/login.js
 *
 * Support Agent Authentication Logic for AI-Powered Customer Support Assistant POC.
 * Supports:
 * - Sign In & Account Creation tab/view toggling
 * - Show/Hide password toggle on all password inputs
 * - Full client-side & server-side validation
 * - Theme switching with localStorage persistence
 * - POC demo account shortcuts
 * - Automatic redirect to /dashboard on authenticated session
 */

(function () {
  "use strict";

  // --- Element Selectors ---
  // Tabs & Views
  const tabLogin = document.getElementById("tab-login");
  const tabRegister = document.getElementById("tab-register");
  const loginSection = document.getElementById("login-section");
  const registerSection = document.getElementById("register-section");
  const cardTitle = document.getElementById("auth-card-title");
  const cardSubtitle = document.getElementById("auth-card-subtitle");
  const linkToRegister = document.getElementById("link-to-register");
  const linkToLogin = document.getElementById("link-to-login");

  // Feedback Banners
  const errorBox = document.getElementById("auth-error");
  const errorText = document.getElementById("auth-error-text");
  const successBox = document.getElementById("auth-success");
  const successText = document.getElementById("auth-success-text");

  // Login Form Elements
  const loginForm = document.getElementById("login-form");
  const loginEmailInput = document.getElementById("login-email-input");
  const loginPasswordInput = document.getElementById("login-password-input");
  const loginSubmitBtn = document.getElementById("login-submit-btn");
  const loginBtnText = document.getElementById("login-btn-text");
  const loginBtnSpinner = document.getElementById("login-btn-spinner");

  // Register Form Elements
  const registerForm = document.getElementById("register-form");
  const regNameInput = document.getElementById("reg-name-input");
  const regEmailInput = document.getElementById("reg-email-input");
  const regPasswordInput = document.getElementById("reg-password-input");
  const regConfirmPasswordInput = document.getElementById("reg-confirm-password-input");
  const registerSubmitBtn = document.getElementById("register-submit-btn");
  const regBtnText = document.getElementById("reg-btn-text");
  const regBtnSpinner = document.getElementById("reg-btn-spinner");

  // Theme & Demo Elements
  const themeToggleBtn = document.getElementById("theme-toggle-btn");
  const demoBoomikaBtn = document.getElementById("demo-boomika-btn");
  const demoAlexBtn = document.getElementById("demo-alex-btn");

  const EMAIL_REGEX = /^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$/;

  // --------------------------------------------------------------------------
  // 1. Theme Management (Matches Dashboard)
  // --------------------------------------------------------------------------
  function applyTheme(theme) {
    if (theme === "light") {
      document.documentElement.setAttribute("data-theme", "light");
      if (themeToggleBtn) {
        themeToggleBtn.setAttribute("aria-label", "Switch to dark theme");
        themeToggleBtn.setAttribute("title", "Switch to dark theme");
      }
    } else {
      document.documentElement.setAttribute("data-theme", "dark");
      if (themeToggleBtn) {
        themeToggleBtn.setAttribute("aria-label", "Switch to light theme");
        themeToggleBtn.setAttribute("title", "Switch to light theme");
      }
    }
    try {
      localStorage.setItem("theme", theme);
    } catch (e) {}
  }

  function initTheme() {
    let savedTheme = "dark";
    try {
      savedTheme = localStorage.getItem("theme") || "dark";
    } catch (e) {}
    applyTheme(savedTheme);

    if (themeToggleBtn) {
      themeToggleBtn.addEventListener("click", () => {
        const current = document.documentElement.getAttribute("data-theme") || "dark";
        const next = current === "light" ? "dark" : "light";
        applyTheme(next);
      });
    }
  }

  // --------------------------------------------------------------------------
  // 2. Feedback Messages
  // --------------------------------------------------------------------------
  function showError(msg) {
    if (errorText) errorText.textContent = msg;
    if (errorBox) errorBox.classList.remove("hidden");
    if (successBox) successBox.classList.add("hidden");
  }

  function hideError() {
    if (errorBox) errorBox.classList.add("hidden");
  }

  function showSuccess(msg) {
    if (successText) successText.textContent = msg;
    if (successBox) successBox.classList.remove("hidden");
    if (errorBox) errorBox.classList.add("hidden");
  }

  function hideSuccess() {
    if (successBox) successBox.classList.add("hidden");
  }

  function clearAlerts() {
    hideError();
    hideSuccess();
  }

  // --------------------------------------------------------------------------
  // 3. Password Visibility Toggle
  // --------------------------------------------------------------------------
  function setupPasswordToggle(btnId, inputId) {
    const btn = document.getElementById(btnId);
    const input = document.getElementById(inputId);
    if (!btn || !input) return;

    btn.addEventListener("click", function (e) {
      e.preventDefault();
      const isPassword = input.getAttribute("type") === "password";
      const newType = isPassword ? "text" : "password";
      input.setAttribute("type", newType);

      const eyeShow = btn.querySelector(".eye-show");
      const eyeHide = btn.querySelector(".eye-hide");

      if (isPassword) {
        // Now visible
        btn.setAttribute("aria-label", "Hide password");
        btn.setAttribute("title", "Hide password");
        if (eyeShow) eyeShow.classList.add("hidden");
        if (eyeHide) eyeHide.classList.remove("hidden");
      } else {
        // Now masked
        btn.setAttribute("aria-label", "Show password");
        btn.setAttribute("title", "Show password");
        if (eyeShow) eyeShow.classList.remove("hidden");
        if (eyeHide) eyeHide.classList.add("hidden");
      }
    });
  }

  // --------------------------------------------------------------------------
  // 4. View Switching (Sign In vs Create Account)
  // --------------------------------------------------------------------------
  function setMode(mode) {
    clearAlerts();

    if (mode === "register") {
      if (tabLogin) {
        tabLogin.classList.remove("active");
        tabLogin.setAttribute("aria-selected", "false");
      }
      if (tabRegister) {
        tabRegister.classList.add("active");
        tabRegister.setAttribute("aria-selected", "true");
      }
      if (loginSection) loginSection.classList.add("hidden");
      if (registerSection) registerSection.classList.remove("hidden");

      if (cardTitle) cardTitle.textContent = "Create Agent Account";
      if (cardSubtitle) cardSubtitle.textContent = "Register a new support agent profile to access live diagnostics.";

      if (regNameInput) regNameInput.focus();
    } else {
      if (tabRegister) {
        tabRegister.classList.remove("active");
        tabRegister.setAttribute("aria-selected", "false");
      }
      if (tabLogin) {
        tabLogin.classList.add("active");
        tabLogin.setAttribute("aria-selected", "true");
      }
      if (registerSection) registerSection.classList.add("hidden");
      if (loginSection) loginSection.classList.remove("hidden");

      if (cardTitle) cardTitle.textContent = "Support Agent Sign In";
      if (cardSubtitle) cardSubtitle.textContent = "Sign in with your authorized support email to access the live diagnostics console.";

      if (loginEmailInput) loginEmailInput.focus();
    }
  }

  // --------------------------------------------------------------------------
  // 5. Demo Account Quick-Fill Helpers
  // --------------------------------------------------------------------------
  function initDemoButtons() {
    if (demoBoomikaBtn) {
      demoBoomikaBtn.addEventListener("click", () => {
        setMode("login");
        if (loginEmailInput) loginEmailInput.value = "boomika@support.industrial.ai";
        if (loginPasswordInput) loginPasswordInput.value = "Industrial@2026";
        clearAlerts();
        if (loginPasswordInput) loginPasswordInput.focus();
      });
    }

    if (demoAlexBtn) {
      demoAlexBtn.addEventListener("click", () => {
        setMode("login");
        if (loginEmailInput) loginEmailInput.value = "alex@support.industrial.ai";
        if (loginPasswordInput) loginPasswordInput.value = "Machinery#400";
        clearAlerts();
        if (loginPasswordInput) loginPasswordInput.focus();
      });
    }
  }

  // --------------------------------------------------------------------------
  // 6. Sign In Submission
  // --------------------------------------------------------------------------
  async function handleLoginSubmit(event) {
    event.preventDefault();
    clearAlerts();

    const email = loginEmailInput ? loginEmailInput.value.trim() : "";
    const password = loginPasswordInput ? loginPasswordInput.value : "";

    if (!email) {
      showError("Please enter your support email address.");
      if (loginEmailInput) loginEmailInput.focus();
      return;
    }

    if (!password) {
      showError("Please enter your password.");
      if (loginPasswordInput) loginPasswordInput.focus();
      return;
    }

    // Set loading state
    if (loginSubmitBtn) loginSubmitBtn.disabled = true;
    if (loginBtnSpinner) loginBtnSpinner.classList.remove("hidden");
    if (loginBtnText) loginBtnText.textContent = "Signing in...";

    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({ email, password })
      });

      const data = await response.json().catch(() => ({}));

      if (response.ok && data.authenticated) {
        window.location.href = "/dashboard";
      } else {
        const message = data.error || "Invalid email or password. Please try again.";
        showError(message);
        if (loginSubmitBtn) loginSubmitBtn.disabled = false;
        if (loginBtnSpinner) loginBtnSpinner.classList.add("hidden");
        if (loginBtnText) loginBtnText.textContent = "Sign In";
        if (loginPasswordInput) {
          loginPasswordInput.value = "";
          loginPasswordInput.focus();
        }
      }
    } catch (err) {
      showError("Connection error: Unable to reach authentication server.");
      if (loginSubmitBtn) loginSubmitBtn.disabled = false;
      if (loginBtnSpinner) loginBtnSpinner.classList.add("hidden");
      if (loginBtnText) loginBtnText.textContent = "Sign In";
    }
  }

  // --------------------------------------------------------------------------
  // 7. Account Creation Submission
  // --------------------------------------------------------------------------
  async function handleRegisterSubmit(event) {
    event.preventDefault();
    clearAlerts();

    const name = regNameInput ? regNameInput.value.trim() : "";
    const email = regEmailInput ? regEmailInput.value.trim() : "";
    const password = regPasswordInput ? regPasswordInput.value : "";
    const confirmPassword = regConfirmPasswordInput ? regConfirmPasswordInput.value : "";

    // Client-side validations
    if (!name) {
      showError("Please enter your full name.");
      if (regNameInput) regNameInput.focus();
      return;
    }

    if (!email) {
      showError("Please enter your work email address.");
      if (regEmailInput) regEmailInput.focus();
      return;
    }

    if (!EMAIL_REGEX.test(email)) {
      showError("Please enter a valid work email address.");
      if (regEmailInput) regEmailInput.focus();
      return;
    }

    if (!password) {
      showError("Please enter a password.");
      if (regPasswordInput) regPasswordInput.focus();
      return;
    }

    if (password.length < 10) {
      showError("Password must be at least 10 characters long.");
      if (regPasswordInput) regPasswordInput.focus();
      return;
    }

    if (password !== confirmPassword) {
      showError("Passwords do not match. Please re-enter.");
      if (regConfirmPasswordInput) regConfirmPasswordInput.focus();
      return;
    }

    // Set loading state
    if (registerSubmitBtn) registerSubmitBtn.disabled = true;
    if (regBtnSpinner) regBtnSpinner.classList.remove("hidden");
    if (regBtnText) regBtnText.textContent = "Creating Account...";

    try {
      const response = await fetch("/api/auth/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({
          name,
          email,
          password,
          confirm_password: confirmPassword
        })
      });

      const data = await response.json().catch(() => ({}));

      if (response.ok && data.authenticated) {
        showSuccess("Account created successfully! Redirecting to dashboard...");
        setTimeout(() => {
          window.location.href = "/dashboard";
        }, 600);
      } else {
        const message = data.error || "Failed to create account. Please check your information.";
        showError(message);
        if (registerSubmitBtn) registerSubmitBtn.disabled = false;
        if (regBtnSpinner) regBtnSpinner.classList.add("hidden");
        if (regBtnText) regBtnText.textContent = "Create Account & Sign In";
      }
    } catch (err) {
      showError("Connection error: Unable to reach authentication server.");
      if (registerSubmitBtn) registerSubmitBtn.disabled = false;
      if (regBtnSpinner) regBtnSpinner.classList.add("hidden");
      if (regBtnText) regBtnText.textContent = "Create Account & Sign In";
    }
  }

  // --------------------------------------------------------------------------
  // 8. Check Existing Authenticated Session
  // --------------------------------------------------------------------------
  async function checkExistingSession() {
    try {
      const res = await fetch("/api/auth/me");
      if (res.ok) {
        const data = await res.json();
        if (data.authenticated) {
          window.location.href = "/dashboard";
        }
      }
    } catch (e) {}
  }

  // --------------------------------------------------------------------------
  // 9. Initialization
  // --------------------------------------------------------------------------
  function init() {
    initTheme();
    initDemoButtons();

    // Setup password visibility toggles
    setupPasswordToggle("toggle-login-password-btn", "login-password-input");
    setupPasswordToggle("toggle-reg-password-btn", "reg-password-input");
    setupPasswordToggle("toggle-reg-confirm-btn", "reg-confirm-password-input");

    // Mode tab and link switchers
    if (tabLogin) tabLogin.addEventListener("click", () => setMode("login"));
    if (tabRegister) tabRegister.addEventListener("click", () => setMode("register"));
    if (linkToRegister) linkToRegister.addEventListener("click", () => setMode("register"));
    if (linkToLogin) linkToLogin.addEventListener("click", () => setMode("login"));

    // Form submits
    if (loginForm) loginForm.addEventListener("submit", handleLoginSubmit);
    if (registerForm) registerForm.addEventListener("submit", handleRegisterSubmit);

    // Check if session already active
    checkExistingSession();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

/* ═══════════════════════════════════════════════
   Research Agent — Frontend Logic
   Chat interface, document attachment, and state
   ═══════════════════════════════════════════════ */

(() => {
    "use strict";

    // ── Helper for safe UUID generation ──
    function generateUUID() {
        if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
            try {
                return crypto.randomUUID();
            } catch (e) {
                // fallback below
            }
        }
        return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
            const r = (Math.random() * 16) | 0;
            const v = c === "x" ? r : (r & 0x3) | 0x8;
            return v.toString(16);
        });
    }

    // ── State ──
    let threadId = generateUUID();
    let isProcessing = false;
    let pendingAttachment = null;
    let uploadPromise = null;

    // ── DOM References ──
    const $ = (sel) => document.querySelector(sel);
    const messagesContainer = $("#messagesContainer");
    const welcomeSection = $("#welcomeSection");
    const userInput = $("#userInput");
    const sendBtn = $("#sendBtn");
    const menuBtn = $("#menuBtn");
    const sidebar = $("#sidebar");
    const sidebarOverlay = $("#sidebarOverlay");
    const newChatBtn = $("#newChatBtn");
    const suggestions = $("#suggestions");

    // Upload & Attachment elements
    const fileInput = $("#fileInput");
    const attachBtn = $("#attachBtn");
    const uploadHint = $("#uploadHint");
    const uploadStatus = $("#uploadStatus");
    const uploadStatusText = $("#uploadStatusText");
    const uploadDismiss = $("#uploadDismiss");
    const uploadProgressBar = $("#uploadProgressBar");
    const attachmentPreviewContainer = $("#attachmentPreviewContainer");
    const attachmentPillName = $("#attachmentPillName");
    const attachmentPillStatus = $("#attachmentPillStatus");
    const attachmentPillIcon = $("#attachmentPillIcon");
    const attachmentPillRemove = $("#attachmentPillRemove");
    const dragDropOverlay = $("#dragDropOverlay");

    // ── Markdown Setup ──
    if (typeof marked !== "undefined" && typeof marked.setOptions === "function") {
        try {
            marked.setOptions({
                highlight: (code, lang) => {
                    if (typeof hljs !== "undefined" && lang && hljs.getLanguage(lang)) {
                        return hljs.highlight(code, { language: lang }).value;
                    }
                    return typeof hljs !== "undefined" ? hljs.highlightAuto(code).value : code;
                },
                breaks: true,
                gfm: true,
            });
        } catch (e) {
            console.warn("Marked setup warning:", e);
        }
    }

    // ── Tool Icon Map ──
    const toolIcons = {
        calculator: "🧮",
        weather: "🌤️",
        file_read: "📖",
        write_file: "✍️",
        Hybrid_Rag: "📄",
        Research_tool: "🔬",
    };

    // ══════════════════════════════════════════════
    // Message Rendering
    // ══════════════════════════════════════════════

    function addMessage(role, content, toolsUsed = [], attachmentInfo = null) {
        // Hide welcome on first message
        const welcome = document.getElementById("welcomeSection");
        if (welcome && !welcome.classList.contains("hidden")) {
            welcome.classList.add("hidden");
            setTimeout(() => {
                const el = document.getElementById("welcomeSection");
                if (el) el.remove();
            }, 350);
        }

        const msgEl = document.createElement("div");
        msgEl.className = `message ${role}`;

        const avatarLabel = role === "user" ? "You" : "AI";
        const avatarInitials = role === "user" ? "U" : "AI";

        let bubbleHTML;
        if (role === "agent") {
            bubbleHTML = renderMarkdown(content);
        } else {
            bubbleHTML = escapeHTML(content);
        }

        // Attachment badge inside user message
        let attachmentBadgeHTML = "";
        if (attachmentInfo) {
            const icon = getFileIcon(attachmentInfo.name);
            attachmentBadgeHTML = `
                <div class="user-attachment-badge">
                    <span class="badge-icon">${icon}</span>
                    <span>${escapeHTML(attachmentInfo.name)}</span>
                    ${attachmentInfo.size ? `<span style="opacity:0.75;font-size:0.7em">(${formatBytes(attachmentInfo.size)})</span>` : ""}
                </div>
            `;
        }

        let toolBadgesHTML = "";
        if (toolsUsed && toolsUsed.length > 0) {
            const badges = toolsUsed.map((t) => {
                const icon = toolIcons[t] || "⚙️";
                return `<span class="tool-badge">
                    <span>${icon}</span>
                    <span>${t}</span>
                </span>`;
            }).join("");
            toolBadgesHTML = `<div class="tool-badges">${badges}</div>`;
        }

        msgEl.innerHTML = `
            <div class="msg-avatar" aria-label="${avatarLabel}">${avatarInitials}</div>
            <div class="msg-content">
                <div class="msg-bubble">
                    ${attachmentBadgeHTML}
                    <div>${bubbleHTML}</div>
                </div>
                ${toolBadgesHTML}
            </div>
        `;

        messagesContainer.appendChild(msgEl);
        scrollToBottom();

        // Apply syntax highlighting to any code blocks
        msgEl.querySelectorAll("pre code").forEach((block) => {
            hljs.highlightElement(block);
        });

        return msgEl;
    }

    function renderMarkdown(text) {
        try {
            return marked.parse(text);
        } catch {
            return escapeHTML(text);
        }
    }

    function escapeHTML(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    function formatBytes(bytes) {
        if (!bytes) return "";
        const kb = bytes / 1024;
        if (kb < 1024) return `${kb.toFixed(1)} KB`;
        return `${(kb / 1024).toFixed(1)} MB`;
    }

    function getFileIcon(filename) {
        const ext = filename.split(".").pop().toLowerCase();
        if (ext === "pdf") return "📄";
        if (["txt", "md"].includes(ext)) return "📝";
        if (ext === "csv") return "📊";
        return "📁";
    }

    // ── Typing Indicator ──
    function showTyping() {
        const el = document.createElement("div");
        el.className = "message agent";
        el.id = "typingMsg";
        el.innerHTML = `
            <div class="msg-avatar" aria-label="AI">AI</div>
            <div class="msg-content">
                <div class="msg-bubble">
                    <div class="typing-indicator">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            </div>
        `;
        messagesContainer.appendChild(el);
        scrollToBottom();
    }

    function hideTyping() {
        const el = document.getElementById("typingMsg");
        if (el) el.remove();
    }

    // ── Error Message ──
    function showError(text) {
        const el = document.createElement("div");
        el.className = "message agent";
        el.innerHTML = `
            <div class="msg-avatar" aria-label="AI">AI</div>
            <div class="msg-content">
                <div class="msg-error">
                    <span>⚠️</span>
                    <span>${escapeHTML(text)}</span>
                </div>
            </div>
        `;
        messagesContainer.appendChild(el);
        scrollToBottom();
    }


    // ══════════════════════════════════════════════
    // Attachment & Upload Management
    // ══════════════════════════════════════════════

    function handleSelectedFile(file) {
        if (!file) return;

        const ext = file.name.split(".").pop().toLowerCase();
        if (!["pdf", "txt", "md", "csv"].includes(ext)) {
            showError("Unsupported file type. Please upload PDF, TXT, MD, or CSV files.");
            return;
        }

        if (file.size > 15 * 1024 * 1024) {
            showError("File is too large (maximum size is 15 MB).");
            return;
        }

        pendingAttachment = file;

        // Show attachment pill
        if (attachmentPreviewContainer) {
            attachmentPreviewContainer.hidden = false;
            attachmentPillName.textContent = file.name;
            attachmentPillIcon.textContent = getFileIcon(file.name);
            attachmentPillStatus.textContent = `Uploading & indexing (${formatBytes(file.size)})…`;
        }

        updateSendBtn();

        // Start upload immediately in background
        uploadPromise = performUpload(file);
    }

    async function performUpload(file) {
        // Show status feedback
        uploadStatus.hidden = false;
        uploadStatus.className = "upload-status";
        uploadDismiss.hidden = true;
        uploadStatusText.textContent = `Indexing "${file.name}" into knowledge base…`;
        uploadProgressBar.style.width = "40%";

        const formData = new FormData();
        formData.append("file", file);

        try {
            uploadProgressBar.style.width = "75%";
            const res = await fetch("/upload", {
                method: "POST",
                body: formData,
            });

            uploadProgressBar.style.width = "100%";

            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `Upload failed (${res.status})`);
            }

            const data = await res.json();

            // Status bar success
            uploadStatus.classList.add("success");
            uploadStatusText.textContent = `✓ "${data.filename}" indexed (${data.chunks_added} chunks added)`;
            uploadDismiss.hidden = false;

            if (attachmentPillStatus) {
                attachmentPillStatus.textContent = `✓ Ready to send (${data.chunks_added} chunks indexed)`;
            }

            return data;
        } catch (err) {
            uploadStatus.classList.add("error");
            uploadStatusText.textContent = `✗ Upload failed: ${err.message}`;
            uploadDismiss.hidden = false;

            if (attachmentPillStatus) {
                attachmentPillStatus.textContent = `✗ Upload error: ${err.message}`;
            }
            throw err;
        }
    }

    function removeAttachment() {
        pendingAttachment = null;
        uploadPromise = null;
        if (attachmentPreviewContainer) {
            attachmentPreviewContainer.hidden = true;
        }
        if (fileInput) {
            fileInput.value = "";
        }
        updateSendBtn();
    }

    function dismissUploadStatus() {
        uploadStatus.hidden = true;
        uploadStatus.className = "upload-status";
        uploadProgressBar.style.width = "0%";
    }


    // ══════════════════════════════════════════════
    // Send Message
    // ══════════════════════════════════════════════

    async function sendMessage(text) {
        const rawText = (text || userInput.value).trim();
        const hasAttachment = Boolean(pendingAttachment);

        if ((!rawText && !hasAttachment) || isProcessing) return;

        isProcessing = true;
        userInput.value = "";
        autoResize();
        updateSendBtn();

        const currentAttachment = pendingAttachment;
        const currentUploadPromise = uploadPromise;

        // Clear preview immediately
        removeAttachment();

        // Build the prompt
        let promptToSend = rawText;
        let displayPrompt = rawText;

        if (currentAttachment) {
            if (!rawText) {
                displayPrompt = `Please analyze and summarize the attached document: **${currentAttachment.name}**. Provide key findings, methodologies, and conclusions.`;
                promptToSend = `Please analyze the uploaded document '${currentAttachment.name}'. Provide a comprehensive summary of its key points, findings, and takeaways using Research_tool or Hybrid_Rag.`;
            } else {
                promptToSend = `${rawText}\n\n[Context: The user attached document '${currentAttachment.name}']`;
            }
        }

        // Add user message with attachment badge
        addMessage("user", displayPrompt, [], currentAttachment ? { name: currentAttachment.name, size: currentAttachment.size } : null);

        // Show typing
        showTyping();

        try {
            // If attachment is still uploading, wait for completion
            if (currentUploadPromise) {
                await currentUploadPromise.catch((err) => {
                    console.warn("Upload had issue, attempting to continue with chat:", err);
                });
            }

            const res = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: promptToSend,
                    thread_id: threadId,
                }),
            });

            hideTyping();

            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `Server error (${res.status})`);
            }

            const data = await res.json();
            threadId = data.thread_id;
            addMessage("agent", data.response, data.tools_used || []);
        } catch (err) {
            hideTyping();
            showError(err.message || "Something went wrong. Please try again.");
        } finally {
            isProcessing = false;
            updateSendBtn();
            userInput.focus();
        }
    }


    // ══════════════════════════════════════════════
    // UI Helpers
    // ══════════════════════════════════════════════

    function scrollToBottom() {
        requestAnimationFrame(() => {
            messagesContainer.scrollTo({
                top: messagesContainer.scrollHeight,
                behavior: "smooth",
            });
        });
    }

    function autoResize() {
        userInput.style.height = "auto";
        userInput.style.height = Math.min(userInput.scrollHeight, 160) + "px";
    }

    function updateSendBtn() {
        const hasText = Boolean(userInput.value.trim());
        const hasAttachment = Boolean(pendingAttachment);
        sendBtn.disabled = (!hasText && !hasAttachment) || isProcessing;
    }

    function newChat() {
        threadId = generateUUID();
        removeAttachment();

        // Remove all messages
        const messages = messagesContainer.querySelectorAll(".message");
        messages.forEach((m) => m.remove());

        // Re-add welcome section if it was removed
        if (!document.getElementById("welcomeSection")) {
            messagesContainer.innerHTML = createWelcomeHTML();
            bindSuggestions();
        }
    }

    function createWelcomeHTML() {
        return `
            <div class="welcome" id="welcomeSection">
                <div class="welcome-glow" aria-hidden="true"></div>
                <div class="welcome-icon">
                    <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="url(#wgrad2)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                        <defs>
                            <linearGradient id="wgrad2" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" style="stop-color:#8B5CF6"/>
                                <stop offset="50%" style="stop-color:#3B82F6"/>
                                <stop offset="100%" style="stop-color:#06B6D4"/>
                            </linearGradient>
                        </defs>
                        <circle cx="11" cy="11" r="8"/>
                        <path d="M21 21l-4.35-4.35"/>
                        <path d="M11 8v6"/>
                        <path d="M8 11h6"/>
                    </svg>
                </div>
                <h2 class="welcome-title">How can I help with your research?</h2>
                <p class="welcome-desc">Ask me about research papers, attach your own PDFs to analyze, perform calculations, or search the web.</p>
                <div class="suggestions" id="suggestions">
                    <button class="suggestion-card" data-query="What is Retrieval Augmented Generation?">
                        <span class="suggestion-emoji">📄</span>
                        <span class="suggestion-label">What is Retrieval Augmented Generation?</span>
                    </button>
                    <button class="suggestion-card" data-query="Explain the RAG pipeline architecture in detail">
                        <span class="suggestion-emoji">🔬</span>
                        <span class="suggestion-label">Explain the RAG pipeline architecture</span>
                    </button>
                    <button class="suggestion-card" data-query="256 * 48 + 1024">
                        <span class="suggestion-emoji">🧮</span>
                        <span class="suggestion-label">Calculate 256 × 48 + 1024</span>
                    </button>
                    <button class="suggestion-card" data-query="What is the weather in Ranchi?">
                        <span class="suggestion-emoji">🌤️</span>
                        <span class="suggestion-label">What's the weather in Ranchi?</span>
                    </button>
                </div>
            </div>
        `;
    }

    // ── Sidebar Toggle (Mobile) ──
    function toggleSidebar() {
        sidebar.classList.toggle("open");
        sidebarOverlay.classList.toggle("visible");
    }

    function closeSidebar() {
        sidebar.classList.remove("open");
        sidebarOverlay.classList.remove("visible");
    }


    // ══════════════════════════════════════════════
    // Event Listeners & Drag/Drop
    // ══════════════════════════════════════════════

    // Send button click
    sendBtn.addEventListener("click", () => sendMessage());

    // Enter to send, Shift+Enter for newline
    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea & update send button state
    ["input", "keyup", "change"].forEach((evt) => {
        userInput.addEventListener(evt, () => {
            autoResize();
            updateSendBtn();
        });
    });

    // Sidebar toggle
    menuBtn.addEventListener("click", toggleSidebar);
    sidebarOverlay.addEventListener("click", closeSidebar);

    // New chat
    newChatBtn.addEventListener("click", newChat);

    // Upload: attach button opens file picker
    if (attachBtn) {
        attachBtn.addEventListener("click", () => fileInput.click());
    }

    if (uploadHint) {
        uploadHint.addEventListener("click", () => fileInput.click());
    }

    // Upload: file selected from file picker
    if (fileInput) {
        fileInput.addEventListener("change", () => {
            if (fileInput.files && fileInput.files.length > 0) {
                handleSelectedFile(fileInput.files[0]);
            }
        });
    }

    // Remove attachment pill
    if (attachmentPillRemove) {
        attachmentPillRemove.addEventListener("click", removeAttachment);
    }

    // Dismiss upload status bar
    if (uploadDismiss) {
        uploadDismiss.addEventListener("click", dismissUploadStatus);
    }

    // ── Drag and Drop Support ──
    let dragCounter = 0;

    window.addEventListener("dragenter", (e) => {
        e.preventDefault();
        dragCounter++;
        if (dragDropOverlay) {
            dragDropOverlay.classList.add("active");
        }
    });

    window.addEventListener("dragleave", (e) => {
        e.preventDefault();
        dragCounter--;
        if (dragCounter <= 0) {
            dragCounter = 0;
            if (dragDropOverlay) {
                dragDropOverlay.classList.remove("active");
            }
        }
    });

    window.addEventListener("dragover", (e) => {
        e.preventDefault();
    });

    window.addEventListener("drop", (e) => {
        e.preventDefault();
        dragCounter = 0;
        if (dragDropOverlay) {
            dragDropOverlay.classList.remove("active");
        }

        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    // ── Paste File Support ──
    window.addEventListener("paste", (e) => {
        if (e.clipboardData && e.clipboardData.files && e.clipboardData.files.length > 0) {
            handleSelectedFile(e.clipboardData.files[0]);
        }
    });

    // ── Theme Manager ──
    const themeBtn = document.getElementById("themeBtn");
    const themeDropdown = document.getElementById("themeDropdown");
    const themeOptions = document.querySelectorAll(".theme-option");

    function applyTheme(themeName) {
        document.documentElement.setAttribute("data-theme", themeName);
        localStorage.setItem("research_agent_theme", themeName);
        themeOptions.forEach((opt) => {
            if (opt.dataset.theme === themeName) {
                opt.classList.add("active");
            } else {
                opt.classList.remove("active");
            }
        });
    }

    // Initialize theme from storage
    const savedTheme = localStorage.getItem("research_agent_theme") || "cosmic";
    applyTheme(savedTheme);

    if (themeBtn && themeDropdown) {
        themeBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            themeDropdown.hidden = !themeDropdown.hidden;
        });

        document.addEventListener("click", (e) => {
            if (!themeDropdown.contains(e.target) && e.target !== themeBtn) {
                themeDropdown.hidden = true;
            }
        });

        themeOptions.forEach((btn) => {
            btn.addEventListener("click", (e) => {
                e.stopPropagation();
                const selected = btn.dataset.theme;
                if (selected) {
                    applyTheme(selected);
                    themeDropdown.hidden = true;
                }
            });
        });
    }

    // Focus input on page load
    userInput.focus();
})();

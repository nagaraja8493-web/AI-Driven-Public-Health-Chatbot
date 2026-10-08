/**
 * HealthAware AI - Chatbot Client-Side Logic
 * Manages model switching, confidence thresholding, Web Speech API (Voice & TTS),
 * dynamic chat bubble rendering, and API communication.
 */

let activeModel = 'ml';
let confidenceThreshold = 0.60;
let ttsEnabled = false;
let recognition = null;
let isRecording = false;

document.addEventListener('DOMContentLoaded', () => {
    // 1. Model switch radio listeners
    const radioML = document.getElementById('radioML');
    const radioDL = document.getElementById('radioDL');
    const headerBadge = document.getElementById('headerModelBadge');

    if (radioML && radioDL) {
        radioML.addEventListener('change', () => {
            if (radioML.checked) {
                activeModel = 'ml';
                if (headerBadge) headerBadge.innerText = 'Logistic Regression (ML)';
            }
        });
        radioDL.addEventListener('change', () => {
            if (radioDL.checked) {
                activeModel = 'dl';
                if (headerBadge) headerBadge.innerText = 'Bi-LSTM (DL)';
            }
        });
    }

    // 2. Threshold slider listener
    const confSlider = document.getElementById('confidenceRange');
    const confDisplay = document.getElementById('thresholdValueDisplay');
    if (confSlider && confDisplay) {
        confSlider.addEventListener('input', (e) => {
            confidenceThreshold = parseInt(e.target.value, 10) / 100.0;
            confDisplay.innerText = `${e.target.value}%`;
        });
    }

    // 3. TTS Toggle Button
    const ttsBtn = document.getElementById('ttsToggleBtn');
    if (ttsBtn) {
        ttsBtn.addEventListener('click', () => {
            ttsEnabled = !ttsEnabled;
            ttsBtn.classList.toggle('active', ttsEnabled);
            ttsBtn.title = ttsEnabled ? "Auto-Speech: Enabled" : "Auto-Speech: Disabled";
        });
    }

    // 4. Clear chat button
    const clearBtn = document.getElementById('clearChatBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            const chatArea = document.getElementById('chatMessagesArea');
            if (chatArea) {
                chatArea.innerHTML = `
                    <div class="message-row bot">
                        <div class="avatar-circle bot"><i class="fa-solid fa-robot"></i></div>
                        <div class="message-content-wrapper">
                            <div class="message-bubble">
                                <strong>Chat history cleared.</strong> How may I assist you with health awareness today?
                                <div class="message-meta-info">
                                    <span>HealthAware AI Bot</span>
                                    <span class="badge-tag intent">greeting</span>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }
        });
    }

    // 5. Setup Web Speech Recognition
    setupVoiceRecognition();

    // 6. Setup Chat Scroll Detection for Jump-to-Latest Button
    const chatArea = document.getElementById('chatMessagesArea');
    const scrollBtn = document.getElementById('scrollToBottomBtn');
    if (chatArea && scrollBtn) {
        chatArea.addEventListener('scroll', () => {
            const distanceToBottom = chatArea.scrollHeight - chatArea.scrollTop - chatArea.clientHeight;
            if (distanceToBottom > 90) {
                scrollBtn.style.display = 'flex';
            } else {
                scrollBtn.style.display = 'none';
            }
        });
    }
});

// Setup Speech Recognition
function setupVoiceRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const micBtn = document.getElementById('voiceMicBtn');
    const inputField = document.getElementById('userInputText');

    if (!SpeechRecognition) {
        if (micBtn) {
            micBtn.style.opacity = '0.5';
            micBtn.title = 'Speech Recognition is not supported in this browser.';
        }
        return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-IN'; // Also understands Indian English / Kannada phonetics

    recognition.onstart = () => {
        isRecording = true;
        if (micBtn) micBtn.classList.add('listening');
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        if (inputField) {
            inputField.value = transcript;
            // Automatically submit after voice input
            const form = document.getElementById('chatForm');
            if (form) form.dispatchEvent(new Event('submit'));
        }
    };

    recognition.onerror = (event) => {
        console.warn('Speech recognition error:', event.error);
        stopRecording();
    };

    recognition.onend = () => {
        stopRecording();
    };

    if (micBtn) {
        micBtn.addEventListener('click', () => {
            if (isRecording) {
                recognition.stop();
            } else {
                try {
                    recognition.start();
                } catch (err) {
                    console.error('Recognition error:', err);
                }
            }
        });
    }
}

function stopRecording() {
    isRecording = false;
    const micBtn = document.getElementById('voiceMicBtn');
    if (micBtn) micBtn.classList.remove('listening');
}

// Speak message aloud using Web SpeechSynthesis
function speakMessage(text) {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel(); // Stop prior audio
    
    // Strip markdown formatting for cleaner speech
    const cleanSpeech = text
        .replace(/[*_#`~]/g, '')
        .replace(/🚨/g, 'Alert: ')
        .replace(/⚠️/g, 'Warning: ')
        .replace(/📞/g, 'Call ')
        .replace(/🚑/g, 'Ambulance: ');

    const utterance = new SpeechSynthesisUtterance(cleanSpeech);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
}

// Send quick suggested question
function sendSuggestedQuestion(questionText) {
    const inputField = document.getElementById('userInputText');
    if (inputField) {
        inputField.value = questionText;
        const form = document.getElementById('chatForm');
        if (form) form.dispatchEvent(new Event('submit'));
    }
}

// Main message submission handler
async function handleMessageSubmit(event) {
    event.preventDefault();
    const inputField = document.getElementById('userInputText');
    const message = inputField.value.trim();
    if (!message) return;

    // Clear input
    inputField.value = '';

    // Render User Bubble
    appendUserMessage(message);

    // Show Typing Indicator
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) typingIndicator.style.display = 'flex';
    scrollToBottom();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                model: activeModel,
                threshold: confidenceThreshold
            })
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();
        if (typingIndicator) typingIndicator.style.display = 'none';

        // Render Bot Bubble
        appendBotMessage(data);

        // Auto TTS if enabled
        if (ttsEnabled) {
            speakMessage(data.response);
        }
    } catch (err) {
        if (typingIndicator) typingIndicator.style.display = 'none';
        appendBotErrorMessage("Sorry, unable to connect to the classification engine. Please make sure the backend is active.");
        console.error('Chat API Error:', err);
    }
}

// Append User Message to Chat Area
function appendUserMessage(text) {
    const chatArea = document.getElementById('chatMessagesArea');
    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const msgRow = document.createElement('div');
    msgRow.className = 'message-row user';
    msgRow.innerHTML = `
        <div class="avatar-circle user"><i class="fa-solid fa-user"></i></div>
        <div class="message-content-wrapper">
            <div class="message-bubble">
                ${escapeHTML(text)}
                <div class="message-meta-info">
                    <span>${now}</span>
                </div>
            </div>
        </div>
    `;
    chatArea.appendChild(msgRow);
    scrollToBottom();
}

// Append Bot Message to Chat Area
function appendBotMessage(data) {
    const chatArea = document.getElementById('chatMessagesArea');
    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const isEmergency = data.is_emergency;
    const isFallback = data.is_fallback;
    const confidencePct = Math.round((data.confidence || 0) * 100);

    let confBadgeClass = 'conf-high';
    if (data.confidence < 0.60) confBadgeClass = 'conf-low';
    else if (data.confidence < 0.80) confBadgeClass = 'conf-med';

    // Format Markdown bold and bullets to HTML
    let formattedText = escapeHTML(data.response)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n• /g, '<br>&bull; ')
        .replace(/\n/g, '<br>');

    const msgRow = document.createElement('div');
    msgRow.className = 'message-row bot';

    let bubbleClass = 'message-bubble';
    if (isEmergency) bubbleClass += ' emergency-bubble';

    // Build breakdown snippet for top intents
    let topIntentsHtml = '';
    if (data.top_intents && data.top_intents.length > 1) {
        topIntentsHtml = `
            <div style="margin-top: 10px; padding-top: 8px; border-top: 1px dashed var(--border-color); font-size: 0.72rem; color: var(--text-muted);">
                <strong>Intent Breakdown:</strong>
                ${data.top_intents.map(item => `<span>${item.intent} (${Math.round(item.confidence * 100)}%)</span>`).join(' &bull; ')}
            </div>
        `;
    }

    msgRow.innerHTML = `
        <div class="avatar-circle bot" style="${isEmergency ? 'background: var(--emergency);' : ''}">
            <i class="fa-solid ${isEmergency ? 'fa-triangle-exclamation' : 'fa-robot'}"></i>
        </div>
        <div class="message-content-wrapper">
            <div class="${bubbleClass}">
                <div>${formattedText}</div>
                ${topIntentsHtml}
                <div class="message-meta-info">
                    <span>${now}</span>
                    <span class="badge-tag intent">${data.intent}</span>
                    <span class="badge-tag ${confBadgeClass}">${confidencePct}% Confidence</span>
                    <span style="font-size: 0.7rem; color: var(--text-muted);">${data.model_used}</span>
                    <button class="icon-btn" style="width: 24px; height: 24px; font-size: 0.75rem; border: none; background: transparent;" onclick="speakMessage('${escapeHTML(data.response).replace(/'/g, "\\'")}')" title="Read Aloud">
                        <i class="fa-solid fa-volume-high"></i>
                    </button>
                </div>
            </div>
        </div>
    `;
    chatArea.appendChild(msgRow);
    scrollToBottom();
}

function appendBotErrorMessage(msg) {
    const chatArea = document.getElementById('chatMessagesArea');
    const msgRow = document.createElement('div');
    msgRow.className = 'message-row bot';
    msgRow.innerHTML = `
        <div class="avatar-circle bot" style="background: var(--emergency);"><i class="fa-solid fa-circle-exclamation"></i></div>
        <div class="message-content-wrapper">
            <div class="message-bubble" style="background: #fee2e2; border-color: #fca5a5; color: #991b1b;">
                ${msg}
            </div>
        </div>
    `;
    chatArea.appendChild(msgRow);
    scrollToBottom();
}

function scrollToBottom(smooth = true) {
    const chatArea = document.getElementById('chatMessagesArea');
    if (!chatArea) return;

    chatArea.scrollTo({
        top: chatArea.scrollHeight,
        behavior: smooth ? 'smooth' : 'auto'
    });

    // Multi-stage adjustment for dynamic content reflow
    setTimeout(() => {
        chatArea.scrollTop = chatArea.scrollHeight;
    }, 50);

    setTimeout(() => {
        chatArea.scrollTop = chatArea.scrollHeight;
    }, 200);

    const scrollBtn = document.getElementById('scrollToBottomBtn');
    if (scrollBtn) {
        scrollBtn.style.display = 'none';
    }
}

function escapeHTML(str) {
    if (!str) return '';
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

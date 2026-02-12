// Kai Enterprise Platform - Frontend Logic

const API_BASE = '';  // Same origin

// View Management
function switchView(viewName) {
    // Hide all views
    document.querySelectorAll('.view-content').forEach(view => {
        view.classList.add('hidden');
    });

    // Remove active class from all nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected view
    const view = document.getElementById(`${viewName}-view`);
    if (view) {
        view.classList.remove('hidden');
    }

    // Add active class to clicked button
    const activeBtn = document.querySelector(`[data-view="${viewName}"]`);
    if (activeBtn) {
        activeBtn.classList.add('active');
    }

    // Update title
    const titles = {
        chat: 'Kai Concierge',
        audit: 'Klaudit Audit',
        creative: 'Creative Studio',
        pmax: 'PMax Deep Dive',
        serp: 'SERP Monitor',
        settings: 'Platform Settings'
    };
    document.getElementById('view-title').textContent = titles[viewName] || 'Kai Platform';

    // Load settings if settings view
    if (viewName === 'settings') {
        loadSettings();
    }
}

// Health Check
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        const statusEl = document.getElementById('health-status');

        if (data.status === 'healthy') {
            statusEl.classList.remove('offline');
            statusEl.title = `Engine: ${data.engine_available ? 'Available' : 'Unavailable'}`;
        } else {
            statusEl.classList.add('offline');
            statusEl.title = 'Offline';
        }
    } catch (error) {
        document.getElementById('health-status').classList.add('offline');
        console.error('Health check failed:', error);
    }
}

// Chat Functions
async function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();

    if (!message) return;

    // Add user message to chat
    addMessage(message, 'user');

    // Clear input
    input.value = '';

    // Show loading
    const loadingId = addMessage('<div class="spinner"></div>', 'assistant');

    try {
        const response = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        const data = await response.json();

        // Remove loading, add response
        document.getElementById(loadingId).remove();
        addMessage(data.response, 'assistant');

    } catch (error) {
        document.getElementById(loadingId).remove();
        addMessage('Error: Could not reach Kai Concierge. Please try again.', 'assistant');
        console.error('Chat error:', error);
    }
}

function addMessage(content, sender) {
    const messagesContainer = document.getElementById('chat-messages');

    // Remove placeholder if it exists
    const placeholder = messagesContainer.querySelector('.text-gray-400');
    if (placeholder) {
        placeholder.remove();
    }

    const messageId = `msg-${Date.now()}`;
    const messageHTML = `
        <div id="${messageId}" class="message-container ${sender}">
            <div class="message ${sender}">
                ${content}
            </div>
        </div>
    `;

    messagesContainer.insertAdjacentHTML('beforeend', messageHTML);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    return messageId;
}

// Audit Functions
async function uploadFile() {
    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];

    if (!file) return;

    const resultsDiv = document.getElementById('audit-results');
    resultsDiv.innerHTML = '<div class="text-center py-8"><div class="spinner mx-auto"></div><p class="mt-4 text-gray-600">Analyzing your data...</p></div>';
    resultsDiv.classList.remove('hidden');

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE}/api/audit`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.status === 'success') {
            displayAuditResults(data);
        } else {
            resultsDiv.innerHTML = `<div class="text-red-600">Error: ${data.detail}</div>`;
        }

    } catch (error) {
        resultsDiv.innerHTML = `<div class="text-red-600">Error: ${error.message}</div>`;
        console.error('Audit error:', error);
    }
}

function displayAuditResults(data) {
    const resultsDiv = document.getElementById('audit-results');

    let html = `
        <div class="bg-green-50 border border-green-200 rounded-lg p-4 mb-4">
            <h4 class="font-semibold text-green-800">✅ Analysis Complete</h4>
            <p class="text-sm text-green-700">Analyzed ${data.rows_analyzed} rows</p>
        </div>
    `;

    // Display summary
    if (data.summary) {
        html += `
            <div class="bg-white border rounded-lg p-4">
                <h4 class="font-semibold mb-2">Summary</h4>
                <pre class="text-sm bg-gray-50 p-4 rounded overflow-x-auto">${JSON.stringify(data.summary, null, 2)}</pre>
            </div>
        `;
    }

    resultsDiv.innerHTML = html;
}

// Creative Functions
async function generateCreative() {
    const product = document.getElementById('product-input').value.trim();
    const audience = document.getElementById('audience-input').value.trim();
    const tone = document.getElementById('tone-input').value;

    if (!product) {
        alert('Please enter a product description');
        return;
    }

    const resultsDiv = document.getElementById('creative-results');
    resultsDiv.innerHTML = '<div class="text-center py-8"><div class="spinner mx-auto"></div><p class="mt-4 text-gray-600">Generating creative variants...</p></div>';
    resultsDiv.classList.remove('hidden');

    try {
        const response = await fetch(`${API_BASE}/api/creative`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                product,
                audience: audience || null,
                tone,
                variants: 3
            })
        });

        const data = await response.json();

        if (data.status === 'success') {
            displayCreativeResults(data.variants);
        } else {
            resultsDiv.innerHTML = `<div class="text-red-600">Error: ${data.detail}</div>`;
        }

    } catch (error) {
        resultsDiv.innerHTML = `<div class="text-red-600">Error: ${error.message}</div>`;
        console.error('Creative error:', error);
    }
}

function displayCreativeResults(variantsJSON) {
    const resultsDiv = document.getElementById('creative-results');

    try {
        // Try to parse JSON from response
        const variants = typeof variantsJSON === 'string' ? JSON.parse(variantsJSON) : variantsJSON;

        let html = '<div class="space-y-4">';

        variants.forEach((variant, index) => {
            html += `
                <div class="creative-variant">
                    <div class="text-sm text-gray-500 mb-2">Variant ${index + 1}</div>
                    <div class="headline">${variant.headline || 'N/A'}</div>
                    <div class="description">${variant.description || 'N/A'}</div>
                    <div class="cta">CTA: ${variant.cta || 'N/A'}</div>
                </div>
            `;
        });

        html += '</div>';
        resultsDiv.innerHTML = html;

    } catch (error) {
        // If not JSON, display as text
        resultsDiv.innerHTML = `
            <div class="bg-white border rounded-lg p-4">
                <pre class="text-sm whitespace-pre-wrap">${variantsJSON}</pre>
            </div>
        `;
    }
}

// Settings Functions
async function loadSettings() {
    const settingsDiv = document.getElementById('settings-info');

    try {
        const response = await fetch(`${API_BASE}/api/settings`);
        const data = await response.json();

        const html = `
            <div class="space-y-4">
                <div class="flex items-center justify-between py-2 border-b">
                    <span class="font-medium">Engine Status</span>
                    <span class="${data.engine_available ? 'text-green-600' : 'text-red-600'}">
                        ${data.engine_available ? '✅ Available' : '❌ Not Available'}
                    </span>
                </div>
                <div class="flex items-center justify-between py-2 border-b">
                    <span class="font-medium">Azure OpenAI</span>
                    <span class="${data.azure_openai_configured ? 'text-green-600' : 'text-red-600'}">
                        ${data.azure_openai_configured ? '✅ Configured' : '❌ Not Configured'}
                    </span>
                </div>
                <div class="flex items-center justify-between py-2 border-b">
                    <span class="font-medium">Azure Search</span>
                    <span class="${data.azure_search_configured ? 'text-green-600' : 'text-red-600'}">
                        ${data.azure_search_configured ? '✅ Configured' : '❌ Not Configured'}
                    </span>
                </div>
                <div class="flex items-center justify-between py-2">
                    <span class="font-medium">Environment</span>
                    <span class="text-gray-600">${data.environment}</span>
                </div>
            </div>
        `;

        settingsDiv.innerHTML = html;

    } catch (error) {
        settingsDiv.innerHTML = '<p class="text-red-600">Error loading settings</p>';
        console.error('Settings error:', error);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Set initial view
    switchView('chat');

    // Start health checks
    checkHealth();
    setInterval(checkHealth, 30000);  // Every 30 seconds
});

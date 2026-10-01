/**
 * Macro Quant Dashboard — Client-side Application
 * Fetches data from FastAPI backend and renders interactive UI
 */

const API_BASE = '';

// ==========================================================================
// State
// ==========================================================================

let currentPage = 'overview';
let surpriseChart = null;
let nowcastChart = null;
let backtestChart = null;

// ==========================================================================
// Navigation
// ==========================================================================

document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        const page = item.dataset.page;
        navigateTo(page);
    });
});

function navigateTo(page) {
    currentPage = page;

    // Update nav
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.querySelector(`[data-page="${page}"]`).classList.add('active');

    // Update pages
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    const pageEl = document.getElementById(`page-${page}`);
    if (pageEl) pageEl.classList.add('active');

    // Update title
    const titles = {
        overview: 'Overview',
        calendar: 'Economic Calendar',
        nowcast: 'Nowcast Estimates',
        signals: 'Trading Signals',
        backtest: 'Backtest Results',
        system: 'System Status'
    };
    document.getElementById('page-title').textContent = titles[page] || page;

    // Load page data
    loadPageData(page);
}

// ==========================================================================
// Data Loading
// ==========================================================================

async function fetchJSON(path) {
    try {
        const res = await fetch(`${API_BASE}${path}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.error(`API Error (${path}):`, err);
        return null;
    }
}

async function loadPageData(page) {
    switch (page) {
        case 'overview': await loadOverview(); break;
        case 'calendar': await loadCalendar(); break;
        case 'nowcast': await loadNowcast(); break;
        case 'signals': await loadSignals(); break;
        case 'backtest': await loadBacktest(); break;
        case 'system': await loadSystem(); break;
    }
}

// ==========================================================================
// Overview Page
// ==========================================================================

async function loadOverview() {
    const [releasesData, calendarData] = await Promise.all([
        fetchJSON('/api/releases'),
        fetchJSON('/api/calendar'),
    ]);

    if (releasesData) renderKPIs(releasesData.releases);
    if (releasesData) renderReleasesTable(releasesData.releases);
    if (calendarData) renderUpcomingList(calendarData.events);

    // Load surprise chart
    await loadSurpriseChart('CPI');
}

function renderKPIs(releases) {
    const grid = document.getElementById('kpi-grid');
    const kpis = releases.filter(r => r.actual !== null).slice(0, 6);

    grid.innerHTML = kpis.map(r => {
        const surpriseClass = r.surprise > 0 ? 'positive' : r.surprise < 0 ? 'negative' : 'neutral';
        const arrow = r.surprise > 0 ? '↑' : r.surprise < 0 ? '↓' : '→';
        const surpriseText = r.surprise !== null
            ? `${arrow} ${r.surprise > 0 ? '+' : ''}${formatNum(r.surprise)}`
            : '—';

        return `
            <div class="kpi-card">
                <div class="kpi-label">${formatIndicator(r.indicator)}</div>
                <div class="kpi-value">${formatNum(r.actual)}</div>
                <div class="kpi-change ${surpriseClass}">
                    ${surpriseText} vs consensus
                </div>
            </div>
        `;
    }).join('');
}

function renderReleasesTable(releases) {
    const tbody = document.getElementById('releases-tbody');
    tbody.innerHTML = releases.filter(r => r.actual !== null).map(r => {
        const surpriseClass = r.surprise > 0 ? 'cell-positive' : r.surprise < 0 ? 'cell-negative' : 'cell-neutral';
        const zClass = Math.abs(r.surprise_zscore || 0) > 1 ? surpriseClass : 'cell-neutral';

        return `
            <tr>
                <td>${formatIndicator(r.indicator)}</td>
                <td>${formatDate(r.release_date)}</td>
                <td>${formatNum(r.actual)}</td>
                <td>${r.consensus !== null ? formatNum(r.consensus) : '—'}</td>
                <td class="${surpriseClass}">${r.surprise !== null ? (r.surprise > 0 ? '+' : '') + formatNum(r.surprise) : '—'}</td>
                <td class="${zClass}">${r.surprise_zscore !== null ? formatNum(r.surprise_zscore, 2) + 'σ' : '—'}</td>
                <td><span class="badge">${r.source}</span></td>
            </tr>
        `;
    }).join('');
}

function renderUpcomingList(events) {
    const list = document.getElementById('upcoming-list');
    list.innerHTML = events.map(e => {
        const d = new Date(e.scheduled_date);
        const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

        return `
            <div class="upcoming-item">
                <div class="upcoming-date">
                    <div class="day">${d.getDate()}</div>
                    <div class="month">${months[d.getMonth()]}</div>
                </div>
                <div class="upcoming-info">
                    <div class="upcoming-name">${e.indicator}</div>
                    <div class="upcoming-detail">
                        Forecast: ${e.forecast !== null ? formatNum(e.forecast) : '—'}
                        &nbsp;|&nbsp; Previous: ${e.previous !== null ? formatNum(e.previous) : '—'}
                    </div>
                </div>
                <span class="upcoming-importance importance-${e.importance}">${e.importance}</span>
            </div>
        `;
    }).join('');
}

// ==========================================================================
// Surprise Chart
// ==========================================================================

async function loadSurpriseChart(indicator) {
    const data = await fetchJSON(`/api/surprises/${indicator}`);
    if (!data || !data.history.length) return;

    const ctx = document.getElementById('surprise-chart').getContext('2d');

    if (surpriseChart) surpriseChart.destroy();

    const labels = data.history.map(h => h.date);
    const values = data.history.map(h => h.zscore);
    const colors = values.map(v => v > 0 ? 'rgba(16, 185, 129, 0.8)' : 'rgba(239, 68, 68, 0.8)');

    surpriseChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: `${indicator} Surprise Z-Score`,
                data: values,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace('0.8', '1')),
                borderWidth: 1,
                borderRadius: 4,
                barPercentage: 0.6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(99, 102, 241, 0.2)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(99, 102, 241, 0.06)' },
                    ticks: { color: '#64748b', font: { size: 11 } },
                },
                y: {
                    grid: { color: 'rgba(99, 102, 241, 0.06)' },
                    ticks: { color: '#64748b', font: { size: 11 } },
                }
            }
        }
    });
}

function switchSurpriseChart(indicator) {
    document.querySelectorAll('#card-surprise-chart .chip').forEach(c => {
        c.classList.toggle('active', c.dataset.indicator === indicator);
    });
    loadSurpriseChart(indicator);
}

// ==========================================================================
// Calendar Page
// ==========================================================================

async function loadCalendar() {
    const data = await fetchJSON('/api/calendar');
    if (!data) return;

    const tbody = document.getElementById('calendar-tbody');
    tbody.innerHTML = data.events.map(e => `
        <tr>
            <td style="font-family: 'JetBrains Mono', monospace;">${formatDateTime(e.scheduled_date)}</td>
            <td style="font-family: 'Inter', sans-serif; font-weight: 600;">${e.indicator}</td>
            <td><span class="upcoming-importance importance-${e.importance}">${e.importance}</span></td>
            <td>${e.forecast !== null ? formatNum(e.forecast) : '—'}</td>
            <td>${e.previous !== null ? formatNum(e.previous) : '—'}</td>
            <td>${e.country}</td>
        </tr>
    `).join('');
}

// ==========================================================================
// Nowcast Page
// ==========================================================================

async function loadNowcast() {
    const data = await fetchJSON('/api/nowcasts');
    if (!data) return;

    // Render nowcast KPI cards
    const cardsEl = document.getElementById('nowcast-cards');
    cardsEl.innerHTML = data.nowcasts.map(n => {
        const diff = n.point_estimate - (n.consensus || n.point_estimate);
        const diffClass = diff > 0 ? 'positive' : diff < 0 ? 'negative' : 'neutral';
        const arrow = diff > 0 ? '↑' : diff < 0 ? '↓' : '→';

        return `
            <div class="kpi-card">
                <div class="kpi-label">${formatIndicator(n.indicator)} Nowcast</div>
                <div class="kpi-value">${formatNum(n.point_estimate)}</div>
                <div class="kpi-change ${diffClass}">
                    ${arrow} ${formatNum(Math.abs(diff))} vs consensus (${formatNum(n.consensus)})
                </div>
            </div>
        `;
    }).join('');

    // Render comparison chart
    renderNowcastChart(data.nowcasts);
}

function renderNowcastChart(nowcasts) {
    const ctx = document.getElementById('nowcast-chart').getContext('2d');
    if (nowcastChart) nowcastChart.destroy();

    const labels = nowcasts.map(n => formatIndicator(n.indicator));

    nowcastChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Nowcast',
                    data: nowcasts.map(n => n.point_estimate),
                    backgroundColor: 'rgba(99, 102, 241, 0.7)',
                    borderColor: 'rgba(99, 102, 241, 1)',
                    borderWidth: 1,
                    borderRadius: 6,
                    barPercentage: 0.35,
                },
                {
                    label: 'Consensus',
                    data: nowcasts.map(n => n.consensus),
                    backgroundColor: 'rgba(148, 163, 184, 0.4)',
                    borderColor: 'rgba(148, 163, 184, 0.8)',
                    borderWidth: 1,
                    borderRadius: 6,
                    barPercentage: 0.35,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#94a3b8', padding: 20, usePointStyle: true, pointStyle: 'rectRounded' }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(99, 102, 241, 0.2)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8', font: { size: 12, weight: '600' } },
                },
                y: {
                    grid: { color: 'rgba(99, 102, 241, 0.06)' },
                    ticks: { color: '#64748b' },
                }
            }
        }
    });
}

// ==========================================================================
// Signals Page
// ==========================================================================

async function loadSignals() {
    const data = await fetchJSON('/api/signals');
    if (!data) return;

    const renderCards = (signals, containerId, isHistorical = false) => {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        if (signals.length === 0) {
            container.innerHTML = `<div class="empty-state">No ${isHistorical ? 'historical' : 'active'} signals.</div>`;
            return;
        }

        container.innerHTML = signals.map(s => {
            const dirClass = s.direction.toLowerCase();
            const confLevel = s.confidence >= 0.7 ? 'high' : s.confidence >= 0.5 ? 'medium' : 'low';
            const statusBadge = isHistorical ? `<span class="badge" style="background: var(--bg-tertiary); color: var(--text-tertiary); margin-left: 8px;">${s.status}</span>` : '';

            return `
                <div class="signal-card ${dirClass} ${isHistorical ? 'historical' : ''}">
                    <div class="signal-header">
                        <div style="display: flex; align-items: center;">
                            <span class="signal-indicator">${formatIndicator(s.indicator)}</span>
                            ${statusBadge}
                        </div>
                        <span class="signal-direction ${s.direction}">${s.direction} ${s.asset}</span>
                    </div>
                    <div class="signal-confidence">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span style="font-size: 0.78rem; color: var(--text-tertiary);">Confidence</span>
                            <span style="font-size: 0.85rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">${(s.confidence * 100).toFixed(0)}%</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill ${confLevel}" style="width: ${s.confidence * 100}%"></div>
                        </div>
                    </div>
                    <div class="signal-reason">${s.reason}</div>
                    <div class="signal-meta">
                        <span><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align: text-top; margin-right: 4px;"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg> Event: ${s.event_time_formatted || formatDate(s.event_time)}</span>
                        <span>Generated: ${formatDateTime(s.generated_at)}</span>
                    </div>
                </div>
            `;
        }).join('');
    };

    renderCards(data.signals, 'signals-container', false);
    renderCards(data.historical, 'historical-signals-container', true);
}

// ==========================================================================
// Backtest Page
// ==========================================================================

async function loadBacktest() {
    const data = await fetchJSON('/api/backtest/summary');
    if (!data) return;

    const m = data.metrics;

    // Render KPIs
    const kpis = document.getElementById('backtest-kpis');
    kpis.innerHTML = `
        <div class="kpi-card">
            <div class="kpi-label">Sharpe Ratio</div>
            <div class="kpi-value">${m.sharpe_ratio.toFixed(2)}</div>
            <div class="kpi-change ${m.sharpe_ratio > 1 ? 'positive' : 'neutral'}">
                ${m.sharpe_ratio > 1.5 ? 'Excellent' : m.sharpe_ratio > 1 ? 'Good' : 'Moderate'}
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Total Return</div>
            <div class="kpi-value">${m.total_return.toFixed(1)}%</div>
            <div class="kpi-change ${m.total_return > 0 ? 'positive' : 'negative'}">
                ${m.annualized_return.toFixed(1)}% annualized
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Max Drawdown</div>
            <div class="kpi-value">${m.max_drawdown.toFixed(1)}%</div>
            <div class="kpi-change ${m.max_drawdown > -10 ? 'positive' : 'negative'}">
                ${m.max_drawdown > -10 ? 'Controlled' : 'High Risk'}
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Win Rate</div>
            <div class="kpi-value">${m.win_rate.toFixed(1)}%</div>
            <div class="kpi-change ${m.win_rate > 50 ? 'positive' : 'negative'}">
                ${m.total_trades} total trades
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Profit Factor</div>
            <div class="kpi-value">${m.profit_factor.toFixed(2)}</div>
            <div class="kpi-change ${m.profit_factor > 1.5 ? 'positive' : 'neutral'}">
                ${m.trades_per_year} trades/year
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Avg Duration</div>
            <div class="kpi-value">${m.avg_trade_duration_minutes}m</div>
            <div class="kpi-change neutral">Event-driven</div>
        </div>
    `;

    document.getElementById('backtest-period').textContent = data.period;

    // Render monthly returns chart
    renderBacktestChart(data.monthly_returns);
}

function renderBacktestChart(monthlyReturns) {
    const ctx = document.getElementById('backtest-chart').getContext('2d');
    if (backtestChart) backtestChart.destroy();

    const labels = monthlyReturns.map(m => m.month);
    const values = monthlyReturns.map(m => m.return);
    const colors = values.map(v => v >= 0 ? 'rgba(16, 185, 129, 0.7)' : 'rgba(239, 68, 68, 0.7)');

    backtestChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Monthly Return %',
                data: values,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace('0.7', '1')),
                borderWidth: 1,
                borderRadius: 6,
                barPercentage: 0.65,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(99, 102, 241, 0.2)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        label: ctx => `${ctx.parsed.y > 0 ? '+' : ''}${ctx.parsed.y.toFixed(1)}%`
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8', font: { size: 11 } },
                },
                y: {
                    grid: { color: 'rgba(99, 102, 241, 0.06)' },
                    ticks: {
                        color: '#64748b',
                        callback: v => `${v > 0 ? '+' : ''}${v}%`
                    },
                }
            }
        }
    });
}

// ==========================================================================
// System Page
// ==========================================================================

async function loadSystem() {
    const data = await fetchJSON('/api/system/status');
    if (!data) return;

    const grid = document.getElementById('system-components');
    const components = data.components;

    const nameMap = {
        database: 'Database',
        fred_api: 'FRED API',
        bls_api: 'BLS API',
        polygon_api: 'Polygon.io',
        trading_economics_api: 'Trading Economics',
        scheduler: 'Scheduler',
        nowcast_model: 'Nowcast Model',
    };

    grid.innerHTML = Object.entries(components).map(([key, comp]) => `
        <div class="system-item">
            <span class="system-dot ${comp.status}"></span>
            <div class="system-info">
                <div class="system-name">${nameMap[key] || key}</div>
                <div class="system-detail">${comp.message || comp.rate_limit || comp.status.replace('_', ' ')}</div>
            </div>
        </div>
    `).join('');

    document.getElementById('system-uptime').textContent = data.uptime || '';
}

// ==========================================================================
// Utilities
// ==========================================================================

function formatNum(val, decimals) {
    if (val === null || val === undefined) return '—';
    const d = decimals !== undefined ? decimals : (Math.abs(val) < 10 ? 1 : 0);
    return Number(val).toLocaleString('en-US', {
        minimumFractionDigits: d,
        maximumFractionDigits: d
    });
}

function formatIndicator(ind) {
    const map = {
        'CPI': 'CPI',
        'CORE_CPI': 'Core CPI',
        'NFP': 'NFP',
        'GDP': 'GDP',
        'PCE': 'PCE',
        'CORE_PCE': 'Core PCE',
        'UNEMPLOYMENT_RATE': 'Unemployment',
        'INITIAL_CLAIMS': 'Jobless Claims',
        'RETAIL_SALES': 'Retail Sales',
        'ISM_MANUFACTURING': 'ISM Mfg',
    };
    return map[ind] || ind;
}

function formatDate(dateStr) {
    const d = new Date(dateStr);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function formatDateTime(dateStr) {
    const d = new Date(dateStr);
    return d.toLocaleDateString('en-US', {
        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
    });
}

// ==========================================================================
// Refresh Button
// ==========================================================================

document.getElementById('btn-refresh').addEventListener('click', () => {
    loadPageData(currentPage);
    const ts = new Date().toLocaleTimeString();
    document.querySelector('#last-updated span').textContent = `Updated ${ts}`;
});

// ==========================================================================
// Initial Load
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
    loadOverview();
});

/**
 * HealthAware AI - Admin Dashboard Client-Side Logic
 * Initializes and dynamically updates Chart.js visualizations for:
 * 1. Most Asked Health Topics (Bar Chart)
 * 2. Intent Distribution (Doughnut Chart)
 * 3. ML vs Bi-LSTM Metrics Benchmark (Grouped Bar Chart)
 * 4. Daily Query Timeline (Line Chart)
 */

let topicsChart = null;
let intentPieChart = null;
let comparisonChart = null;
let usageLineChart = null;

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    loadDashboardData();
});

function initCharts() {
    // 1. Topics Bar Chart
    const ctxTopics = document.getElementById('topicsBarChart');
    if (ctxTopics) {
        topicsChart = new Chart(ctxTopics, {
            type: 'bar',
            data: {
                labels: ['Diabetes', 'Fever', 'Nutrition', 'Hypertension', 'Cold', 'Vaccination', 'Asthma', 'Mental Health'],
                datasets: [{
                    label: 'Query Volume',
                    data: [18, 15, 12, 10, 8, 7, 5, 4],
                    backgroundColor: 'rgba(2, 132, 199, 0.75)',
                    borderColor: '#0284c7',
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: { beginAtZero: true, grid: { color: '#f1f5f9' } },
                    x: { grid: { display: false } }
                }
            }
        });
    }

    // 2. Intent Pie / Doughnut Chart
    const ctxPie = document.getElementById('intentPieChart');
    if (ctxPie) {
        intentPieChart = new Chart(ctxPie, {
            type: 'doughnut',
            data: {
                labels: ['Diabetes', 'Fever', 'Nutrition', 'Hypertension', 'Others'],
                datasets: [{
                    data: [25, 20, 18, 15, 22],
                    backgroundColor: [
                        '#0284c7', '#10b981', '#8b5cf6', '#f59e0b', '#64748b'
                    ],
                    borderWidth: 2,
                    borderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });
    }

    // 3. ML vs DL Comparison Grouped Bar Chart
    const ctxComp = document.getElementById('comparisonChart');
    if (ctxComp) {
        comparisonChart = new Chart(ctxComp, {
            type: 'bar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                datasets: [
                    {
                        label: 'Logistic Regression (ML)',
                        data: [100, 100, 100, 100],
                        backgroundColor: 'rgba(2, 132, 199, 0.8)',
                        borderRadius: 6
                    },
                    {
                        label: 'Bi-LSTM (Deep Learning)',
                        data: [98.5, 98.7, 98.5, 98.6],
                        backgroundColor: 'rgba(139, 92, 246, 0.8)',
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { min: 80, max: 100, grid: { color: '#f1f5f9' } },
                    x: { grid: { display: false } }
                }
            }
        });
    }

    // 4. Daily Usage Line Chart
    const ctxLine = document.getElementById('usageLineChart');
    if (ctxLine) {
        usageLineChart = new Chart(ctxLine, {
            type: 'line',
            data: {
                labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Today'],
                datasets: [{
                    label: 'Queries',
                    data: [12, 19, 14, 25, 22, 30, 15],
                    fill: true,
                    borderColor: '#0d9488',
                    backgroundColor: 'rgba(13, 148, 136, 0.1)',
                    tension: 0.35,
                    pointBackgroundColor: '#0d9488'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, grid: { color: '#f1f5f9' } },
                    x: { grid: { display: false } }
                }
            }
        });
    }
}

async function loadDashboardData() {
    try {
        // Fetch comparison data
        const compRes = await fetch('/api/comparison');
        if (compRes.ok) {
            const compData = await compRes.json();
            updateComparisonUI(compData);
        }

        // Fetch stats data
        const statsRes = await fetch('/api/admin/stats');
        if (statsRes.ok) {
            const stats = await statsRes.json();
            updateStatsUI(stats);
        }
    } catch (err) {
        console.warn('Dashboard data fetch warning:', err);
    }
}

function updateComparisonUI(data) {
    if (!data) return;
    const ml = data.ml || {};
    const dl = data.dl || {};

    if (dl.test_accuracy) {
        const tableAcc = document.getElementById('tableDlAcc');
        const tablePrec = document.getElementById('tableDlPrec');
        const tableRecall = document.getElementById('tableDlRecall');
        const tableF1 = document.getElementById('tableDlF1');
        const tableTime = document.getElementById('tableDlTime');

        if (tableAcc) tableAcc.innerText = `${(dl.test_accuracy * 100).toFixed(2)}%`;
        if (tablePrec) tablePrec.innerText = `${(dl.test_precision * 100).toFixed(2)}%`;
        if (tableRecall) tableRecall.innerText = `${(dl.test_recall * 100).toFixed(2)}%`;
        if (tableF1) tableF1.innerText = `${(dl.test_f1_score * 100).toFixed(2)}%`;
        if (tableTime) tableTime.innerText = `${dl.training_time_seconds}s`;

        // Update comparison chart if loaded
        if (comparisonChart) {
            const mlAcc = ml.test_accuracy ? ml.test_accuracy * 100 : 100;
            const mlPrec = ml.test_precision ? ml.test_precision * 100 : 100;
            const mlRec = ml.test_recall ? ml.test_recall * 100 : 100;
            const mlF1 = ml.test_f1_score ? ml.test_f1_score * 100 : 100;

            const dlAcc = dl.test_accuracy * 100;
            const dlPrec = dl.test_precision * 100;
            const dlRec = dl.test_recall * 100;
            const dlF1 = dl.test_f1_score * 100;

            comparisonChart.data.datasets[0].data = [mlAcc, mlPrec, mlRec, mlF1];
            comparisonChart.data.datasets[1].data = [dlAcc, dlPrec, dlRec, dlF1];
            comparisonChart.update();
        }
    }
}

function updateStatsUI(stats) {
    if (!stats) return;

    // Update topics bar chart if there is real conversation distribution
    if (stats.intent_distribution && stats.intent_distribution.length > 0 && topicsChart) {
        const labels = stats.intent_distribution.map(i => i.intent);
        const counts = stats.intent_distribution.map(i => i.count);
        topicsChart.data.labels = labels;
        topicsChart.data.datasets[0].data = counts;
        topicsChart.update();

        if (intentPieChart) {
            intentPieChart.data.labels = labels.slice(0, 5);
            intentPieChart.data.datasets[0].data = counts.slice(0, 5);
            intentPieChart.update();
        }
    }
}

function refreshAdminStats() {
    loadDashboardData();
}

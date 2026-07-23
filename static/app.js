// State Management
const state = {
  orders: [],
  order_lines: [],
  datasetLoaded: false,
  activeTab: 'revenue',
  charts: {
    revenue: null,
    product: null,
    customer: null
  },
  lastResults: null
};

// Initialization
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initFormControls();
  initEventListeners();
  initCharts();
});

// Tab Navigation
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      
      tab.classList.add('active');
      const tabName = tab.getAttribute('data-tab');
      state.activeTab = tabName;
      document.getElementById(`${tabName}-tab`).classList.add('active');
    });
  });
}

// Controls Logic
function initFormControls() {
  const revenueSlider = document.getElementById('revenueMonths');
  const revenueSliderVal = document.getElementById('revenueMonthsVal');
  if (revenueSlider) {
    revenueSlider.addEventListener('input', (e) => {
      revenueSliderVal.textContent = `${e.target.value} months`;
    });
  }

  const productMode = document.getElementById('productSelectMode');
  const nameGroup = document.getElementById('productNameGroup');
  const stepsGroup = document.getElementById('productStepsGroup');

  if (productMode) {
    productMode.addEventListener('change', (e) => {
      const val = e.target.value;
      nameGroup.style.display = (val === 'single') ? 'flex' : 'none';
      stepsGroup.style.display = (val === 'multistep') ? 'flex' : 'none';
    });
  }
}

// Event Listeners
function initEventListeners() {
  document.getElementById('loadSampleDataBtn').addEventListener('click', loadSampleData);
  
  // Revenue
  document.getElementById('predictRevenueBtn').addEventListener('click', runRevenuePredict);
  document.getElementById('forecastRevenueBtn').addEventListener('click', runRevenueForecast);
  document.getElementById('trainRevenueBtn').addEventListener('click', () => trainModel('/api/revenue/train', 'Revenue Model'));

  // Product
  document.getElementById('runProductBtn').addEventListener('click', runProductAction);
  document.getElementById('trainProductBtn').addEventListener('click', () => trainModel('/api/products/train', 'Product Model'));

  // Customer
  document.getElementById('analyzeCustomersBtn').addEventListener('click', runCustomerAnalysis);
  document.getElementById('trainCustomerBtn').addEventListener('click', () => trainModel('/api/customers/train', 'Customer Model'));

  // Search Table
  document.getElementById('tableSearchInput').addEventListener('input', filterTable);
}

// Sample Data Fetcher
async function loadSampleData() {
  const btn = document.getElementById('loadSampleDataBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Loading Data...';

  try {
    const response = await fetch('/api/sample-data');
    const data = await response.json();

    if (data.status === 'success') {
      state.orders = data.orders;
      state.order_lines = data.order_lines;
      state.datasetLoaded = true;

      document.getElementById('dataStatusDot').classList.add('active');
      document.getElementById('dataStatusText').textContent = '24-Month Dataset Loaded';
      document.getElementById('dataStatusDetails').textContent = `${data.summary.total_orders} Orders | ${data.summary.unique_products} Products`;

      btn.textContent = '✅ Dataset Ready';
      btn.style.background = 'var(--accent-emerald)';
      
      // Populate product search dropdown helper
      if (data.order_lines.length > 0) {
        const firstProd = data.order_lines[0].product_name || data.order_lines[0].product;
        document.getElementById('productNameInput').placeholder = `e.g. ${firstProd}`;
      }
    }
  } catch (err) {
    alert('Failed to load sample dataset: ' + err.message);
    btn.textContent = '⚡ Load Sample Data';
  } finally {
    btn.disabled = false;
  }
}

function checkDataLoaded() {
  if (!state.datasetLoaded) {
    alert('Please click "Load Sample Data" first before running predictions.');
    return false;
  }
  return true;
}

// ── REVENUE ACTIONS ──────────────────────────────────────────────────────────

async function runRevenuePredict() {
  if (!checkDataLoaded()) return;

  try {
    const res = await fetch('/api/revenue/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        orders: state.orders,
        order_lines: state.order_lines
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const val = data.prediction;
    document.getElementById('kpiRevenueNext').textContent = formatCurrency(val);
    document.getElementById('kpiRevenueBadge').textContent = 'Next Month Single Prediction';
    document.getElementById('kpiRevenueAvg').textContent = formatCurrency(val);
    document.getElementById('kpiRevenueTotal').textContent = formatCurrency(val);
    document.getElementById('kpiHorizonSub').textContent = '1 Month Horizon';

    renderTable([
      { Metric: 'Immediate Next Month Predicted Revenue', Value: formatCurrency(val) }
    ]);
  } catch (err) {
    alert('Revenue prediction failed: ' + err.message);
  }
}

async function runRevenueForecast() {
  if (!checkDataLoaded()) return;
  const months = parseInt(document.getElementById('revenueMonths').value);

  try {
    const res = await fetch('/api/revenue/forecast', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        orders: state.orders,
        order_lines: state.order_lines,
        months: months
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const forecasts = data.forecasts;
    const labels = Object.keys(forecasts);
    const values = Object.values(forecasts);

    const total = values.reduce((a, b) => a + b, 0);
    const avg = total / values.length;
    const nextMonthVal = values[0];

    document.getElementById('kpiRevenueNext').textContent = formatCurrency(nextMonthVal);
    document.getElementById('kpiRevenueBadge').textContent = `Step 1 of ${months}`;
    document.getElementById('kpiRevenueAvg').textContent = formatCurrency(avg);
    document.getElementById('kpiRevenueTotal').textContent = formatCurrency(total);
    document.getElementById('kpiHorizonSub').textContent = `${months}-Month Cumulative`;

    updateRevenueChart(labels, values);

    const tableRows = labels.map(dateStr => ({
      'Forecast Month': dateStr,
      'Predicted Revenue': formatCurrency(forecasts[dateStr])
    }));
    renderTable(tableRows);

  } catch (err) {
    alert('Revenue forecast failed: ' + err.message);
  }
}

// ── PRODUCT ACTIONS ──────────────────────────────────────────────────────────

async function runProductAction() {
  if (!checkDataLoaded()) return;
  const mode = document.getElementById('productSelectMode').value;
  
  if (mode === 'all' || mode === 'single') {
    const prodName = (mode === 'single') ? document.getElementById('productNameInput').value.trim() : null;
    try {
      const res = await fetch('/api/products/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          orders: state.orders,
          order_lines: state.order_lines,
          product_name: prodName || null
        })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      const predictions = data.predictions;
      const totalQty = predictions.reduce((sum, item) => sum + item.predicted_demand, 0);
      const topItem = [...predictions].sort((a, b) => b.predicted_demand - a.predicted_demand)[0];

      document.getElementById('kpiProductTotalQty').textContent = `${Math.round(totalQty)} units`;
      document.getElementById('kpiProductTopName').textContent = topItem ? topItem.product_name : 'N/A';
      document.getElementById('kpiProductTopQty').textContent = topItem ? `${Math.round(topItem.predicted_demand)} units` : '0';
      document.getElementById('kpiProductActiveCount').textContent = predictions.length;

      updateProductBarChart(predictions.map(p => p.product_name), predictions.map(p => p.predicted_demand));

      const tableRows = predictions.map(p => ({
        'Product Name': p.product_name,
        'Product ID': p.product_id,
        'Predicted Demand (Units)': Math.round(p.predicted_demand)
      }));
      renderTable(tableRows);

    } catch (err) {
      alert('Product prediction failed: ' + err.message);
    }
  } else if (mode === 'multistep') {
    const steps = parseInt(document.getElementById('productSteps').value);
    try {
      const res = await fetch('/api/products/forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          orders: state.orders,
          order_lines: state.order_lines,
          steps: steps
        })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      const forecasts = data.forecasts;
      document.getElementById('kpiProductActiveCount').textContent = `${steps} Steps`;
      renderTable(forecasts.map(f => ({
        'Step': `Step ${f.horizon_step}`,
        'Product': f.product_name,
        'Forecasted Demand': Math.round(f.predicted_demand)
      })));
    } catch (err) {
      alert('Multi-step product forecast failed: ' + err.message);
    }
  } else if (mode === 'category') {
    try {
      const res = await fetch('/api/products/forecast/category', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          orders: state.orders,
          order_lines: state.order_lines
        })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      const catForecasts = data.category_forecasts;
      updateProductBarChart(catForecasts.map(c => `Category ${c.category_id}`), catForecasts.map(c => c.predicted_category_demand));

      renderTable(catForecasts.map(c => ({
        'Category ID': c.category_id,
        'Active Products': c.active_products,
        'Predicted Category Demand': Math.round(c.predicted_category_demand)
      })));
    } catch (err) {
      alert('Category forecast failed: ' + err.message);
    }
  }
}

// ── CUSTOMER ACTIONS ─────────────────────────────────────────────────────────

async function runCustomerAnalysis() {
  if (!checkDataLoaded()) return;

  try {
    const res = await fetch('/api/customers/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ orders: state.orders })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const customers = data.customer_analysis;
    document.getElementById('kpiCustomerTotal').textContent = customers.length;

    const vips = customers.filter(c => c.segment && c.segment.includes('VIP'));
    const vipRatio = customers.length > 0 ? ((vips.length / customers.length) * 100).toFixed(1) : '0';
    document.getElementById('kpiCustomerVipRatio').textContent = `${vipRatio}%`;
    document.getElementById('kpiCustomerVipCount').textContent = `${vips.length} VIP Customers`;

    const repeats = customers.map(c => c.repeat_probability).filter(p => p !== undefined && p !== null);
    const avgRepeat = repeats.length > 0 ? (repeats.reduce((a, b) => a + b, 0) / repeats.length * 100).toFixed(1) : '0.0';
    document.getElementById('kpiCustomerAvgRepeat').textContent = `${avgRepeat}%`;

    // Segment Distribution
    const segmentCounts = {};
    customers.forEach(c => {
      const seg = c.segment || 'Unassigned';
      segmentCounts[seg] = (segmentCounts[seg] || 0) + 1;
    });

    updateCustomerDoughnutChart(Object.keys(segmentCounts), Object.values(segmentCounts));

    renderTable(customers.map(c => ({
      'Customer': c.customer,
      'Segment': c.segment || 'N/A',
      'Repeat Probability': c.repeat_probability ? `${(c.repeat_probability * 100).toFixed(1)}%` : 'N/A',
      'Activity Score': c.activity_score !== undefined ? c.activity_score : 'N/A'
    })));

  } catch (err) {
    alert('Customer analysis failed: ' + err.message);
  }
}

// ── TRAIN ACTIONS ───────────────────────────────────────────────────────────

async function trainModel(endpoint, modelName) {
  if (!checkDataLoaded()) return;

  if (!confirm(`Trigger hyperparameter tuning and model retraining for ${modelName}? This will fit the model using the current dataset.`)) return;

  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        orders: state.orders,
        order_lines: state.order_lines
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    alert(`🎉 ${modelName} retraining complete!\nSaved to: ${data.saved_path || 'models directory'}`);
  } catch (err) {
    alert(`Training failed: ${err.message}`);
  }
}

// ── CHARTS RENDERING ────────────────────────────────────────────────────────

function initCharts() {
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#9ca3af', font: { family: 'Plus Jakarta Sans' } } }
    },
    scales: {
      x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } },
      y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } }
    }
  };

  // Revenue Chart
  const ctxRev = document.getElementById('revenueChart').getContext('2d');
  state.charts.revenue = new Chart(ctxRev, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Predicted Revenue ($)',
        data: [],
        borderColor: '#6366f1',
        backgroundColor: 'rgba(99, 102, 241, 0.15)',
        fill: true,
        tension: 0.3,
        pointRadius: 4
      }]
    },
    options: chartOptions
  });

  // Product Chart
  const ctxProd = document.getElementById('productChart').getContext('2d');
  state.charts.product = new Chart(ctxProd, {
    type: 'bar',
    data: {
      labels: [],
      datasets: [{
        label: 'Demand (Units)',
        data: [],
        backgroundColor: '#06b6d4',
        borderRadius: 6
      }]
    },
    options: chartOptions
  });

  // Customer Chart
  const ctxCust = document.getElementById('customerChart').getContext('2d');
  state.charts.customer = new Chart(ctxCust, {
    type: 'doughnut',
    data: {
      labels: [],
      datasets: [{
        data: [],
        backgroundColor: ['#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#8b5cf6']
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#9ca3af' } } }
    }
  });
}

function updateRevenueChart(labels, values) {
  if (state.charts.revenue) {
    state.charts.revenue.data.labels = labels;
    state.charts.revenue.data.datasets[0].data = values;
    state.charts.revenue.update();
  }
}

function updateProductBarChart(labels, values) {
  if (state.charts.product) {
    state.charts.product.data.labels = labels;
    state.charts.product.data.datasets[0].data = values;
    state.charts.product.update();
  }
}

function updateCustomerDoughnutChart(labels, values) {
  if (state.charts.customer) {
    state.charts.customer.data.labels = labels;
    state.charts.customer.data.datasets[0].data = values;
    state.charts.customer.update();
  }
}

// ── TABLE RENDERER & SEARCH ──────────────────────────────────────────────────

function renderTable(rows) {
  state.lastResults = rows;
  const headRow = document.getElementById('tableHeadRow');
  const bodyRow = document.getElementById('tableBodyRow');

  if (!rows || rows.length === 0) {
    bodyRow.innerHTML = '<tr><td colspan="4" class="empty-state">No results available.</td></tr>';
    return;
  }

  const columns = Object.keys(rows[0]);
  headRow.innerHTML = columns.map(col => `<th>${col}</th>`).join('');

  populateBody(rows, columns);
}

function populateBody(rows, columns) {
  const bodyRow = document.getElementById('tableBodyRow');
  bodyRow.innerHTML = rows.map(row => `
    <tr>
      ${columns.map(col => `<td>${row[col] !== undefined ? row[col] : ''}</td>`).join('')}
    </tr>
  `).join('');
}

function filterTable(e) {
  if (!state.lastResults) return;
  const q = e.target.value.toLowerCase();
  const filtered = state.lastResults.filter(row => {
    return Object.values(row).some(val => String(val).toLowerCase().includes(q));
  });
  if (state.lastResults.length > 0) {
    populateBody(filtered, Object.keys(state.lastResults[0]));
  }
}

// Helpers
function formatCurrency(val) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val || 0);
}

// script.js — Minimal reports + workflow visualization

let resultsData = null;
let workflowData = null;
let workflowsMap = {};
let currentWorkflowId = null;
let networkInstance = null;
let showEdgeLabels = true;
let useHierarchicalLayout = false;

async function fetchResultsData() {
  try {
    const response = await fetch('data/web_results.json');
    if (!response.ok) throw new Error('File not found or inaccessible');
    return await response.json();
  } catch (err) {
    return null;
  }
}

function renderTable(data) {
  const tableBody = document.querySelector('#results-table tbody');
  const algos = data.algorithms || [];
  tableBody.innerHTML = algos.map(algo => {
    const isSuccess = algo.success_rate === 100;
    return `
    <tr>
      <td><strong>${algo.display_name || algo.name}</strong></td>
      <td>${algo.avg_execution_time?.toFixed(4) ?? '-'}</td>
      <td>$${algo.avg_cost?.toFixed(2) ?? '-'}</td>
      <td>${algo.success_rate?.toFixed(0) ?? '-'}%</td>
      <td><span class="status-badge ${isSuccess ? 'success' : ''}">${isSuccess ? 'Success' : 'Partial'}</span></td>
    </tr>
  `;
  }).join('');
}

async function loadWorkflowData() {
  if (!resultsData || !resultsData.workflows) return null;
  workflowData = resultsData.workflows;
  workflowsMap = {};
  for (const wf of workflowData) workflowsMap[wf.workflow_id] = wf;
  return workflowData;
}

function populateWorkflowSelector() {
  const sel = document.getElementById('workflow-selector');
  if (!workflowData || !sel) return;
  sel.innerHTML = '';
  for (const wf of workflowData) {
    const opt = document.createElement('option');
    opt.value = wf.workflow_id;
    opt.textContent = `${wf.domain ? wf.domain.charAt(0).toUpperCase() + wf.domain.slice(1) : ''} • ${wf.type ? wf.type.replace(/_/g, ' ') : wf.workflow_id}`;
    sel.appendChild(opt);
  }
  currentWorkflowId = workflowData[0]?.workflow_id;
  sel.value = currentWorkflowId;
  sel.addEventListener('change', () => {
    currentWorkflowId = sel.value;
    renderWorkflow(currentWorkflowId);
  });
}

function getNodeStyle(node) {
  // Only minimal styles; feel free to reduce further
  const color = node.is_start ? '#10b981' : node.is_end ? '#ef4444' : '#3b82f6';
  return {
    id: node.id,
    label: node.label,
    color: { background: color, border: '#1e3a8a' },
    shape: node.is_start||node.is_end ? 'diamond' : 'ellipse',
    size: 32,
    borderWidth: 2,
    font: { size: 14, color: '#1e293b', face: 'Segoe UI' },
    title: `<strong>${node.label}</strong><br>Type: ${node.task_type}`
  };
}

function getEdgeStyle(edge) {
  return {
    from: edge.from,
    to: edge.to,
    arrows: 'to',
    color: '#6b7280',
    width: 1,
    label: '',
  };
}

function renderWorkflow(workflowId) {
  const container = document.getElementById('workflow-graph');
  const loading = document.getElementById('workflow-graph-loading');
  if (!workflowsMap[workflowId] || !container) return;
  loading.style.display = 'flex';
  setTimeout(() => {
    const wf = workflowsMap[workflowId];
    if (!wf) return;
    const nodes = new vis.DataSet(wf.nodes.map(getNodeStyle));
    const edges = new vis.DataSet(wf.edges.map(getEdgeStyle));
    if (networkInstance) { networkInstance.destroy(); networkInstance = null; }
    networkInstance = new vis.Network(container, { nodes, edges }, {
      layout: {
        randomSeed: 42,
        improvedLayout: true
      },
      physics: {
        enabled: true,
        stabilization: {
          enabled: true,
          iterations: 200,
          updateInterval: 25
        },
        barnesHut: {
          gravitationalConstant: -8000,
          centralGravity: 0.3,
          springLength: 150,
          springConstant: 0.04,
          damping: 0.09,
          avoidOverlap: 0.5
        }
      },
      nodes: { 
        borderWidth: 2, 
        shadow: {
          enabled: true,
          color: 'rgba(0,0,0,0.1)',
          size: 5,
          x: 2,
          y: 2
        },
        font: { face: 'Segoe UI', align: 'center', color: '#1e293b' } 
      },
      edges: { 
        color: { color: '#6b7280', highlight: '#3b82f6' }, 
        arrows: 'to', 
        smooth: { type: 'cubicBezier', roundness: 0.5 },
        width: 2
      },
      interaction: { 
        hover: true, 
        tooltipDelay: 200, 
        zoomView: true, 
        dragView: true,
        navigationButtons: false,
        keyboard: {
          enabled: false
        }
      }
    });
    
    networkInstance.once('stabilizationIterationsDone', () => {
      networkInstance.setOptions({ physics: { enabled: false } });
      loading.style.display = 'none';
    });
    
    networkInstance.on('afterDrawing', () => {
      if (loading.style.display !== 'none') {
        loading.style.display = 'none';
      }
    });
    
    updateWorkflowInfoPanel(wf);
  }, 20);
}

function updateWorkflowInfoPanel(wf) {
  const info = document.getElementById('workflow-info-panel');
  if (!info) return;
  info.innerHTML = `<strong>${wf.type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</strong><br>
    Domain: <b>${wf.domain}</b><br>
    Nodes: ${wf.metadata.total_nodes}, Edges: ${wf.metadata.total_edges}<br>
    Estimated Cost: $${Number(wf.metadata.estimated_total_cost).toFixed(2)}<br>
    Estimated Time: ${Number(wf.metadata.estimated_total_time_ms).toLocaleString()} ms`;
}

function setupWorkflowControls() {
  document.getElementById('reset-graph-btn').onclick = () => { 
    if (networkInstance) {
      networkInstance.fit({ animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
    }
  };
  document.getElementById('export-png-btn').onclick = () => {
    if (!networkInstance) return;
    try {
      const canvas = networkInstance.canvas.frame.canvas;
      const img = canvas.toDataURL('image/png');
      const a = document.createElement('a');
      a.href = img;
      a.download = `workflow-${currentWorkflowId || 'export'}.png`;
      a.click();
    } catch (e) {
      console.error('Export failed:', e);
    }
  };
}

function renderAnalysis(data) {
  // Best algorithm
  const algos = data.algorithms || [];
  if (algos.length === 0) return;
  
  const bestAlgo = algos.reduce((best, algo) => 
    algo.avg_execution_time < best.avg_execution_time ? algo : best
  );
  
  document.querySelector('.algo-name').textContent = bestAlgo.display_name || bestAlgo.name;
  document.getElementById('best-time').textContent = `${bestAlgo.avg_execution_time.toFixed(4)}s`;
  document.getElementById('best-cost').textContent = `$${bestAlgo.avg_cost.toFixed(2)}`;
  
  // Performance by domain
  const domainPerf = document.getElementById('domain-performance');
  const workflowResults = data.results_by_workflow || [];
  
  const domains = {};
  workflowResults.forEach(wf => {
    if (!domains[wf.domain]) {
      const bestForDomain = wf.results.reduce((best, r) => 
        r.avg_time < best.avg_time ? r : best
      );
      domains[wf.domain] = bestForDomain.algorithm;
    }
  });
  
  domainPerf.innerHTML = Object.entries(domains).map(([domain, algo]) => {
    const algoData = algos.find(a => a.name === algo);
    const displayName = algoData ? algoData.display_name : algo;
    return `
      <div class="domain-item">
        <span class="domain-name">${domain.charAt(0).toUpperCase() + domain.slice(1)}</span>
        <span class="domain-winner">${displayName}</span>
      </div>
    `;
  }).join('');
  
  // Key findings
  const findings = document.getElementById('key-findings');
  const findingsData = [];
  
  // All optimal cost?
  const allSameCost = algos.every(a => Math.abs(a.avg_cost - algos[0].avg_cost) < 0.01);
  if (allSameCost) {
    findingsData.push('All algorithms find optimal solutions with identical costs');
  }
  
  // Speed comparison
  const fastest = algos[0];
  const slowest = algos[algos.length - 1];
  const speedup = (slowest.avg_execution_time / fastest.avg_execution_time).toFixed(1);
  findingsData.push(`${fastest.display_name} is ${speedup}× faster than ${slowest.display_name}`);
  
  // Success rates
  const allSuccess = algos.every(a => a.success_rate === 100);
  if (allSuccess) {
    findingsData.push('100% success rate across all algorithms and workflows');
  }
  
  // Workflow size analysis
  const avgNodes = workflowResults.reduce((sum, wf) => sum + wf.nodes, 0) / workflowResults.length;
  findingsData.push(`Tested across ${workflowResults.length} workflows averaging ${Math.round(avgNodes)} nodes`);
  
  findings.innerHTML = findingsData.map(f => `<li>${f}</li>`).join('');
  
  // Recommendation
  const recommendation = document.getElementById('recommendation');
  const summary = data.summary || {};
  recommendation.innerHTML = `
    <p><strong>${summary.recommendation || 'Use ' + bestAlgo.display_name + ' for best performance'}</strong></p>
    <p style="margin-top: 0.5rem; font-size: 0.9375rem; color: var(--gray-700);">
      ${allSameCost ? 'All algorithms achieve optimal cost. ' : ''}
      Choose ${bestAlgo.display_name} for fastest execution time 
      ${allSuccess ? 'with guaranteed solution quality.' : '.'}
    </p>
  `;
}

function renderConclusion(data) {
  const algos = data.algorithms || [];
  const workflowResults = data.results_by_workflow || [];
  const metadata = data.metadata || {};
  const summary = data.summary || {};
  
  if (algos.length === 0) return;
  
  // Sort algorithms by execution time
  const sortedAlgos = [...algos].sort((a, b) => a.avg_execution_time - b.avg_execution_time);
  const bestAlgo = sortedAlgos[0];
  const worstAlgo = sortedAlgos[sortedAlgos.length - 1];
  
  // Check if all costs are the same
  const allSameCost = algos.every(a => Math.abs(a.avg_cost - algos[0].avg_cost) < 0.01);
  const allSuccess = algos.every(a => a.success_rate === 100);
  
  // Summary text
  const summaryText = document.getElementById('conclusion-summary-text');
  summaryText.innerHTML = `
    <p>This comprehensive benchmark evaluated <strong>${algos.length} graph algorithms</strong> across <strong>${metadata.workflows_tested || workflowResults.length} real-world workflows</strong> from ${(metadata.domains || []).length} business domains. Our testing reveals that ${allSameCost ? 'all algorithms successfully find optimal solutions with identical costs, demonstrating their correctness' : 'solution quality varies across algorithms'}. However, <strong>execution time differs dramatically</strong>, with performance variations of up to ${(worstAlgo.avg_execution_time / bestAlgo.avg_execution_time).toFixed(1)}× between the fastest and slowest algorithms.</p>
    <p>The benchmark achieved a <strong>${metadata.success_rate?.toFixed(1) || '100'}% overall success rate</strong> across ${metadata.total_benchmarks || 'all'} test runs, validating the robustness of these implementations for production use.</p>
  `;
  
  // Performance comparison
  const perfContent = document.getElementById('conclusion-performance');
  perfContent.innerHTML = `
    <div class="performance-comparison">
      ${sortedAlgos.map((algo, idx) => {
        const relativeSpeed = algo.avg_execution_time / bestAlgo.avg_execution_time;
        const isBest = idx === 0;
        const isWorst = idx === sortedAlgos.length - 1;
        return `
          <div class="perf-item ${isBest ? 'best' : ''} ${isWorst ? 'worst' : ''}">
            <span class="perf-rank">${idx + 1}</span>
            <div class="perf-details">
              <strong>${algo.display_name}</strong>
              <div class="perf-metrics">
                <span>${algo.avg_execution_time.toFixed(4)}s</span>
                ${!isBest ? `<span class="perf-relative">(${relativeSpeed.toFixed(1)}× slower)</span>` : '<span class="perf-badge">Fastest</span>'}
              </div>
            </div>
          </div>
        `;
      }).join('')}
    </div>
  `;
  
  // Key takeaways
  const takeaways = document.getElementById('conclusion-takeaways');
  const takeawaysList = [];
  
  if (allSameCost) {
    takeawaysList.push('<li><strong>Solution Optimality:</strong> All algorithms find the optimal path with identical costs, confirming their correctness.</li>');
  }
  
  takeawaysList.push(`<li><strong>Performance Winner:</strong> ${bestAlgo.display_name} delivers the fastest execution time at ${bestAlgo.avg_execution_time.toFixed(4)}s average.</li>`);
  
  if (worstAlgo.avg_execution_time / bestAlgo.avg_execution_time > 100) {
    takeawaysList.push(`<li><strong>Performance Gap:</strong> Significant ${(worstAlgo.avg_execution_time / bestAlgo.avg_execution_time).toFixed(0)}× difference highlights the importance of algorithm selection.</li>`);
  } else {
    takeawaysList.push(`<li><strong>Efficiency Matters:</strong> Choosing the right algorithm can improve performance by ${(worstAlgo.avg_execution_time / bestAlgo.avg_execution_time).toFixed(1)}×.</li>`);
  }
  
  if (allSuccess) {
    takeawaysList.push('<li><strong>Production Ready:</strong> 100% success rate demonstrates reliability across diverse workflow types.</li>');
  }
  
  const domains = [...new Set(workflowResults.map(wf => wf.domain))];
  takeawaysList.push(`<li><strong>Domain Coverage:</strong> Tested across ${domains.length} domains: ${domains.join(', ')}.</li>`);
  
  takeaways.innerHTML = takeawaysList.join('');
  
  // Conclusion metadata
  const conclusionMeta = document.getElementById('conclusion-meta');
  const generatedDate = metadata.generated_at ? new Date(metadata.generated_at).toLocaleDateString('en-US', { 
    year: 'numeric', 
    month: 'long', 
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  }) : 'N/A';
  
  conclusionMeta.innerHTML = `
    <div class="meta-items">
      <span class="meta-item"><strong>Generated:</strong> ${generatedDate}</span>
      <span class="meta-item"><strong>Total Benchmarks:</strong> ${metadata.total_benchmarks || 'N/A'}</span>
      <span class="meta-item"><strong>Algorithms Tested:</strong> ${algos.length}</span>
      <span class="meta-item"><strong>Workflows:</strong> ${metadata.workflows_tested || workflowResults.length}</span>
    </div>
  `;
}

async function initMinimalSite() {
  resultsData = await fetchResultsData();
  if (!resultsData) return;
  renderTable(resultsData);
  renderAnalysis(resultsData);
  renderConclusion(resultsData);
  await loadWorkflowData();
  populateWorkflowSelector();
  setupWorkflowControls();
  renderWorkflow(currentWorkflowId);
}

window.addEventListener('DOMContentLoaded', initMinimalSite);

// Minimal JS for micro-interactions
function drillToCluster(idx){
  window.location.href = '/cluster/' + idx;
}

function exportCluster(idx){
  window.location.href = '/export_cluster?idx=' + encodeURIComponent(idx);
}

// If there are manual/csv toggles in the templates, this will handle them gracefully
document.addEventListener('DOMContentLoaded', function () {
  const manualBtn = document.getElementById('manualBtn');
  const csvBtn = document.getElementById('csvBtn');
  const manualForm = document.getElementById('manualForm');
  const csvForm = document.getElementById('csvForm');
  if (manualBtn && csvBtn && manualForm && csvForm) {
    manualBtn.addEventListener('click', () => {
      manualBtn.classList.add('active');
      csvBtn.classList.remove('active');
      manualForm.classList.remove('hidden');
      csvForm.classList.add('hidden');
    });
    csvBtn.addEventListener('click', () => {
      csvBtn.classList.add('active');
      manualBtn.classList.remove('active');
      csvForm.classList.remove('hidden');
      manualForm.classList.add('hidden');
    });
  }
});

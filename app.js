const toast = document.querySelector('#toast');
function notify(message){ toast.textContent=message; toast.classList.add('show'); setTimeout(()=>toast.classList.remove('show'),3200); }
document.querySelectorAll('.select-btn').forEach(btn=>btn.addEventListener('click',()=>{document.querySelectorAll('.select-btn').forEach(b=>{b.classList.remove('selected');b.textContent='Select option'});btn.classList.add('selected');btn.innerHTML='Selected by agent <span>✓</span>';notify(`Option ${btn.dataset.option} selected for review.`)}));
document.querySelectorAll('#approve,#approveTop').forEach(btn=>btn.addEventListener('click',()=>{notify('Recommendation approved — proposal draft generated.');document.querySelector('.status-pill').innerHTML='<span class="dot"></span> Proposal ready';document.querySelector('.status-pill').style.color='#3d705e';}));
document.querySelector('#edit').addEventListener('click',()=>notify('Scope editor opened — human edits are ready to capture.'));
document.querySelector('#showHistory').addEventListener('click',()=>notify('Change history: 3 state updates since the first enquiry.'));

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const modalElement = document.getElementById('resultModal');
    const resultModal = new bootstrap.Modal(modalElement);
    
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultContent = document.getElementById('resultContent');
    
    // Result elements
    const iconContainer = document.getElementById('iconContainer');
    const resultTitle = document.getElementById('resultTitle');
    const resultMessage = document.getElementById('resultMessage');

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show modal and loading state
        resultModal.show();
        loadingSpinner.classList.remove('d-none');
        resultContent.classList.add('d-none');
        
        // Gather data
        const formData = {
            age: document.getElementById('age').value,
            gender: document.getElementById('gender').value,
            height: document.getElementById('height').value,
            weight: document.getElementById('weight').value,
            ap_hi: document.getElementById('ap_hi').value,
            ap_lo: document.getElementById('ap_lo').value,
            cholesterol: document.getElementById('cholesterol').value,
            gluc: document.getElementById('gluc').value,
            smoke: document.getElementById('smoke').checked ? 1 : 0,
            alco: document.getElementById('alco').checked ? 1 : 0,
            active: document.getElementById('active').checked ? 1 : 0
        };

        // Simulate API call delay for UX
        setTimeout(() => {
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData),
            })
            .then(response => response.json())
            .then(data => {
                loadingSpinner.classList.add('d-none');
                resultContent.classList.remove('d-none');
                
                if (data.error) {
                    showError(data.error);
                } else {
                    showResult(data);
                }
            })
            .catch((error) => {
                loadingSpinner.classList.add('d-none');
                resultContent.classList.remove('d-none');
                showError('Something went wrong. Please try again.');
                console.error('Error:', error);
            });
        }, 800); 
    });

    function showResult(data) {
        // Clear previous icon
        iconContainer.innerHTML = '';
        
        if (data.prediction === 1) {
            // Risk
            iconContainer.innerHTML = '<i class="fa-solid fa-heart-crack result-icon-danger"></i>';
            resultTitle.textContent = 'High Risk Detected';
            resultTitle.className = 'fw-bold mb-2 text-danger';
            resultMessage.textContent = 'The model predicts a high probability of cardiovascular disease based on the provided metrics. Please consult a cardiologist.';
        } else {
            // Healthy
            iconContainer.innerHTML = '<i class="fa-solid fa-heart-circle-check result-icon-safe"></i>';
            resultTitle.textContent = 'Health Metrics Look Good';
            resultTitle.className = 'fw-bold mb-2 text-success';
            resultMessage.textContent = 'The model predicts a low risk of cardiovascular disease. Keep up the healthy lifestyle!';
        }
    }

    function showError(msg) {
        iconContainer.innerHTML = '<i class="fa-solid fa-triangle-exclamation text-warning" style="font-size: 4rem;"></i>';
        resultTitle.textContent = 'Error';
        resultTitle.className = 'fw-bold mb-2 text-warning';
        resultMessage.textContent = msg;
    }
});

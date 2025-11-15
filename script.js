
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    const diagnosisSection = document.getElementById('diagnosis-section');
    const riskSection = document.getElementById('risk-section');
    const recommendationsList = document.getElementById('recommendations-list');
    const apiBase = window.location.origin.startsWith('file')
        ? 'http://127.0.0.1:5000'
        : window.location.origin;
    const apiEndpoint = `${apiBase}/predict`;

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const submitButton = form.querySelector('button[type="submit"]');
        submitButton.disabled = true;
        submitButton.textContent = 'جاري التوقع...';

        const formData = {
            age: parseFloat(document.getElementById('age').value),
            bmi: parseFloat(document.getElementById('bmi').value),
            HbA1c_level: parseFloat(document.getElementById('HbA1c_level').value),
            blood_glucose_level: parseFloat(document.getElementById('blood_glucose_level').value),
            hypertension: document.getElementById('hypertension').checked ? 1 : 0,
            heart_disease: document.getElementById('heart_disease').checked ? 1 : 0,
        };

        try {
            const response = await fetch(apiEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Server responded with status: ${response.status}`);
            }

            const result = await response.json();
            displayResult(result);

        } catch (error) {
            console.error('Error:', error);
            displayError(error.message);
        } finally {
            submitButton.disabled = false;
            submitButton.textContent = 'توقع الآن';
        }
    });

    function displayResult(result) {
        resultContainer.classList.remove('hidden', 'result-diabetic', 'result-not-diabetic');
        recommendationsList.innerHTML = ''; // Clear previous recommendations

        // Display Diagnosis
        diagnosisSection.textContent = `التشخيص: ${result.diagnosis}`;
        diagnosisSection.className = result.prediction === 1 ? 'diagnosis-diabetic' : 'diagnosis-not-diabetic';

        // Display Risk Level
        riskSection.textContent = `مستوى الخطورة: ${result.risk_level}`;
        riskSection.className = `risk-${result.risk_level}`;

        // Display Recommendations
        result.recommendations.forEach(rec => {
            const li = document.createElement('li');
            li.textContent = rec;
            recommendationsList.appendChild(li);
        });

        // Style container border
        resultContainer.classList.add(result.prediction === 1 ? 'result-diabetic' : 'result-not-diabetic');
    }

    function displayError(errorMessage) {
        resultContainer.classList.remove('hidden');
        resultContainer.classList.add('result-diabetic');
        diagnosisSection.textContent = `خطأ: ${errorMessage}. يرجى التأكد من أن الخادم الخلفي يعمل.`;
        riskSection.textContent = '';
        recommendationsList.innerHTML = '';
    }
});

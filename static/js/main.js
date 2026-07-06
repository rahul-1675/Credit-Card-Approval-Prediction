document.addEventListener('DOMContentLoaded', () => {
    // 1. Dynamic Slider Indicators
    const rangeSliders = [
        { id: 'Age', displayId: 'val-Age', suffix: ' years' },
        { id: 'Employment_Duration_Years', displayId: 'val-Employment_Duration_Years', suffix: ' years' },
        { id: 'Annual_Income', displayId: 'val-Annual_Income', prefix: '$', format: true },
        { id: 'Loan_Amount', displayId: 'val-Loan_Amount', prefix: '$', format: true },
        { id: 'Existing_Debt', displayId: 'val-Existing_Debt', prefix: '$', format: true },
        { id: 'Credit_Score', displayId: 'val-Credit_Score' },
        { id: 'Credit_Inquiries', displayId: 'val-Credit_Inquiries' }
    ];

    rangeSliders.forEach(sliderInfo => {
        const slider = document.getElementById(sliderInfo.id);
        const display = document.getElementById(sliderInfo.displayId);
        
        if (slider && display) {
            const updateDisplay = () => {
                let val = parseFloat(slider.value);
                let text = '';
                
                if (sliderInfo.format) {
                    text = val.toLocaleString('en-US', { maximumFractionDigits: 0 });
                } else {
                    text = val.toString();
                }
                
                if (sliderInfo.prefix) text = sliderInfo.prefix + text;
                if (sliderInfo.suffix) text = text + sliderInfo.suffix;
                
                display.textContent = text;
            };
            
            slider.addEventListener('input', updateDisplay);
            updateDisplay(); // Init
        }
    });

    // 2. Form Validations
    const applyForm = document.getElementById('applyForm');
    if (applyForm) {
        applyForm.addEventListener('submit', (e) => {
            const age = parseInt(document.getElementById('Age').value);
            const income = parseFloat(document.getElementById('Annual_Income').value);
            const loan = parseFloat(document.getElementById('Loan_Amount').value);
            const debt = parseFloat(document.getElementById('Existing_Debt').value);
            const creditScore = parseInt(document.getElementById('Credit_Score').value);
            
            let errors = [];
            
            if (age < 18 || age > 100) {
                errors.push("Applicant age must be between 18 and 100 years.");
            }
            if (income <= 0) {
                errors.push("Annual Income must be greater than $0.");
            }
            if (loan <= 0) {
                errors.push("Requested Loan Amount must be greater than $0.");
            }
            if (debt < 0) {
                errors.push("Existing Debt cannot be negative.");
            }
            if (creditScore < 300 || creditScore > 850) {
                errors.push("Credit Score must be a valid FICO score (300 to 850).");
            }
            
            if (errors.length > 0) {
                e.preventDefault();
                alert("Validation Errors:\n\n" + errors.join("\n"));
            }
        });
    }

    // 3. Animate the Approval Probability Gauge
    const gaugeFill = document.getElementById('gauge-fill');
    if (gaugeFill) {
        const probVal = parseFloat(gaugeFill.dataset.probability);
        // Radius of circle is 80, circumference = 2 * PI * 80 = ~502.65
        const circumference = 2 * Math.PI * 80;
        
        gaugeFill.style.strokeDasharray = circumference;
        gaugeFill.style.strokeDashoffset = circumference; // Start fully empty
        
        // Trigger reflow to restart transition
        void gaugeFill.getBoundingClientRect();
        
        // Calculate offset (e.g. 80% full -> offset is 20% of circumference)
        const offset = circumference * (1 - probVal);
        gaugeFill.style.strokeDashoffset = offset;
    }

    // 4. API Interactive Try-It Section
    const tryApiForm = document.getElementById('tryApiForm');
    const apiResponsePre = document.getElementById('apiResponse');
    
    if (tryApiForm && apiResponsePre) {
        tryApiForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            apiResponsePre.textContent = 'Sending request...';
            
            const formData = new FormData(tryApiForm);
            const dataObj = {};
            
            // Map formData into JSON structure
            formData.forEach((value, key) => {
                // Parse numbers where possible
                if (['Age', 'Credit_Score', 'Credit_History', 'Credit_Inquiries'].includes(key)) {
                    dataObj[key] = parseInt(value);
                } else if (['Employment_Duration_Years', 'Annual_Income', 'Loan_Amount', 'Existing_Debt'].includes(key)) {
                    dataObj[key] = parseFloat(value);
                } else {
                    dataObj[key] = value;
                }
            });
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(dataObj)
                });
                
                const result = await response.json();
                apiResponsePre.textContent = JSON.stringify(result, null, 2);
            } catch (err) {
                apiResponsePre.textContent = `Error calling API:\n${err.message}`;
            }
        });
    }
});

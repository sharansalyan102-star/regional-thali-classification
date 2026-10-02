// ============================================================
// REGIONAL THALI CLASSIFICATION - FRONTEND SCRIPT
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    
    // ------------------------------------------------------------
    // DOM ELEMENTS
    // ------------------------------------------------------------
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    
    const emptyPreviewState = document.getElementById('empty-preview-state');
    const activePreviewState = document.getElementById('active-preview-state');
    const imagePreview = document.getElementById('image-preview');
    const previewFilename = document.getElementById('preview-filename');
    
    const classifyBtn = document.getElementById('classify-btn');
    const removeBtn = document.getElementById('remove-btn');
    
    const loadingState = document.getElementById('loading-state');
    const resultContainer = document.getElementById('result-container');
    
    const nonFoodAlert = document.getElementById('non-food-alert');
    const nonFoodTitle = document.getElementById('non-food-title');
    const nonFoodBadge = document.getElementById('non-food-badge');
    const nonFoodMessage = document.getElementById('non-food-message');

    const lowConfidenceAlert = document.getElementById('low-confidence-alert');
    const lowConfidenceTitle = document.getElementById('low-confidence-title');
    const lowConfidenceBadge = document.getElementById('low-confidence-badge');
    const lowConfidenceMessage = document.getElementById('low-confidence-message');

    const predictionResult = document.getElementById('prediction-result');
    const thaliInfoCard = document.getElementById('thali-info-card');
    
    const predictedClassName = document.getElementById('predicted-class-name');
    const predictedRegion = document.getElementById('predicted-region');
    const confidenceBadge = document.getElementById('confidence-badge');
    
    // Probability elements
    const probBars = {
        'Bengali Thali': {
            bar: document.getElementById('prob-bengali-bar'),
            val: document.getElementById('prob-bengali-val')
        },
        'Kerala Sadhya': {
            bar: document.getElementById('prob-kerala-bar'),
            val: document.getElementById('prob-kerala-val')
        },
        'Rajasthani Thali': {
            bar: document.getElementById('prob-rajasthani-bar'),
            val: document.getElementById('prob-rajasthani-val')
        }
    };
    
    // Thali Info elements
    const infoDescription = document.getElementById('info-description');
    const infoIngredients = document.getElementById('info-ingredients');
    const infoDishes = document.getElementById('info-dishes');
    const infoCharacteristics = document.getElementById('info-characteristics');

    let selectedFile = null;

    // ------------------------------------------------------------
    // EVENT LISTENERS: FILE PICKER & DROPZONE
    // ------------------------------------------------------------
    
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Drag & Drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files[0]) {
            handleFileSelect(files[0]);
        }
    });

    // ------------------------------------------------------------
    // FILE HANDLING & PREVIEW
    // ------------------------------------------------------------
    
    function handleFileSelect(file) {
        if (!file.type.match('image.*')) {
            alert('Please select a valid image file (JPG, JPEG, PNG, WEBP).');
            return;
        }

        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            if (previewFilename) previewFilename.textContent = file.name;
            
            // UI state transition
            emptyPreviewState.classList.add('hidden');
            activePreviewState.classList.remove('hidden');
            resultContainer.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    removeBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        
        activePreviewState.classList.add('hidden');
        emptyPreviewState.classList.remove('hidden');
        resultContainer.classList.add('hidden');
    });

    // ------------------------------------------------------------
    // PREDICTION / API CLASSIFICATION CALL
    // ------------------------------------------------------------
    
    classifyBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI loading state
        activePreviewState.classList.add('hidden');
        loadingState.classList.remove('hidden');
        resultContainer.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            loadingState.classList.add('hidden');
            activePreviewState.classList.remove('hidden');

            if (data.success) {
                renderResults(data);
            } else {
                alert('Classification Error: ' + (data.error || 'Unknown error occurred.'));
            }

        } catch (error) {
            console.error('Fetch error:', error);
            loadingState.classList.add('hidden');
            activePreviewState.classList.remove('hidden');
            alert('Failed to connect to the classification server. Make sure app.py is running.');
        }
    });

    // ------------------------------------------------------------
    // RENDER CLASSIFICATION RESULTS & METADATA
    // ------------------------------------------------------------
    
    function renderResults(data) {
        resultContainer.classList.remove('hidden');

        // CASE 1: STAGE 1 NON-FOOD REJECTION (< 50% Food Probability)
        if (data.is_food === false) {
            nonFoodAlert.classList.remove('hidden');
            lowConfidenceAlert.classList.add('hidden');
            predictionResult.classList.add('hidden');

            nonFoodTitle.textContent = data.prediction || 'Not a Recognized Thali';
            if (nonFoodBadge && typeof data.food_probability !== 'undefined') {
                nonFoodBadge.textContent = `Food Likelihood: ${data.food_probability.toFixed(1)}%`;
            }
            nonFoodMessage.textContent = data.message || 'This image does not appear to be a recognized thali. Please upload a clear image of a Bengali Thali, Kerala Sadhya, or Rajasthani Thali.';

        } 
        // CASE 2: STAGE 2 LOW CONFIDENCE (< 80% Confidence)
        else if (data.is_low_confidence === true) {
            nonFoodAlert.classList.add('hidden');
            lowConfidenceAlert.classList.remove('hidden');
            predictionResult.classList.remove('hidden');

            lowConfidenceTitle.textContent = data.prediction || "Low Classification Confidence";
            lowConfidenceBadge.textContent = `Uncertain (${data.confidence.toFixed(1)}% < 80%)`;
            lowConfidenceMessage.textContent = data.message || "The image may contain food, but the model is not confident enough (confidence < 80%) to identify the regional thali. Please upload a clearer image showing the complete thali.";

            // In prediction result card: DO NOT display the class name when confidence < 80%!
            predictedClassName.textContent = "Uncertain Thali Prediction";
            confidenceBadge.textContent = `${data.confidence.toFixed(1)}% Confidence (< 80%)`;
            confidenceBadge.className = "pill-badge pill-warning";
            predictedRegion.textContent = "Confidence is below 80% threshold — No region assigned";

            // Hide specific thali cuisine info card so no thali is claimed as confident
            if (thaliInfoCard) thaliInfoCard.classList.add('hidden');

            // Render probability breakdown bars for Bengali, Kerala, Rajasthani
            const probs = data.probabilities || {};
            Object.keys(probBars).forEach(clsName => {
                const item = probBars[clsName];
                const val = probs[clsName] || 0.0;
                item.val.textContent = `${val.toFixed(1)}%`;
                item.bar.style.width = `${Math.max(val, 2.0)}%`;
                const parentItem = item.val.closest('.prob-item');
                if (parentItem) parentItem.classList.remove('highlight');
            });

        } 
        // CASE 3: STAGE 2 HIGH CONFIDENCE (>= 80% Confidence)
        else {
            nonFoodAlert.classList.add('hidden');
            lowConfidenceAlert.classList.add('hidden');
            predictionResult.classList.remove('hidden');
            if (thaliInfoCard) thaliInfoCard.classList.remove('hidden');

            predictedClassName.textContent = data.prediction;
            confidenceBadge.textContent = `${data.confidence.toFixed(1)}% Confidence`;
            confidenceBadge.className = "pill-badge pill-success";

            const thaliInfo = data.thali_info || {};
            predictedRegion.textContent = `Region: ${thaliInfo.region || 'India'}`;

            // Update Progress Bars & Values
            const probs = data.probabilities || {};
            Object.keys(probBars).forEach(clsName => {
                const item = probBars[clsName];
                const val = probs[clsName] || 0.0;
                item.val.textContent = `${val.toFixed(1)}%`;
                item.bar.style.width = `${Math.max(val, 2.0)}%`;
                const parentItem = item.val.closest('.prob-item');
                if (parentItem) {
                    if (data.prediction.toLowerCase().includes(clsName.toLowerCase().split(' ')[0])) {
                        parentItem.classList.add('highlight');
                    } else {
                        parentItem.classList.remove('highlight');
                    }
                }
            });

            // Populate Thali Details Card
            infoDescription.textContent = thaliInfo.description || 'N/A';
            infoCharacteristics.textContent = thaliInfo.characteristics || 'N/A';

            // Common Ingredients Pills
            infoIngredients.innerHTML = '';
            (thaliInfo.common_ingredients || []).forEach(item => {
                const pill = document.createElement('span');
                pill.className = 'tag-pill';
                pill.textContent = item;
                infoIngredients.appendChild(pill);
            });

            // Typical Dishes Pills
            infoDishes.innerHTML = '';
            (thaliInfo.typical_dishes || []).forEach(dish => {
                const pill = document.createElement('span');
                pill.className = 'tag-pill';
                pill.textContent = dish;
                infoDishes.appendChild(pill);
            });
        }

        // Smooth scroll to result
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // ------------------------------------------------------------
    // SMOOTH SCROLL NAVIGATION HIGHLIGHTS
    // ------------------------------------------------------------
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            const targetId = link.getAttribute('href');
            if (targetId.startsWith('#')) {
                navLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');
            }
        });
    });
});

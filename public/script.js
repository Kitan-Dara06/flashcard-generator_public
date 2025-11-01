document.addEventListener('DOMContentLoaded', function() {
    const passwordOverlay = document.getElementById('passwordOverlay');
    const passwordInput = document.getElementById('passwordInput');
    const passwordSubmit = document.getElementById('passwordSubmit');
    const passwordError = document.getElementById('passwordError');
    const isAuthenticated = sessionStorage.getItem('authenticated') === 'true';
    if (isAuthenticated) {
        passwordOverlay.classList.add('hidden');
    }
    passwordSubmit.addEventListener('click', checkPassword);
    passwordInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            checkPassword();
        }
    });
    async function checkPassword() {
        const enteredPassword = passwordInput.value;
        try {
            const response = await fetch('/api/verify_password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },

                body: JSON.stringify({
                    password: enteredPassword
                })
            });
            const data = await response.json();
            if (data.success && data.authenticated) {

                sessionStorage.setItem('authenticated', 'true');

                passwordOverlay.classList.add('hidden');

                passwordError.classList.add('hidden');

                passwordInput.value = '';

            } else {

                passwordError.classList.remove('hidden');

                passwordInput.value = '';

                passwordInput.focus();

            }

        } catch (error) {

            passwordError.textContent = '❌ Error verifying password';

            passwordError.classList.remove('hidden');

            passwordInput.value = '';

        }

    }

   
    const fileInput = document.getElementById('fileInput');
    const generateBtn = document.getElementById('generateBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const error = document.getElementById('error');
    const flashcardsContainer = document.getElementById('flashcards');
    const resultsCount = document.getElementById('resultsCount');
    const downloadBtn = document.getElementById('downloadBtn');
    
    let currentFlashcards = [];

    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            generateBtn.disabled = false;
            generateBtn.textContent = `🧠 Generate from ${file.name}`;
        } else {
            generateBtn.disabled = true;
            generateBtn.textContent = '🧠 Generate Flashcards';
        }
    });

    generateBtn.addEventListener('click', async function() {
        const file = fileInput.files[0];
        if (!file) return;

        showLoading();
        
        try {
            const base64File = await fileToBase64(file);
            
            const response = await fetch('/api/generate_flashcards', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    file_content: base64File.split(',')[1], // Remove data:mime;base64, prefix
                    file_type: file.type
                })
            });

            const data = await response.json();
            
            if (data.success) {
                currentFlashcards = data.flashcards;
                showResults(data.flashcards);
            } else {
                showError(data.error || 'Failed to generate flashcards');
            }
        } catch (err) {
            showError('Failed to generate flashcards: ' + err.message);
        }
    });

    downloadBtn.addEventListener('click', function() {
        if (currentFlashcards.length === 0) return;
        
        const dataStr = JSON.stringify(currentFlashcards, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = 'flashcards.json';
        link.click();
    });

    function fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
        });
    }

    function showLoading() {
        loading.classList.remove('hidden');
        results.classList.add('hidden');
        error.classList.add('hidden');
    }

    function showResults(flashcards) {
        loading.classList.add('hidden');
        results.classList.remove('hidden');
        error.classList.add('hidden');
        
        resultsCount.textContent = `✅ Generated ${flashcards.length} flashcards!`;
        
        flashcardsContainer.innerHTML = '';
        flashcards.forEach((card, index) => {
            const flashcardEl = createFlashcardElement(card, index + 1);
            flashcardsContainer.appendChild(flashcardEl);
        });
    }

    function showError(message) {
        loading.classList.add('hidden');
        results.classList.add('hidden');
        error.classList.remove('hidden');
        error.textContent = '❌ ' + message;
    }

    function createFlashcardElement(card, number) {
        const flashcard = document.createElement('div');
        flashcard.className = 'flashcard';
        
        flashcard.innerHTML = `
            <div class="flashcard-inner" onclick="this.classList.toggle('flip')">
                <div class="flashcard-front">
                    <div class="flashcard-number">Q${number}</div>
                    <div>${escapeHtml(card.question)}</div>
                </div>
                <div class="flashcard-back">
                    <div class="flashcard-number">A${number}</div>
                    <div>${escapeHtml(card.answer)}</div>
                </div>
            </div>
        `;
        
        return flashcard;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML.replace(/\n/g, '<br>');
    }
});

document.addEventListener('DOMContentLoaded', () => {
    // Στοιχεία DOM
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('snapshot-canvas');
    const cameraPlaceholder = document.getElementById('camera-placeholder');
    const cameraStatus = document.getElementById('camera-status');
    const scanOverlay = document.getElementById('scan-overlay');
    const btnToggleCamera = document.getElementById('btn-toggle-camera');
    const btnCapture = document.getElementById('btn-capture');

    // Αποτελέσματα Συναισθήματος
    const placeholderState = document.querySelector('.placeholder-state');
    const activeEmotion = document.getElementById('active-emotion');
    const dominantEmoji = document.getElementById('dominant-emoji');
    const dominantName = document.getElementById('dominant-name');
    const dominantConf = document.getElementById('dominant-conf');
    const breakdownList = document.getElementById('breakdown-list');

    // Προτάσεις Ταινιών
    const recSection = document.getElementById('recommendations-section');
    const moviesGrid = document.getElementById('movies-grid');
    const btnRefresh = document.getElementById('btn-refresh-movies');

    // Χειροκίνητα κουμπιά
    const quickEmoButtons = document.querySelectorAll('.btn-emo');

    let stream = null;
    let isCameraRunning = false;
    let currentEmotion = null;

    // 1. Διαχείριση Κάμερας (Start / Stop)
    btnToggleCamera.addEventListener('click', async () => {
        if (!isCameraRunning) {
            await startCamera();
        } else {
            stopCamera();
        }
    });

    async function startCamera() {
        try {
            cameraStatus.textContent = "Σύνδεση...";
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                },
                audio: false
            });

            video.srcObject = stream;
            video.style.display = 'block';
            cameraPlaceholder.style.display = 'none';

            isCameraRunning = true;
            btnToggleCamera.innerHTML = '<span class="btn-icon">⏹️</span> Διακοπή Κάμερας';
            btnToggleCamera.classList.remove('btn-secondary');
            btnToggleCamera.classList.add('btn-outline');
            btnCapture.disabled = false;
            cameraStatus.textContent = "Ενεργή";
            cameraStatus.classList.add('active');
        } catch (err) {
            console.error("Σφάλμα πρόσβασης στην κάμερα:", err);
            alert("Δεν ήταν δυνατή η πρόσβαση στην κάμερα. Βεβαιώσου ότι έχεις δώσει άδεια στον browser!");
            cameraStatus.textContent = "Σφάλμα";
        }
    }

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
        video.style.display = 'none';
        cameraPlaceholder.style.display = 'flex';
        isCameraRunning = false;
        btnToggleCamera.innerHTML = '<span class="btn-icon">▶️</span> Έναρξη Κάμερας';
        btnToggleCamera.classList.add('btn-secondary');
        btnCapture.disabled = true;
        cameraStatus.textContent = "Ανενεργή";
        cameraStatus.classList.remove('active');
    }

    // 2. Λήψη Snapshot και Αποστολή στο Flask Backend
    btnCapture.addEventListener('click', async () => {
        if (!isCameraRunning) return;

        // Ορισμός διαστάσεων canvas
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');

        // Σχεδιασμός τρέχοντος frame στο canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Μετατροπή σε Base64 JPEG
        const imageData = canvas.toDataURL('image/jpeg', 0.85);

        // Έλεγχος στρατηγικής
        const strategy = document.querySelector('input[name="strategy"]:checked').value;

        // UI Feedback: Έναρξη σάρωσης
        scanOverlay.style.display = 'block';
        btnCapture.disabled = true;
        btnCapture.textContent = "Ανάλυση...";

        try {
            const res = await fetch('/api/analyze-and-recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData, strategy: strategy })
            });

            const data = await res.json();
            if (data.success) {
                renderResults(data);
            } else {
                alert(data.error || "Προέκυψε σφάλμα κατά την ανάλυση.");
            }
        } catch (err) {
            console.error("API error:", err);
            alert("Αποτυχία επικοινωνίας με τον server.");
        } finally {
            scanOverlay.style.display = 'none';
            btnCapture.disabled = false;
            btnCapture.innerHTML = '<span class="btn-icon">⚡</span> Ανάλυση Διάθεσης';
        }
    });

    // 3. Χειροκίνητη επιλογή συναισθήματος (χωρίς κάμερα)
    quickEmoButtons.forEach(btn => {
        btn.addEventListener('click', async () => {
            const emo = btn.getAttribute('data-emo');
            const strategy = document.querySelector('input[name="strategy"]:checked').value;

            try {
                const res = await fetch('/api/recommend-manual', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ emotion: emo, strategy: strategy })
                });
                const data = await res.json();
                if (data.success) {
                    renderResults(data);
                }
            } catch (err) {
                console.error("API error:", err);
            }
        });
    });

    // 4. Κουμπί Ανανέωσης Προτάσεων
    btnRefresh.addEventListener('click', async () => {
        if (!currentEmotion) return;
        const strategy = document.querySelector('input[name="strategy"]:checked').value;

        try {
            const res = await fetch('/api/recommend-manual', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ emotion: currentEmotion, strategy: strategy })
            });
            const data = await res.json();
            if (data.success) {
                renderMovies(data.movies);
            }
        } catch (err) {
            console.error("API error:", err);
        }
    });

    // 5. Εμφάνιση Αποτελεσμάτων στο UI
    function renderResults(data) {
        currentEmotion = data.emotion.name;

        // Εμφάνιση active emotion κάρτας
        placeholderState.style.display = 'none';
        activeEmotion.style.display = 'block';

        dominantEmoji.textContent = data.emotion.emoji;
        dominantName.textContent = `${data.emotion.label_el} (${data.emotion.name})`;
        if (data.emotion.confidence) {
            dominantConf.textContent = `${data.emotion.confidence}%`;
            dominantConf.parentElement.style.display = 'inline-block';
        } else {
            dominantConf.parentElement.style.display = 'none';
        }

        // Εμφάνιση μπαρών ποσοστών αν υπάρχουν
        breakdownList.innerHTML = '';
        if (data.emotion.breakdown && data.emotion.breakdown.length > 0) {
            document.querySelector('.breakdown-container').style.display = 'block';
            data.emotion.breakdown.forEach(item => {
                const row = document.createElement('div');
                row.className = 'breakdown-item';
                row.innerHTML = `
                    <span class="emo-name">${item.emoji} ${item.label_el}</span>
                    <div class="bar-wrapper">
                        <div class="bar-fill" style="width: ${item.percentage}%; background-color: ${item.color};"></div>
                    </div>
                    <span class="emo-pct">${item.percentage}%</span>
                `;
                breakdownList.appendChild(row);
            });
        } else {
            document.querySelector('.breakdown-container').style.display = 'none';
        }

        // Εμφάνιση Ταινιών
        renderMovies(data.movies);
    }

    function renderMovies(movies) {
        recSection.style.display = 'block';
        moviesGrid.innerHTML = '';

        if (!movies || movies.length === 0) {
            moviesGrid.innerHTML = '<p class="no-movies">Δεν βρέθηκαν προτάσεις.</p>';
            return;
        }

        movies.forEach(movie => {
            const card = document.createElement('div');
            card.className = 'movie-card';

            const genresHtml = movie.genres.map(g => `<span class="genre-tag">${g}</span>`).join('');
            
            const tmdbLink = movie.tmdb_id ? 
                `<a href="https://www.themoviedb.org/movie/${movie.tmdb_id}" target="_blank" class="btn-movie">TMDB ↗</a>` : '';
            const imdbLink = movie.imdb_id ? 
                `<a href="https://www.imdb.com/title/tt${movie.imdb_id}" target="_blank" class="btn-movie">IMDb ↗</a>` : '';

            card.innerHTML = `
                <div class="movie-card-top">
                    ${movie.year ? `<span class="movie-year">${movie.year}</span>` : ''}
                    <h3 class="movie-title">${movie.title}</h3>
                    <div class="movie-genres">${genresHtml}</div>
                </div>
                <div>
                    <div class="movie-stats">
                        <span class="rating-badge">⭐ ${movie.rating} / 5.0</span>
                        <span class="votes-badge">${movie.votes} ψήφοι</span>
                    </div>
                    <div class="movie-actions">
                        ${tmdbLink}
                        ${imdbLink}
                    </div>
                </div>
            `;
            moviesGrid.appendChild(card);
        });
    }
});


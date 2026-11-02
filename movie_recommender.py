import os
import re
import pandas as pd
import numpy as np

class MovieRecommender:
    """
    Σύστημα Συστάσεων Ταινιών βασισμένο σε Συναισθήματα και το MovieLens Dataset.
    Χρησιμοποιεί τον αλγόριθμο Weighted Rating (IMDB formula) και αντιστοίχιση Συναισθήματος -> Genres.
    """
    
    # Χαρτογράφηση Συναισθημάτων σε Είδη Ταινιών (Genres)
    # - 'booster': Ανύψωση / Βελτίωση διάθεσης (Mood Repair)
    # - 'mirror': Ταίριασμα με την τρέχουσα διάθεση (Mood Congruence)
    EMOTION_GENRE_MAP = {
        'booster': {
            'Sad': ['Comedy', 'Animation', 'Adventure', 'Children', 'Musical'],
            'Angry': ['Comedy', 'Animation', 'Fantasy', 'Adventure', 'Sci-Fi'],
            'Fear': ['Comedy', 'Animation', 'Romance', 'Adventure'],
            'Disgust': ['Comedy', 'Animation', 'Adventure', 'Fantasy'],
            'Happy': ['Comedy', 'Adventure', 'Animation', 'Musical', 'Romance'],
            'Surprise': ['Sci-Fi', 'Mystery', 'Adventure', 'Fantasy'],
            'Neutral': ['Drama', 'Action', 'Comedy', 'Crime', 'Sci-Fi']
        },
        'mirror': {
            'Sad': ['Drama', 'Romance'],
            'Angry': ['Action', 'Crime', 'Thriller', 'War'],
            'Fear': ['Horror', 'Mystery', 'Thriller'],
            'Disgust': ['Horror', 'Crime', 'Mystery'],
            'Happy': ['Comedy', 'Adventure', 'Animation', 'Musical'],
            'Surprise': ['Mystery', 'Sci-Fi', 'Thriller'],
            'Neutral': ['Drama', 'Comedy', 'Action', 'Sci-Fi', 'Documentary']
        }
    }

    def __init__(self, data_dir='ml-latest-small', default_min_votes=20):
        self.data_dir = data_dir
        self.default_min_votes = default_min_votes
        self.movies_df = None
        self.ratings_df = None
        self.links_df = None
        self.combined_df = None
        self._load_and_process_data()

    def _load_and_process_data(self):
        """Φόρτωση των αρχείων CSV και υπολογισμός του IMDB Weighted Rating."""
        movies_path = os.path.join(self.data_dir, 'movies.csv')
        ratings_path = os.path.join(self.data_dir, 'ratings.csv')
        links_path = os.path.join(self.data_dir, 'links.csv')

        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)
        links = pd.read_csv(links_path)

        # Υπολογισμός στατιστικών βαθμολογιών ανά ταινία:
        # v = πλήθος ψήφων, R = μέση βαθμολογία
        stats = ratings.groupby('movieId').agg(
            vote_count=('rating', 'count'),
            avg_rating=('rating', 'mean')
        ).reset_index()

        # Συνένωση πινάκων
        df = movies.merge(stats, on='movieId', how='inner')
        df = df.merge(links, on='movieId', how='left')

        # Υπολογισμός Weighted Rating (IMDB Formula):
        # WR = (v / (v + m)) * R + (m / (v + m)) * C
        C = df['avg_rating'].mean()
        m = self.default_min_votes

        def weighted_rating(row):
            v = row['vote_count']
            R = row['avg_rating']
            return (v / (v + m) * R) + (m / (v + m) * C)

        df['weighted_score'] = df.apply(weighted_rating, axis=1)

        # Εξαγωγή του έτους κυκλοφορίας από τον τίτλο (π.χ. "Toy Story (1995)" -> 1995)
        def extract_year(title):
            match = re.search(r'\((\d{4})\)', title)
            return int(match.group(1)) if match else None

        # Καθαρισμός τίτλου (αφαίρεση του έτους από το string)
        def clean_title(title):
            return re.sub(r'\s*\(\d{4}\)', '', title).strip()

        df['year'] = df['title'].apply(extract_year)
        df['clean_title'] = df['title'].apply(clean_title)

        self.combined_df = df
        print(f"Επιτυχής φόρτωση {len(df)} ταινιών (Παγκόσμιος μέσος όρος C={C:.2f}, ελάχιστες ψήφοι m={m}).")

    def recommend(self, emotion, strategy='booster', top_n=10, min_votes=None, diverse=True):
        """
        Προτείνει ταινίες με βάση το συναίσθημα.
        
        :param emotion: 'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'
        :param strategy: 'booster' (βελτίωση διάθεσης) ή 'mirror' (ταίριασμα διάθεσης)
        :param top_n: Πλήθος προτεινόμενων ταινιών
        :param min_votes: Ελάχιστος αριθμός ψήφων (default: 20)
        :param diverse: Αν True, επιλέγει τυχαία από τις κορυφαίες για ποικιλία σε κάθε κλήση
        :return: List of dicts με τις προτεινόμενες ταινίες
        """
        if emotion not in self.EMOTION_GENRE_MAP['booster']:
            raise ValueError(f"Μη έγκυρο συναίσθημα '{emotion}'.")

        target_genres = self.EMOTION_GENRE_MAP[strategy].get(emotion, [])
        threshold = min_votes if min_votes is not None else self.default_min_votes

        # Φιλτράρισμα ταινιών με βάση τις ελάχιστες ψήφους
        filtered = self.combined_df[self.combined_df['vote_count'] >= threshold].copy()

        # Υπολογισμός πόσα από τα επιθυμητά genres περιέχει η κάθε ταινία
        pattern = '|'.join(target_genres)
        matched = filtered[filtered['genres'].str.contains(pattern, case=False, na=False)].copy()

        # Ταξινόμηση κατά weighted_score
        sorted_movies = matched.sort_values(by='weighted_score', ascending=False)

        if diverse and len(sorted_movies) > top_n:
            # Δειγματοληψία από το top 3x για να μην βγαίνουν πάντα οι ίδιες ταινίες
            pool_size = min(len(sorted_movies), top_n * 3)
            candidates = sorted_movies.head(pool_size)
            selected = candidates.sample(n=top_n)
            # Αναδιάταξη κατά βαθμολογία
            selected = selected.sort_values(by='weighted_score', ascending=False)
        else:
            selected = sorted_movies.head(top_n)

        # Μετατροπή σε λίστα λεξικών (JSON-ready)
        results = []
        for _, row in selected.iterrows():
            results.append({
                'movie_id': int(row['movieId']),
                'title': row['clean_title'],
                'full_title': row['title'],
                'year': int(row['year']) if pd.notnull(row['year']) else None,
                'genres': [g for g in row['genres'].split('|') if g != '(no genres listed)'],
                'rating': round(float(row['avg_rating']), 1),
                'votes': int(row['vote_count']),
                'score': round(float(row['weighted_score']), 2),
                'tmdb_id': int(row['tmdbId']) if pd.notnull(row['tmdbId']) else None,
                'imdb_id': str(row['imdbId']).zfill(7) if pd.notnull(row['imdbId']) else None
            })

        return results

if __name__ == '__main__':
    recommender = MovieRecommender()
    
    print("\n--- Δοκιμή Σύστασης για 'Sad' με Strategy 'booster' (Ανύψωση Διάθεσης) ---")
    movies = recommender.recommend('Sad', strategy='booster', top_n=5, diverse=False)
    for i, m in enumerate(movies, 1):
        print(f"{i}. {m['title']} ({m['year']}) - {', '.join(m['genres'])} | ⭐ {m['rating']}/5 (TMDB ID: {m['tmdb_id']})")

    print("\n--- Δοκιμή Σύστασης για 'Fear' με Strategy 'mirror' (Ταίριασμα Διάθεσης) ---")
    movies = recommender.recommend('Fear', strategy='mirror', top_n=5, diverse=False)
    for i, m in enumerate(movies, 1):
        print(f"{i}. {m['title']} ({m['year']}) - {', '.join(m['genres'])} | ⭐ {m['rating']}/5 (TMDB ID: {m['tmdb_id']})")

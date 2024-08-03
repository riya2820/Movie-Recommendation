# Movie Recommendation System 

This project is a movie recommendation system developed using machine learning algorithms, Python's sci-kit learn, and pandas library. The system uses three types of algorithms, content-based filtering collaborative-based filtering, and demographic filtering to generate movie recommendations for the user based on their past movie preferences.

# Recommendation Approaches

This system leverages three distinct recommendation strategies to cater to different user preferences and scenarios:

1. Content-Based Filtering
recommends movies by analyzing the content of movies and user preferences. It uses movie metadata, such as genre, director, description, actors, etc., to find a match between the movie's features and the user's preferences. This method assumes if a user liked a particular movie, they will also like movies that are similar in content.

2. Collaborative-Based Filtering
generates movie recommendations based on past behavior of users in the dataset and not on the content of the movies themselves. Forms a network of such behaviors and suggests movies based on user patterns. 

3. Demographic Filtering
recommends movies based on the demographic characteristics of users, such as age, location, gender, etc. This approach assumes that users with similar demographic features will have similar movie preferences. 

# Installation

To install the project, follow these steps:

Clone this repository: git clone https://github.com/your-username/movie-recommendation-system.git.
Navigate to the project directory: cd movie-recommendation-system.
Install the dependencies: pip install pandas numpy scikit-learn.


# Limitations

The movie recommendation system has certain limitations, including:

The system's accuracy depends on the quality and quantity of the data provided.
The system does not take into account the user's mood or context when recommending movies.

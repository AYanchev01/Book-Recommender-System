import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

books = pd.read_csv('data/BX-Books.csv', sep=";", on_bad_lines='skip', encoding='latin-1',low_memory=False)


# Information about the dataframe
# print(books.shape)
# print(books.columns)

#Leaving only columns needed for analysis
books = books[['ISBN','Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]

# Lets remane some wierd columns name
books.rename(columns={"Book-Title":'Title',
                      'Book-Author':'Author',
                     "Year-Of-Publication":'Year',
                     "Publisher":"Publisher"},inplace=True)



# Now load the second dataframe

users = pd.read_csv('data/BX-Users.csv', sep=";", on_bad_lines='skip', encoding='latin-1',low_memory=False)
# print(users.shape)
print(users.columns)

#print(users.head())

ratings = pd.read_csv('data/BX-Book-Ratings.csv', sep=";", on_bad_lines='skip', encoding='latin-1',low_memory=False)

# print(ratings.columns)

#print(ratings.head())

# fig,ax = plt.subplots(figsize=(15,10))
# sns.distplot(ratings['Book-Rating'], ax=ax)
# ax.set_title('Distribution of ratings')
# ax.set_xlabel('Rating')
# plt.show()


# Lets store users who had at least rated more than 500 books
best_user_ratings = ratings['User-ID'].value_counts() > 500



#print(best_user_ratings[best_user_ratings].shape)

# Leaving only the best users in the dataframe

#print(ratings.shape)
ratings = ratings[ratings['User-ID'].isin(best_user_ratings[best_user_ratings].index)]
#print(ratings.shape)
#print(ratings.head())


# Now join ratings with books

ratings_with_books = ratings.merge(books, on='ISBN')
#print(ratings_with_books.head())

ratings_count = ratings_with_books.groupby('Title')['Book-Rating'].count().reset_index()
print(ratings_count.head())
ratings_count.rename(columns={'Book-Rating':'Ratings-Count'},inplace=True)
print(ratings_count.head())

merged_with_ratings_and_books = ratings_with_books.merge(ratings_count, on='Title')
#print(merged_with_ratings_and_books.head())

# fig,ax = plt.subplots(figsize=(15,10))
# sns.distplot(merged_with_ratings_and_books['Ratings-Count'], ax=ax)
# ax.set_title('Distribution of ratings count')
# ax.set_xlabel('Rating Count')
# plt.show()

# Leaving only the books with more than 40 ratings
final_rating = merged_with_ratings_and_books[merged_with_ratings_and_books['Ratings-Count'] >= 40]


# Drop the duplicates from the final_rating dataframe
final_rating.drop_duplicates(['User-ID', 'Title'], inplace=True)

# plt.figure(figsize=(15,10))
# ax=sns.relplot(data=final_rating, x='Ratings-Count', y='Book-Rating')
# plt.title('Relationship between ratings count and book rating')
# ax.set_axis_labels('Ratings Count', 'Book Rating')
# plt.show()


# Lets create a pivot table
book_pivot = final_rating.pivot_table(columns='User-ID', index='Title', values= 'Book-Rating')
# print(book_pivot.head())

book_pivot.fillna(0, inplace=True)
#print(book_pivot.head())

book_sparse = csr_matrix(book_pivot)
#print(book_sparse)

# Now import our clustering algoritm which is Nearest Neighbors this is an unsupervised ml algo
model = NearestNeighbors(algorithm= 'brute')

model.fit(book_sparse)


def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6 )
    
    for i in range(len(suggestion)):
            books = book_pivot.index[suggestion[i]]
            for j in books:
                if j == book_name:
                    print(f"You searched '{book_name}'\n")
                    print("The suggestion books are: \n")
                else:
                    print(j)

book_name = "Harry Potter and the Chamber of Secrets (Book 2)"
recommend_book(book_name)

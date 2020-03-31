import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score, make_scorer, accuracy_score
from typing import Tuple, List
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import pickle

def load_data(database_filepath: str)->Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    '''
    Function for loading the database into pandas DataFrames
    Args: database_filepath: the path of the database
    Returns:    X: features (messages)
                y: categories (one-hot encoded)
                An ordered list of categories
    '''
    # Loading the database into a pandas DataFrame
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM disaster_response",engine) 

    # Making DataFrame with only features
    X = df['message']

    # Making DataFrame with relevant categories
    y = df.drop(['id', 'message', 'original', 'genre'],  axis=1).astype(float)
    categories = y.columns.values
    return X, y, categories

def tokenize(text:str)->List[str] :
    '''
    Function for tokenizing string
    Args: Text string
    Returns: List of tokens
    '''
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    
    return [lemmatizer.lemmatize(x).lower().strip() for x in tokens]

def build_model()->GridSearchCV:
    '''
    Function for building pipeline and GridSearch
    Args: None
    Returns: Model
    '''
    # Pipeline for transforming data, fitting to model and predicting
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())])
    
    # Parameters for GridSearch (simplified due to time challenges :-( )
    parameters = {
        'clf__min_samples_split': [5,10, 15],
        'clf__n_estimators': [50, 100, 150]}

    # GridSearch with the above parameters
    cv = GridSearchCV(pipeline, param_grid=parameters,
                      scoring='accuracy',verbose= 1,n_jobs =-1)

    return pipeline

def evaluate_model(model: GridSearchCV, X_test: pd.DataFrame, Y_test: pd.DataFrame, category_names: List)->None:
    '''
    Function for evaluating model by printing a classification report
    Args:   Model, features, labels to evaluate, and a list of categories
    Returns: Classification report
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    for  idx, cat in enumerate(Y_test.columns.values):
        print("{} -- {}".format(cat, accuracy_score(Y_test.values[:,idx], y_pred[:, idx])))
    print("accuracy = {}".format(accuracy_score(Y_test, y_pred)))

def save_model(model: GridSearchCV, model_filepath: str)-> None:
    '''
    Function for saving the model as picklefile
    Args: Model, filepath
    Returns: Nothing. Saves model to pickle file
    '''
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
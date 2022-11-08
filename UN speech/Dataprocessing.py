import os
import sys
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import sklearn
from sklearn.cluster import KMeans
from collections import Counter
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist
from pywsd.utils import lemmatize_sentence
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud, ImageColorGenerator

nltk.download("omw-1.4")
nltk.download("wordnet")


try:
    nltk.data.find("punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

dir_path = os.path.dirname(os.path.realpath(__file__))
main_data_dir = os.path.join(dir_path, "TXT")


def open_speech(file_path):
    """
    This function opens a file with the correct formatting
    :param file_path:
    :return:
    """

    file = open(file_path, encoding="utf-8-sig")
    data = file.read()

    return data


def remove_line_number(speech):
    """
    removes the line number at the beginning of speech

    Parameters
    ---------
    speech : str
        piece of text
    """

    pattern = "\n|^\d+.*?(\w)"
    speech = re.sub(pattern, "\n\g<1>", speech)
    pattern = "\t"
    speech = re.sub(pattern, "", speech)
    pattern = "\n\n"
    speech = re.sub(pattern, "\n", speech)
    pattern = "^\n *"
    speech = re.sub(pattern, "", speech)

    return speech


def stem_token(token):
    """
    Stems the given token using the PorterStemmer from the nltk library
    Input: a single token
    Output: the stem of the token
    """
    ps = PorterStemmer()
    stemmed_word = ps.stem(token)
    return stemmed_word


def lemmatize_token(token):
    """
    lemmatize the token using nltk library
    Input: a single token
    Output: the lemmatization of the token
    """
    wordnet = WordNetLemmatizer()
    lemmatized_word = wordnet.lemmatize(token)
    return lemmatized_word


def preprocess_speech(speech):
    """
    This function does the preprocessing
    :param data:
    :return:
    """
    # put all characters in lower case
    speech = speech.lower()

    # only keep the tokens of the data
    tokens = nltk.word_tokenize(speech)

    # lemmatizing
    tokens = [lemmatize_token(token) for token in tokens]

    # remove stop words and non-alphabetic from all the text
    sw = nltk.corpus.stopwords.words("english")
    no_sw = []
    for w in tokens:
        if (w not in sw) and w.isalpha():
            no_sw.append(w)

    return no_sw


def get_index(mylist, value):
    try:
        index = mylist.index(value)
    except:
        index = np.nan
    return index


def determine_sentiment(data):
    """
    This function determines the sentiment of a string (in this case a speech)
    :param data: the speech / a string
    :return:
    """
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(data)
    return sentiment


def count_word_occurence(speech, mode):
    """
    Feature vector from unstructured text.
     - Count encoding, used to represent the frequency of words in a vocabulary for a document
     - TF-IDF encoding, used to represent normalized word frequency scores in a vocabulary.
    :param document: iterable object with speech(es)
    :param mode: string with encoding technique to use. Can be - ['count','tf-idf']
    :return: returns list with top 20 most important words
    """
    if mode == "count":
        obj = CountVectorizer(lowercase=True, stop_words="english")
        word_occ = obj.fit_transform(speech)
    elif mode == "tf-idf":
        obj = TfidfVectorizer(lowercase=True, stop_words="english")
        word_occ = obj.fit_transform(speech)

    text_features = pd.DataFrame(word_occ.toarray(), columns=obj.get_feature_names())
    # Get the top 10 words per country
    top10_country = text_features.apply(
        lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=10
    )
    # Get the top 20 words for the whole year
    avg_tfidf = text_features.mean(axis=0)
    top20_year = avg_tfidf.nlargest(20).index
    return top20_year, top10_country


def top10words_percountry(documents, mode, n=10):
    """
    Feature vector from unstructured text.
     - Count encoding, used to represent the frequency of words in a vocabulary for a document
     - TF-IDF encoding, used to represent normalized word frequency scores in a vocabulary.
    :param documents: iterable object with speech(es) ['speech1 from Albania', 'speech 2 from Denmark']
    :param mode: string with encoding technique to use. Can be - ['count','tf-idf']
    :return: returns list with top 10 most important words
    """
    if mode == "count":
        obj = CountVectorizer(lowercase=True, stop_words="english")
        word_occ = obj.fit_transform(documents)
    elif mode == "tf-idf":
        obj = TfidfVectorizer(lowercase=True, stop_words="english")
        word_occ = obj.fit_transform(documents)

    text_features = pd.DataFrame(word_occ.toarray(), columns=obj.get_feature_names())
    # Get the top 10 words per country
    topwords_country = text_features.apply(
        lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=n
    )

    return topwords_country.values.tolist()


def topnwords_peryear(documents, mode, n=10):
    """
    Vectorizes documents from all years and calculates most important words based on the preferred method.
    :param documents: list with the speeches per year
    :param mode: string with encoding technique to use. Can be - ['count','tf-idf']
    :return: n most important words per year
    """
    if mode == "count":
        obj = CountVectorizer(lowercase=True, stop_words="english")
        word_occ = obj.fit_transform(documents)
    elif mode == "tf-idf":
        obj = TfidfVectorizer(lowercase=True, stop_words="english")
        word_occ = obj.fit_transform(documents)
    text_features = pd.DataFrame(word_occ.toarray(), columns=obj.get_feature_names())
    # Get the top 10 words per year
    topwords_year = text_features.apply(
        lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=n
    )
    # Get the top 20 words for the whole year
    # avg_tfidf = text_features.mean(axis=0)
    return topwords_year.values.tolist()


def count_most_used_words(data, n):
    """
    This functions a list of the n most used words with their occurrence rate
    :param data: the string to gather the word occurrences from
    :param n: number of most used words in the speeches
    :return: two lists, first with the most occurring words and then their occurrence rate
    """

    occ_words = Counter(data).most_common(n)

    words = [occ_tup[0] for occ_tup in occ_words]
    counts = [occ_tup[1] for occ_tup in occ_words]
    return words, counts


def topnwords_perspeech(documents, mode, n=100):
    if mode == "count":
        obj = CountVectorizer(lowercase=True, stop_words="english")
        word_occ = obj.fit_transform(documents)
    elif mode == "tf-idf":
        obj = TfidfVectorizer(lowercase=True, stop_words="english")
        word_occ = obj.fit_transform(documents)

    text_features = pd.DataFrame(word_occ.toarray(), columns=obj.get_feature_names())
    # Get the top 10 words per country
    top10_country = text_features.apply(
        lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=n
    )
    return top10_country.values.tolist()


def filter_common_words(words):
    common_words = [
        "united",
        "nations",
        "country",
        "countries",
        "viet",
        "nam",
        "state",
        "assembly",
        "today",
        "netherlands",
        "states",
        "general",
        "israeli",
        "people",
        "world",
        "peoples",
        "region",
        "international",
        "syria",
        "syrian",
        "new",
    ]
    return [word for word in words if word not in common_words]


def happiness_df_cleanup(happiness_df):
    """
    Creating a dictionary to map the wrong countries name in
    happiness dataframe with correct country name. This is because
    we need to merge happiness dataframe with speeches dataframe
    using country name. And we found cases where country names are
    not the same between these two data set.
    See function check_countryname_consistency as well.
    """

    country_mapping = {
        "Vietnam": "Viet Nam",
        "Moldova": "Republic of Moldova",
        "Laos": "Lao People's Democratic Republic",
        "Somaliland region": "Somalia",
        "Kosovo": None,
        "Taiwan Province of China": None,
        "United Kingdom": "United Kingdom of Great Britain and Northern Ireland",
        "United States": "United States of America",
        "South Korea": "Republic of Korea",
        "Ivory Coast": "Côte d’Ivoire",
        "Czech Republic": "Czechia",
        "Swaziland": "Eswatini",
        "Russia": "Russian Federation",
        "Hong Kong S.A.R. of China": "China-Hong Kong Special Administrative Region",
        "Palestinian Territories": "State of Palestine",
        "Tanzania": "United Republic of Tanzania",
        "Syria": "Syrian Arab Republic",
        "North Cyprus": None,
        "Bolivia": "Bolivia (Plurinational State of)",
        "Congo (Kinshasa)": "Democratic Republic of the Congo",
        "Venezuela": "Venezuela (Bolivarian Republic of)",
        "Iran": "Iran (Islamic Republic of)",
        "Congo (Brazzaville)": "Congo",
    }
    # Replace the country names in happiness dataframe with the correct country names in df-codes.
    happiness_df = (
        happiness_df.reset_index()
        .replace({"Country name": country_mapping})
        .set_index(["Country name", "year"])
    )

    return happiness_df


def speeches_df_cleanup(speeches_df):
    """
    Replace YDYE (yemen) and POR (Portugal) with correct iso_alpho3 code.
    The remaing 'DDR', 'YUG', 'EU', 'CSK' are not considered countries by
    the UN or don't exist anymore, so we can consider removing them out of
    dataset because we don't have happiness data for these "countries".
    """
    speeches_df["country"] = (
        speeches_df["country"].str.replace("YDYE", "YEM").replace("POR", "PRT")
    )
    return speeches_df


def check_countryname_consistency(happiness_df, df_codes):
    happiness_countries = set(happiness_df.reset_index()["Country name"].unique())
    iso_countries = set(df_codes["Country or Area"])
    print(
        f"The following countries in happiness_df do not appear in iso_countries: {happiness_countries - iso_countries}"
    )


def check_isocode_consistency(speeches_df, df_codes):
    speech_codes = set(speeches_df["country"].unique())
    iso_codes = set(df_codes["ISO-alpha3 Code"].unique())
    print(
        f"The following codes in speeches_df do not appear in df_codes: {speech_codes - iso_codes}"
    )


def interpolate(df, col, country):
    minidf = pd.DataFrame({col: df[col].loc[:, country].interpolate(method="slinear")})
    minidf["country"] = country
    minidf = minidf.reset_index()
    minidf.set_index(["year", "country"], inplace=True)
    return df.update(minidf)


def multi_interpolate(df, columns, countries):
    for country in tqdm(countries):
        for column in columns:
            try:
                interpolate(df, column, country)
            except:
                print(f"FAIL: {country} : {column}")
    return df


def count_list_occurences(list, x):
    """This function counts the number of times an element appears
    in a list
    :param file: a list and an element
    :return: the number of times the element appeared in list
    """
    counter = 0
    for i in list:
        if x == i:
            counter += 1
    return counter


def count_referenced_countries(data, countries, years):
    """This function iterates over the speeches in the selected years
    and counts the number of times each country appeared in each speech,
    and the number of speeches that mentioned the respective country

    :param file: the dataframe, a list of countries and years
    :return: the number of references of each country per year and
    the number of speeches referencing the respective country per year
    """
    for y in years:
        year = data.loc[(y)]
        year["processed_speech"] = year.apply(
            lambda row: preprocess_speech_(row["Speech"]), axis=1
        )
        for i in countries:
            s = str(i) + "_ref_count_" + str(y)
            year[s] = year.apply(
                lambda row: count_list_occurences(row["processed_speech"], i), axis=1
            )  # only looks for usa, not united states, america etc
            print("# references", i, "(", y, ")", sum(year[s]))
            s1 = "bool_" + str(i) + "_ref_" + str(y)
            year[s1] = np.where(year[s] == 0, False, True)
            print("# speeches referencing", i, "(", y, ")", sum(year[s1]))


def plot_sentiment_country_vs_year(country_code):
    """
    This function plots the sentiment scores of a specific country over the years
    :param country_code: the code of the country that you want the sentiment development plot for
    :return:
    """

    plt.figure()
    plt.title("{} speech sentiment".format(country_code))
    plt.plot(
        speeches_df.loc[speeches_df["country"] == country_code]["year"],
        speeches_df.loc[speeches_df["country"] == country_code]["pos_sentiment"],
        label="positive",
    )
    plt.plot(
        speeches_df.loc[speeches_df["country"] == country_code]["year"],
        speeches_df.loc[speeches_df["country"] == country_code]["neu_sentiment"],
        label="neutral",
    )
    plt.plot(
        speeches_df.loc[speeches_df["country"] == country_code]["year"],
        speeches_df.loc[speeches_df["country"] == country_code]["neg_sentiment"],
        label="negative",
    )
    plt.legend()
    plt.xlabel("Time (year)", fontsize=14)
    plt.ylabel("Sentiment", fontsize=14)
    plt.savefig("{}_sentiment_development.png".format(country_code), dpi=300)
    plt.show()


def generate_wordcloud(important_words, year):
    """
    Generate word cloud from list of most important words. Plots it.
    :param important_words: list with the top words.
    """

    word_counter = Counter(important_words)
    wordcloud = WordCloud(background_color="white").generate_from_frequencies(
        word_counter
    )
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title(str(year))
    plt.show()

    return wordcloud


def plot_correlation_matrix(speeches_df, corr_cols):
    """
    This function creates a correlation matrix of specific columns from the speeches df
    :param speeches_df: dataframe containing all the information about the speeches
    :param corr_cols: a list of columns that we cant to calculate the correlation for
    :return:
    """
    matrix = np.triu(speeches_df.loc[:, corr_cols].corr())

    plt.figure()
    sns.heatmap(
        speeches_df.loc[:, corr_cols].corr(), vmin=-1, vmax=1, center=0, mask=matrix
    )
    plt.show()


if __name__ == "__main__":

    # True --> run preprocessing and save the results, False --> just do the data analysis with your previously saved
    # dataframe file (always have to do a preprocessing run to save the dataframe of course)
    do_preprocessing = False

    if do_preprocessing:
        speeches_df = pd.DataFrame(
            columns=[
                "session_nr",
                "year",
                "country",
                "word_count",
                "pos_sentiment",
                "neu_sentiment",
                "neg_sentiment",
                "average_sentence_length",
                "most_used_words",
                "speech",
            ]
        )

        num_directories = len(next(os.walk(main_data_dir))[1])

        # loop through all directories of the data
        for root, subdirectories, files in tqdm(
            os.walk(main_data_dir), total=num_directories, desc="directory: "
        ):

            # remove all the files starting with '.' (files created by opening a mac directory on a windows PC,
            # so will only do something if you are working on a windows PC
            files_without_dot = [file for file in files if not file.startswith(".")]

            # loop through files and extract data
            for file in tqdm(files_without_dot, desc="files: ", leave=False):
                country, session_nr, year = file.replace(".txt", "").split("_")

                # open a speech with the correct formatting
                speech_data = open_speech(os.path.join(root, file))
                speech_data = remove_line_number(speech_data)

                # preprocess the data
                preprocessed_bag_of_words = preprocess_speech(speech_data)

                # calculate all the features through functions
                word_count = count_total_words(preprocessed_bag_of_words)
                most_used_words = count_most_used_words(preprocessed_bag_of_words, 20)
                occs_of_spec_words = count_specific_words(
                    preprocessed_bag_of_words, ["economy"]
                )
                sentiment_of_speech = determine_sentiment(speech_data)
                average_sentence_length = determine_average_sentence_length(speech_data)

                # append the line of features to the dataframe
                speeches_df = speeches_df.append(
                    {
                        "session_nr": int(session_nr),
                        "year": int(year),
                        "country": country,
                        "word_count": word_count,
                        "pos_sentiment": sentiment_of_speech["pos"],
                        "neu_sentiment": sentiment_of_speech["neu"],
                        "neg_sentiment": sentiment_of_speech["neg"],
                        "average_sentence_length": average_sentence_length,
                        "most_used_words": most_used_words,
                        "speech": speech_data,
                    },
                    ignore_index=True,
                )

        # read in country codes and happiness data
        df_codes = pd.read_csv("UNSD — Methodology.csv", delimiter=",")
        happiness_df = pd.read_excel("DataPanelWHR2021C2.xls", index_col=[0, 1])

        # check if country names and iso codes are consistent before merging
        check_isocode_consistency(speeches_df, df_codes)
        check_countryname_consistency(happiness_df, df_codes)

        # cleanup dataframes before merging
        speeches_df = speeches_df_cleanup(speeches_df)
        happiness_df = happiness_df_cleanup(happiness_df)

        speeches_df = speeches_df.merge(
            df_codes, how="left", left_on="country", right_on="ISO-alpha3 Code"
        )

        # add the happiness dataframe to the

        speech_happi_merged_df = pd.merge(
            speeches_df,
            happiness_df,
            how="left",
            left_on=["year", "Country or Area"],
            right_on=["year", "Country name"],
        )

        speech_happi_merged_df.to_csv("preprocessed_dataframe.csv")

    speeches_df = pd.read_csv("preprocessed_dataframe.csv")
    # Turn preprocessed speech into text
    speeches_df["preprocessed_speech"] = speeches_df["preprocessed_speech"].apply(
        lambda x: " ".join(ast.literal_eval(x))
    )

    # Create the Empty Lists
    top10_aux = []
    texts_over_years = []
    # Loop through the years
    for year in range(speeches_df["year"].min(), speeches_df["year"].max() + 1):
        # Select view of the year
        year_data = speeches_df[speeches_df["year"] == year]
        # Calculate top 10 per year
        topwords_countryyear = top10words_percountry(
            year_data["preprocessed_speech"], mode="tf-idf"
        )
        # Append to Dataframe
        top10_aux.append(topwords_countryyear)
        # Aggregate all speeches
        year_text = " ".join(speeches_df[speeches_df["year"] == year]["speech"])
        # Append Dataframe
        texts_over_years.append(year_text)

    # Add column 'top10words' to main df for each speech
    speeches_df["top10words"] = top10_aux
    # Get main words throughout all years
    words_over_years_df = pd.DataFrame(
        {
            "years": [x for x in range(1970, 2021)],
            "topwords": topnwords_peryear(texts_over_years, mode="tf-idf", n=10),
        }
    )

    corr_cols = [
        "year",
        "word_count",
        "pos_sentiment",
        "neg_sentiment",
        "neu_sentiment",
        "average_sentence_length",
        "Life Ladder",
        "Log GDP per capita",
        "Social support",
        "Freedom to make " "life choices",
        "Generosity",
        "Perceptions of corruption",
    ]
    plot_correlation_matrix(speeches_df, corr_cols)

    countries = speeches_df.reset_index()["country"].unique()
    cols = [
        "Life Ladder",
        "Log GDP per capita",
        "Social support",
        "Healthy life expectancy at birth",
        "Freedom to make life choices",
        "Generosity",
        "Perceptions of corruption",
        "Positive affect",
        "Negative affect",
    ]

    speeches_df.set_index(["year", "country"], inplace=True)

    # plotting how most common words evolve over years per country

    fig, axs = plt.subplots(2, 2, sharey=True, figsize=(13, 13))
    for i, country in enumerate(["SYR", "USA", "NLD", "VNM"]):
        #     try:
        ax = axs.flatten()[i]
        times = np.arange(2010, 2021)
        top_per_year_country = topnwords_perspeech(
            [
                speeches_df[
                    (speeches_df["year"] == year) & (speeches_df["country"] == country)
                ]["speech"].values[-1]
                for year in times
            ],
            "tf-idf",
            n=100,
        )
        # take all speeches of a specific country
        merged_speeches = " ".join(
            speeches_df[speeches_df["country"] == country]["preprocessed_speech"].values
        )
        top_allyears = topnwords_perspeech([merged_speeches], "tf-idf", n=100)
        top_allyears = [filter_common_words(top_allyears[0])]
        score_dict = {}
        for word in top_allyears[0][:8]:
            indices = [
                100 - get_index(top_per_year_country[i], word)
                for i in range(len(top_per_year_country))
            ]
            score_dict[word] = indices
        for word in score_dict.keys():
            #             None
            size = [0 if np.isnan(n) else n for n in score_dict[word]]
            ax.scatter(times, score_dict[word], label=word, s=size)
        ax.legend()
        ax.set_xlabel("Year")
        ax.set_ylabel("Score")
        ax.set_title(country)
        pass
    plt.show()

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from matplotlib import rcParams
import text_preprocessing as tp

rcParams['figure.figsize'] = 10, 5
tqdm.pandas()

class IMDbDataset(pd.DataFrame):
    def __init__(self, data_file):
        super().__init__(pd.read_csv(data_file))
        self.clean_reviews()

    def clean_reviews(self):
        """
        Apply text preprocessing to clean reviews.
        """
        self['review'] = self['review'].apply(lambda x: tp.remove_html(x))
        self['clean_review'] = self['review'].progress_apply(self.preprocess_review)

    def preprocess_review(self, x):
        """
        Preprocess review text.
        """
        return tp.preprocess_doc(x) if isinstance(x, str) else x
    
    def get_word_cloud_data(self, text_data):
        """
        Generate word cloud data.
        """
        wc = WordCloud(stopwords=STOPWORDS, 
                    background_color="black", contour_width=2,
                    max_words=2000, max_font_size=256,
                    random_state=42)
        wc.generate(' '.join(text_data))
        return wc

    def get_review_counts(self):
        """
        Get counts of positive and negative reviews.
        """
        count_good = len(self[self['sentiment'] == 'positive'])
        count_bad = len(self[self['sentiment'] == 'negative'])
        return count_good, count_bad

    def good_bad_reviews(self):
        """
        Plot counts of positive and negative reviews.
        """
        count_good, count_bad = self.get_review_counts()
        print('Total Counts of both sets: Good Reviews {}, Bad Reviews {}'.format(count_good, count_bad))

        # Plot counts
        self.plot_counts(count_good, count_bad)

    def plot_counts(self, count_good, count_bad, save_as="./images/plots/review_counts.jpg"):
        """
        Plot counts of positive and negative reviews.
        """
        plt.rcParams['figure.figsize'] = (6, 6)
        plt.bar(0, count_good, width=0.6, label='Positive Reviews', color='limegreen')
        plt.bar(2, count_bad, width=0.6, label='Negative Reviews', color='tomato')
        plt.legend()
        plt.ylabel('Count of Reviews', color='white')
        plt.xlabel('Types of Reviews', color='white')
        plt.xticks(color='white')
        plt.yticks(color='white')
        plt.savefig(save_as, bbox_inches='tight', facecolor='black')

    def display_word_cloud(self, text_data, save_as):
        """
        Display word cloud for given text data.
        """
        wc = self.get_word_cloud_data(text_data)
        plt.subplots(figsize=(10, 10))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis('off')
        plt.savefig(save_as, facecolor='black', bbox_inches='tight')

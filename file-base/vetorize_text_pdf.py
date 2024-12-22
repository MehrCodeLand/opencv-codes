import PyPDF2

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(text1, text2):
    # Convert texts to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])

    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])
    return cosine_sim[0][0]



def main():
    text1 = read_text_file('t1')
    text2 = read_text_file('t2')

    cosine_sim = calculate_cosine_similarity(text1, text2)

    print(cosine_sim)


main()

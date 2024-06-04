# Note: Run this app with streamlit: streamlit run anonymizer_app.py

import logging

import pandas as pd
import streamlit as st
from pandas.errors import ParserError
from text_anonymizer import TextAnonymizer
from text_anonymizer.default_settings import DEFAULT_SETTINGS

logging.getLogger().setLevel(logging.WARN)
# st.set_option('client.toolbarMode', 'viewer')
st.set_page_config(layout="wide", page_title='Anonymisaattori')
 # Vaihda "auto", "expanded" tai "collapsed"

@st.cache_resource
def init_anonymizer():
    return TextAnonymizer(languages=['fi'], debug_mode=False)


# Init anonymizer as singleton
text_anonymizer = init_anonymizer()

# User recognizers
recognizer_options = DEFAULT_SETTINGS.mask_mapppings.keys()


# create a function to render the page
def view_csv_form():
    st.title("CSV anonymizer")
    st.write("Upload CSV file to anonymize")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            separator = st.radio(
                "Separator",
                options=[',', ';'],
            )

        with col2:
            encoding = st.text_input('File encoding', value='utf-8',
                                     help='Character set or file encoding of uploaded file, eg. UTF-8, ISO-8859-1')

        dataframe = None
        try:
            dataframe = pd.read_csv(uploaded_file, sep=separator, dtype=str, encoding=encoding, index_col=0)
            #dataframe = dataframe.astype(str)
        except UnicodeDecodeError as ude:
            st.write("Please verify file encoding.")
            dataframe = None
        except ParserError as ude:
            st.write("Please verify separator.")
            dataframe = None
        except:
            st.write("Unknown error. Please verify that uploaded file is CSV file.")
            dataframe = None

        if dataframe is not None and not dataframe.empty:

            st.write("First 5 rows of uploaded file")
            if dataframe is not None:
                st.write(dataframe.head(5))
            st.write("Select column(s) to be anonymized")

            columns = st.multiselect(
                "Anonymized columns",
                options=dataframe.columns.values.tolist(),
            )

            recognizers = st.multiselect(
                "Optional: select active recognizers. By default all recognizers are active.",
                options=recognizer_options,
                help="Use this if you want to process your text using only subset of recognizable entities. If none is "
                     "selected, by default all recognizers are active."
            )

            # Anonymize first couple rows
            sample = dataframe.head(5).copy()
            for c in columns:
                try:
                    sample[c] = sample[c].apply(
                        lambda x: text_anonymizer.anonymize(x, user_recognizers=recognizers).anonymized_text)
                except:
                    st.write(f"Warning: Column {c} is not suitable for anonymization.")

            st.write("Anonymized sample:")
            st.write(sample)

            @st.cache
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            if st.button('Confirm selection and anonymize uploaded file'):
                for c in columns:
                    try:
                        # TODO: Implement anonymize_dataframe()
                        st.write(f"Anonymization of column {c} in progress...")
                        dataframe[c] = dataframe[c].apply(
                            lambda x: text_anonymizer.anonymize(x, user_recognizers=recognizers).anonymized_text)
                    except:
                        st.write(f"Error: Column {c} is not suitable for anonymization.")
                st.success("Anonymization ready.")
                csv = convert_df(dataframe)

                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='anonymized.csv',
                    mime='text/csv',
                )


def view_text_form():
    recognizers = st.multiselect(
        "Optional: select active recognizers. By default all recognizers are active.",
        options=recognizer_options,
        help="Use this if you want to process your text using only subset of recognizable entities. If none is "
             "selected, by default all recognizers are active."

    )
    text = st.text_area("Text to anonymize")

    if st.button('Anonymize') and text:
        anonymized_text = text_anonymizer.anonymize(text, user_recognizers=recognizers).anonymized_text
        st.success("Anonymization ready.")
        st.text(anonymized_text)


def render_page(page_selection):
    if page_selection == 'csv':
        view_csv_form()
    if page_selection == 'text':
        view_text_form()


# Function to hide the Streamlit hamburger menu and footer
def hide_hamburger_menu():
    '''
    Function to hide the Streamlit hamburger menu and footer with CSS
    They are visible when page is loading.
    '''
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

def load_local_css(file_name):
    with open(file_name) as f:
        st.write(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def create_custom_header_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        header_html = f.read()
    st.markdown(header_html, unsafe_allow_html=True)

# create the app
def main():

    hide_hamburger_menu()

    create_custom_header_from_file('streamlit_app/header_template.html')

    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Loading CSS
    load_local_css("streamlit_app/all.css")

    st.title('Anonymizer service')

    # set the initial page to None
    pages = ['text', 'csv']

    # create a dropdown to select the page
    page_selection = st.selectbox('Select input media type', pages)

    # render the page
    render_page(page_selection)


if __name__ == '__main__':
    main()
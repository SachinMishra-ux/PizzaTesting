import streamlit as st
import pandas as pd

# Load the CSV file
@st.cache_data
def load_data():
    # Replace 'your_file.csv' with the actual path to your CSV file
    df = pd.read_csv('PizzaTesting/dataverification/train_data_fixed.csv', delimiter='|')
    return df

# Function to generate HTML with colored tags
def generate_colored_tags(tags):
    tag_colors = {
        'B-Quantity': 'lightgreen',
        'B-Num': 'lightblue',
        'B-Top': 'lightcoral',
        'I-Top': 'Aquamarine',
        'B-PT': 'BlueViolet',
        'B-DT':'Cornsilk',
        'I-Quantity':'Gold',
        'B-Size': 'Lavender',
        'B-Not': 'LimeGreen',
        'B-Crust':'MediumAquaMarine',
        'B-SPLIT':'Orchid'
        # Add more tags and colors as needed
    }

    colored_tags = [f'<span style="background-color:{tag_colors.get(tag, "white")}">{tag}</span>' for tag in tags]
    return ' '.join(colored_tags)

# Main function to run the app
def main():
    st.title('Simple Streamlit App')

    # Load the data
    data = load_data()

    # Get the total number of rows in the DataFrame
    total_rows = len(data)

    # Initialize page_number using st.session_state
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 1

    # Sidebar for page navigation
    page_number = st.sidebar.number_input('Page Number', value=st.session_state.page_number, min_value=1, max_value=(total_rows // 30) + 1)

    # Calculate the start and end indices for the current page
    start_index = (page_number - 1) * 30
    end_index = min(page_number * 30, total_rows)

    # Display the sentences and corresponding tags one below the other for each row
    for index in range(start_index, end_index):
        st.write(f"{index}: {data['Sentence'][index]}")
        colored_tags = generate_colored_tags(data['Tags'][index].split())
        #st.write(f"{data['Tags'][index]}", unsafe_allow_html=True)
        st.write(f"{index}: {colored_tags}", unsafe_allow_html=True)


if __name__ == '__main__':
    main()

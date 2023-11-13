import streamlit as st
import pandas as pd

# Load the CSV file
@st.cache_data
def load_data():
    # Replace 'your_file.csv' with the actual path to your CSV file
    df = pd.read_csv('train_data_fixed.csv', delimiter='|')
    return df

# Function to display tags with different colors
def highlight_tags(tags):
    tag_colors = {
        'B-Num': 'lightgreen',
        'B-Top': 'lightblue',
        'I-Top': 'lightcoral',
        # Add more tags and colors as needed
    }
    return [f'background-color: {tag_colors.get(tag, "white")}' for tag in tags]

# Main function to run the app
def main():
    st.title('Data Validation')

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

    # Display the data for the current page
    # Display the data for the current page with column names one below the other
    st.dataframe(data.iloc[start_index:end_index].style.apply(highlight_tags, subset=['Tags']))


if __name__ == '__main__':
    main()

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time

def show():
    st.title("✅ Python Essentials for AI")
    
    st.markdown("""
    This module covers the Python fundamentals you'll need for working with AI and ML technologies. 
    We'll explore data manipulation, API interactions, and more.
    """)
    
    # Create tabs for different topics
    tabs = st.tabs(["Basic Python", "Pandas & NumPy", "API Interaction", "Practice"])
    
    # Basic Python tab
    with tabs[0]:
        st.header("Python Fundamentals")
        
        st.subheader("Variables and Data Types")
        code_col1, output_col1 = st.columns([3, 2])
        
        with code_col1:
            st.code("""
# Basic variable types
text = "Hello, AI world!"  # string
number = 42                # integer
decimal = 3.14159          # float
is_active = True           # boolean

# Data structures
my_list = [1, 2, 3, 4, 5]
my_dict = {"name": "LLM", "type": "language model"}
my_tuple = (10, 20, 30)
my_set = {1, 2, 3, 3, 3}  # Duplicates removed

# Print each variable
print(f"String: {text}")
print(f"Integer: {number}")
print(f"Float: {decimal}")
print(f"Boolean: {is_active}")
print(f"List: {my_list}")
print(f"Dictionary: {my_dict}")
print(f"Tuple: {my_tuple}")
print(f"Set: {my_set}")
            """)
        
        with output_col1:
            st.write("Output:")
            st.text("""
String: Hello, AI world!
Integer: 42
Float: 3.14159
Boolean: True
List: [1, 2, 3, 4, 5]
Dictionary: {'name': 'LLM', 'type': 'language model'}
Tuple: (10, 20, 30)
Set: {1, 2, 3}
            """)
        
        st.subheader("Control Flow")
        code_col2, output_col2 = st.columns([3, 2])
        
        with code_col2:
            st.code("""
# If statements
temperature = 75

if temperature > 80:
    status = "It's hot!"
elif temperature > 60:
    status = "It's pleasant."
else:
    status = "It's cold!"

print(f"Temperature status: {status}")

# Loops
print("\\nFor loop example:")
for i in range(5):
    print(f"Iteration {i}")

print("\\nWhile loop example:")
counter = 3
while counter > 0:
    print(f"Countdown: {counter}")
    counter -= 1
print("Blast off!")
            """)
        
        with output_col2:
            st.write("Output:")
            st.text("""
Temperature status: It's pleasant.

For loop example:
Iteration 0
Iteration 1
Iteration 2
Iteration 3
Iteration 4

While loop example:
Countdown: 3
Countdown: 2
Countdown: 1
Blast off!
            """)
        
        st.subheader("Functions")
        code_col3, output_col3 = st.columns([3, 2])
        
        with code_col3:
            st.code("""
# Basic function
def greet(name):
    return f"Hello, {name}!"

# Function with default parameter
def analyze_text(text, analysis_type="length"):
    if analysis_type == "length":
        return len(text)
    elif analysis_type == "words":
        return len(text.split())
    else:
        return "Unknown analysis type"

# Function with multiple return values
def text_metrics(text):
    char_count = len(text)
    word_count = len(text.split())
    return char_count, word_count

# Examples
sample_text = "Large language models are transforming AI."
print(greet("AI Enthusiast"))
print(f"Text length: {analyze_text(sample_text)}")
print(f"Word count: {analyze_text(sample_text, 'words')}")

chars, words = text_metrics(sample_text)
print(f"The text has {chars} characters and {words} words.")
            """)
        
        with output_col3:
            st.write("Output:")
            st.text("""
Hello, AI Enthusiast
Text length: 43
Word count: 6
The text has 43 characters and 6 words.
            """)
            
        st.subheader("List Comprehensions")
        code_col4, output_col4 = st.columns([3, 2])
        
        with code_col4:
            st.code("""
# List comprehension examples
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Square all numbers
squares = [n**2 for n in numbers]
print(f"Squares: {squares}")

# Filter even numbers
evens = [n for n in numbers if n % 2 == 0]
print(f"Even numbers: {evens}")

# Combined operation (square of even numbers)
even_squares = [n**2 for n in numbers if n % 2 == 0]
print(f"Squares of even numbers: {even_squares}")
            """)
        
        with output_col4:
            st.write("Output:")
            st.text("""
Squares: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
Even numbers: [2, 4, 6, 8, 10]
Squares of even numbers: [4, 16, 36, 64, 100]
            """)
    
    # Pandas & NumPy tab
    with tabs[1]:
        st.header("Data Manipulation with Pandas & NumPy")
        
        st.subheader("NumPy Basics")
        code_col5, output_col5 = st.columns([3, 2])
        
        with code_col5:
            st.code("""
import numpy as np

# Create arrays
array1 = np.array([1, 2, 3, 4, 5])
matrix1 = np.array([[1, 2, 3], [4, 5, 6]])

# Array operations
print(f"Original array: {array1}")
print(f"Array + 5: {array1 + 5}")
print(f"Array * 2: {array1 * 2}")
print(f"Array squared: {array1 ** 2}")
print(f"Mean of array: {array1.mean()}")

# Matrix operations
print(f"\\nMatrix shape: {matrix1.shape}")
print(f"Matrix sum: {matrix1.sum()}")
print(f"Matrix column means: {matrix1.mean(axis=0)}")
print(f"Matrix row means: {matrix1.mean(axis=1)}")
            """)
        
        with output_col5:
            st.write("Output:")
            st.text("""
Original array: [1 2 3 4 5]
Array + 5: [6 7 8 9 10]
Array * 2: [2 4 6 8 10]
Array squared: [1 4 9 16 25]
Mean of array: 3.0

Matrix shape: (2, 3)
Matrix sum: 21
Matrix column means: [2.5 3.5 4.5]
Matrix row means: [2. 5.]
            """)
        
        st.subheader("Pandas DataFrames")
        
        # Create a sample dataframe
        df = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'Age': [24, 27, 22, 32, 29],
            'Department': ['AI', 'Data Science', 'ML', 'AI', 'Data Science'],
            'Salary': [75000, 82000, 65000, 95000, 79000]
        })
        
        code_col6, output_col6 = st.columns([3, 2])
        
        with code_col6:
            st.code("""
import pandas as pd

# Create a sample dataframe
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [24, 27, 22, 32, 29],
    'Department': ['AI', 'Data Science', 'ML', 'AI', 'Data Science'],
    'Salary': [75000, 82000, 65000, 95000, 79000]
})

# Basic DataFrame operations
print("DataFrame:")
print(df)

print("\\nSummary statistics:")
print(df.describe())

print("\\nGroupby operation - Average salary by department:")
print(df.groupby('Department')['Salary'].mean())

# Filtering
print("\\nEmployees in AI department:")
print(df[df['Department'] == 'AI'])

print("\\nEmployees with salary > 80000:")
print(df[df['Salary'] > 80000])
            """)
        
        with output_col6:
            st.write("Output:")
            st.dataframe(df)
            
            st.write("Summary statistics:")
            st.dataframe(df.describe())
            
            st.write("Average salary by department:")
            st.dataframe(df.groupby('Department')['Salary'].mean().reset_index())
            
            st.write("Employees in AI department:")
            st.dataframe(df[df['Department'] == 'AI'])
            
            st.write("Employees with salary > 80000:")
            st.dataframe(df[df['Salary'] > 80000])
        
        st.subheader("Data Visualization")
        
        code_col7, output_col7 = st.columns([1, 1])
        
        with code_col7:
            st.code("""
# Basic chart with Pandas and Matplotlib
import matplotlib.pyplot as plt

# Bar chart of salaries
df.plot(kind='bar', x='Name', y='Salary', figsize=(10, 6))
plt.title('Salary Comparison')
plt.xlabel('Employee')
plt.ylabel('Salary ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
            """)
        
        with output_col7:
            # Use Streamlit to create the chart
            st.bar_chart(df.set_index('Name')['Salary'])
            st.caption("Salary Comparison by Employee")
        
    # API Interaction tab
    with tabs[2]:
        st.header("Interacting with APIs")
        
        st.info("APIs (Application Programming Interfaces) are crucial for AI development as they allow us to access external data and services.")
        
        st.subheader("Making API Requests")
        code_col8, output_col8 = st.columns([3, 2])
        
        with code_col8:
            st.code("""
import requests
import json

# GET request to a public API
response = requests.get('https://jsonplaceholder.typicode.com/todos/1')
print(f"Status code: {response.status_code}")

# Parse JSON response
data = response.json()
print("\\nResponse data:")
print(json.dumps(data, indent=2))

# GET with parameters
params = {'limit': 2}
response = requests.get('https://jsonplaceholder.typicode.com/posts', params=params)
print("\\nGET with parameters:")
print(json.dumps(response.json(), indent=2))

# POST request
new_post = {
    'title': 'Generative AI Workshop',
    'body': 'Learning about LLMs and RAG',
    'userId': 1
}
response = requests.post('https://jsonplaceholder.typicode.com/posts', json=new_post)
print("\\nPOST request result:")
print(json.dumps(response.json(), indent=2))
            """)
        
        with output_col8:
            st.write("Output:")
            with st.spinner("Making API requests..."):
                try:
                    # First request
                    response1 = requests.get('https://jsonplaceholder.typicode.com/todos/1')
                    st.text(f"Status code: {response1.status_code}")
                    
                    data1 = response1.json()
                    st.text("\nResponse data:")
                    st.json(data1)
                    
                    # Second request
                    params = {'limit': 2}
                    response2 = requests.get('https://jsonplaceholder.typicode.com/posts', params=params)
                    st.text("\nGET with parameters:")
                    st.json(response2.json())
                    
                    # Third request
                    new_post = {
                        'title': 'Generative AI Workshop',
                        'body': 'Learning about LLMs and RAG',
                        'userId': 1
                    }
                    response3 = requests.post('https://jsonplaceholder.typicode.com/posts', json=new_post)
                    st.text("\nPOST request result:")
                    st.json(response3.json())
                except Exception as e:
                    st.error(f"Error making API requests: {e}")
        
        st.subheader("Error Handling in API Calls")
        
        code_col9, _ = st.columns([1, 0.2])
        
        with code_col9:
            st.code("""
import requests

def safe_api_call(url, method='get', data=None, timeout=5):
    try:
        if method.lower() == 'get':
            response = requests.get(url, timeout=timeout)
        elif method.lower() == 'post':
            response = requests.post(url, json=data, timeout=timeout)
        else:
            return {'error': 'Method not supported'}
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Return the JSON data
        return response.json()
    
    except requests.exceptions.HTTPError as errh:
        return {'error': f"HTTP Error: {errh}"}
    except requests.exceptions.ConnectionError as errc:
        return {'error': f"Connection Error: {errc}"}
    except requests.exceptions.Timeout as errt:
        return {'error': f"Timeout Error: {errt}"}
    except requests.exceptions.RequestException as err:
        return {'error': f"Request Exception: {err}"}
    except ValueError as err:
        return {'error': f"Value Error (likely invalid JSON): {err}"}

# Example usage
result = safe_api_call('https://jsonplaceholder.typicode.com/posts/1')
print(result)

# Example with error
result_error = safe_api_call('https://nonexistent-api.example.com')
print(result_error)
            """)
        
        st.info("Proper error handling is essential when working with external APIs to make your applications robust.")
    
    # Practice tab
    with tabs[3]:
        st.header("Practice Exercises")
        
        st.subheader("Exercise 1: Data Filtering")
        
        # Example data
        data = pd.DataFrame({
            'product': ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard', 'Mouse', 'Headphones'],
            'category': ['Computing', 'Mobile', 'Mobile', 'Computing', 'Accessories', 'Accessories', 'Audio'],
            'price': [1200, 800, 400, 350, 100, 50, 150],
            'stock': [15, 25, 10, 20, 50, 100, 30]
        })
        
        st.write("Sample Product Data:")
        st.dataframe(data)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.write("Task 1: Filter products with price > 200")
            user_code1 = st.text_area("Your code:", height=100, key="exercise1_1", 
                                    placeholder="# Write your code to filter the data\n# Example: high_price_products = data[data['price'] > 200]")
            
            if st.button("Run Code", key="run1"):
                try:
                    if user_code1.strip():
                        # Create a local environment with the data
                        local_vars = {'data': data}
                        
                        # Execute the user's code
                        exec(user_code1, {}, local_vars)
                        
                        # Check if any results were created
                        results_found = False
                        for var_name, var_value in local_vars.items():
                            if var_name != 'data' and isinstance(var_value, pd.DataFrame):
                                st.write(f"Result ({var_name}):")
                                st.dataframe(var_value)
                                results_found = True
                        
                        if not results_found:
                            st.warning("No DataFrame results found. Make sure you create a variable to store your filtered data.")
                    else:
                        st.warning("Please enter some code before running.")
                except Exception as e:
                    st.error(f"Error executing code: {e}")
        
        with col_b:
            st.write("Task 2: Find the average price by category")
            user_code2 = st.text_area("Your code:", height=100, key="exercise1_2", 
                                    placeholder="# Write your code to calculate average price by category\n# Example: avg_by_category = data.groupby('category')['price'].mean()")
            
            if st.button("Run Code", key="run2"):
                try:
                    if user_code2.strip():
                        # Create a local environment with the data
                        local_vars = {'data': data}
                        
                        # Execute the user's code
                        exec(user_code2, {}, local_vars)
                        
                        # Check if any results were created
                        results_found = False
                        for var_name, var_value in local_vars.items():
                            if var_name != 'data' and (isinstance(var_value, pd.DataFrame) or isinstance(var_value, pd.Series)):
                                st.write(f"Result ({var_name}):")
                                st.dataframe(var_value)
                                results_found = True
                        
                        if not results_found:
                            st.warning("No DataFrame or Series results found. Make sure you create a variable to store your results.")
                    else:
                        st.warning("Please enter some code before running.")
                except Exception as e:
                    st.error(f"Error executing code: {e}")
        
        st.subheader("Exercise 2: NumPy Array Operations")
        
        col_c, col_d = st.columns(2)
        
        with col_c:
            st.write("Create a 5x5 matrix and find its transpose, sum of each row, and the maximum value in each column.")
            user_code3 = st.text_area("Your code:", height=150, key="exercise2", 
                                    placeholder="# Example:\n# import numpy as np\n# matrix = np.random.randint(1, 100, size=(5, 5))\n# transpose = matrix.transpose()\n# row_sums = matrix.sum(axis=1)\n# col_maxes = matrix.max(axis=0)")
            
            if st.button("Run Code", key="run3"):
                try:
                    if user_code3.strip():
                        # Create a local environment
                        local_vars = {}
                        
                        # Execute the user's code
                        exec('import numpy as np\n' + user_code3, {}, local_vars)
                        
                        # Look for matrix and results
                        for var_name, var_value in local_vars.items():
                            if isinstance(var_value, np.ndarray):
                                st.write(f"Result ({var_name}):")
                                st.write(var_value)
                    else:
                        st.warning("Please enter some code before running.")
                except Exception as e:
                    st.error(f"Error executing code: {e}")
        
        with col_d:
            st.write("Example Solution:")
            st.code("""
import numpy as np

# Create a 5x5 matrix with random integers
matrix = np.random.randint(1, 100, size=(5, 5))
print("Original Matrix:")
print(matrix)

# Find the transpose
transpose = matrix.transpose()
print("\nTranspose:")
print(transpose)

# Find the sum of each row
row_sums = matrix.sum(axis=1)
print("\nRow sums:")
print(row_sums)

# Find the maximum value in each column
col_maxes = matrix.max(axis=0)
print("\nColumn maximums:")
print(col_maxes)
            """)
            
            # Show a sample output
            np.random.seed(42)  # For reproducibility
            sample_matrix = np.random.randint(1, 100, size=(5, 5))
            
            st.write("Sample Output:")
            st.write("Original Matrix:")
            st.write(sample_matrix)
            
            st.write("Transpose:")
            st.write(sample_matrix.transpose())
            
            st.write("Row sums:")
            st.write(sample_matrix.sum(axis=1))
            
            st.write("Column maximums:")
            st.write(sample_matrix.max(axis=0))
            
    st.markdown("---")
    st.info("This module covered essential Python concepts needed for AI development. In the next module, we'll explore the basics of Generative AI and LLMs.")

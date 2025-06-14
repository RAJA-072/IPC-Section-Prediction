# IPC Query Processing System

## Overview
This project utilizes a dataset of Indian Penal Code (IPC) sections to provide relevant legal references based on user queries. It employs Google Gemini API for natural language processing (NLP) to summarize user input and match it to the appropriate IPC section.

## Dataset Description
The dataset (`sections_desc.csv`) contains details of IPC sections, including:
- **Title**: The name of the IPC section
- **Link**: A reference URL for further details.
- **Section**: The IPC section number.
- **Description**: A brief explanation of the IPC section

### Sample Data
| Title                                          | Link                                    | Section   | Description |
|-----------------------------------------------|----------------------------------------|-----------|-------------|
| IPC Section 1 » Title and extent of operation | http://devgan.in/ipc/section/1/        | Section 1  | This Act shall be called the Indian Penal Code... |
| IPC Section 33 » "Act". "Omission". | http://devgan.in/ipc/section/33/       | Section 33 | The word “act” denotes as well as series of acts... |

## Query Processing Workflow
1. **User Input**: The user submits a description of an incident.
2. **Text Summarization**: The Gemini API processes the query to generate a summary and conclusion.
3. **NLP Matching**: The summarized query is analyzed to identify the most relevant IPC sections.
4. **Output**: The system returns the IPC section number, title, description, and reference link.

### Example Query
#### Input:
```
A man forcefully entered his neighbor’s house, broke a window to gain entry, and stole valuable jewelry while the owner was away. The victim suffered emotional distress.
```
#### Generated Summary:
```
A burglary incident involving forced entry, property damage, and theft, leading to emotional distress for the victim.
```
#### Suggested IPC Sections:
- **Section 441**: Criminal Trespass
- **Section 378**: Theft
- **Section 425**: Mischief

## Setup Instructions
### Prerequisites
- Python 3.8+
- `pip install google-generativeai pandas openai`
- A valid Google Gemini API key

### Usage
```python
import google.generativeai as genai
import pandas as pd

# Load IPC dataset
df = pd.read_csv("sections_desc.csv")

# Configure Gemini API
genai.configure(api_key='YOUR_GOOGLE_API_KEY')
model = genai.GenerativeModel('gemini-1.5-flash')

# Process user query
query = "A man broke into a house and stole valuables."
response = model.generate_content(query)
print("Summary:", response.text)
```

## Future Improvements
- Improve NLP matching using embeddings.
- Implement a search-based retrieval method for faster IPC section identification.
- Expand the dataset with more detailed legal references.




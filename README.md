# AI Agent Platform with CSV Data Analysis

This application enables users to create and interact with AI agents that can analyze documents including CSV files with advanced data analysis capabilities.

## Features

- Create multiple AI agents with customized system prompts and models
- Upload and process various document types:
  - PDF documents for text extraction
  - Word documents (DOCX) for content analysis
  - CSV files for data analysis and statistical processing
  - Excel files (XLSX) for spreadsheet analysis
- Retrieve information from documents using natural language queries
- Advanced RAG (Retrieval-Augmented Generation) system for context-aware responses
- Flexible deployment options with standalone frontend and backend

## CSV Data Analysis Capabilities

The system provides robust analysis for CSV data including:

- Automatic statistical summary (mean, median, standard deviation, etc.)
- Column type detection and data profiling
- Categorical data analysis with unique value identification
- Numerical data processing with appropriate calculations
- Performance of quantitative analysis in response to user queries
- Data-driven insights based on the uploaded CSV files

## Technical Architecture

The application consists of two main components:

1. **Node.js Frontend Server**: React-based interface with shadcn/UI components
2. **Python Backend Server**: FastAPI service with advanced document processing capabilities

### Key Technologies:

- **RAG Processing**: LangGraph-based retrieval, generation, and validation pipeline
- **Document Storage**: ChromaDB vector database (with fallback to JSON storage)
- **Data Analysis**: Pandas for CSV and Excel data processing
- **Text Extraction**: PyPDF2 and python-docx for document parsing

## Installation

Please see the `installation_guide.md` file for detailed setup instructions.

## Flexible Deployment Options

This application supports multiple deployment scenarios:

- **All-in-One**: Start both frontend and backend with `./start.sh`
- **Separate Components**: Run `./start-python.sh` for the backend and `./start-frontend.sh` for the frontend
- **Frontend Only**: Run with an existing Python backend using `./start.sh --skip-python`

## Sample Data

The repository includes sample data files for testing the analysis capabilities:
- `attached_assets/sp500epsest.xlsx`: Standard & Poor's 500 EPS estimates data

## Usage

1. Create an agent by clicking the "Create Agent" button
2. Upload documents in the Document Management section
3. Ask questions about the data using natural language
4. View the AI-generated responses based on your data

Example questions for CSV analysis:
- "What are the total sales by product?"
- "Which region had the highest revenue?"
- "What's the average number of units sold across all products?"
- "Show me sales trends by date"
- "Compare revenue between different regions"

## System Requirements

- Python 3.9+ with required packages (see installation_guide.md)
- Node.js 16+ with npm
- OpenAI API key
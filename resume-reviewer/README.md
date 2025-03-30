# Resume Reviewer

A web application that uses Google's Gemini AI to analyze resumes and provide detailed feedback.

## Features

- User authentication (register, login, logout)
- Resume upload and analysis
- AI-powered feedback on resume structure, content, and ATS compatibility
- Dashboard to view all uploaded resumes and their analyses

## Tech Stack

- **Backend**: Flask, SQLAlchemy
- **Frontend**: HTML, CSS, JavaScript
- **AI**: Google Gemini API
- **Database**: SQLite

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Gemini API key: `GEMINI_API_KEY=your_api_key_here`
   - Set a secret key for Flask: `SECRET_KEY=your_secret_key_here`

### Running the Application

1. Start the Flask server:
   ```
   python app.py
   ```
2. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Register for an account
2. Log in to your account
3. Upload your resume (PDF, DOCX, or TXT format)
4. View the AI-generated analysis and feedback
5. Make improvements to your resume based on the suggestions

## Project Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates
- `static/`: CSS, JavaScript, and image files
- `uploads/`: Directory for uploaded resumes

## License

MIT
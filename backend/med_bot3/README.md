# Med-Bot AI Educational Assistant

An AI-powered educational chatbot that processes documents, images, and text to provide intelligent responses about uploaded content. Built with Google Gemini AI and Supabase for scalable document processing and vector search.

## 🚀 Features

- **Document Processing**: Upload PDFs, Word docs, images, and text files
- **AI-Powered Chat**: Ask questions about your uploaded content
- **Vector Search**: Fast similarity search using pgvector embeddings
- **Voice Interaction**: Speech-to-text question input
- **Multi-User Support**: User-isolated data with secure authentication
- **Exam Generation**: Create practice exams from course content
- **Flashcard Creation**: Generate study flashcards automatically

## 🏗️ Architecture

- **Frontend**: Python-based CLI interface
- **AI Models**: Google Gemini Pro for text generation and embeddings
- **Database**: Supabase with PostgreSQL and pgvector extension
- **Storage**: Supabase Storage for file uploads
- **Authentication**: Supabase Auth with Row Level Security

## 📋 Prerequisites

- Python 3.8+
- Supabase account and project
- Google AI Studio API key
- PostgreSQL with pgvector extension (handled by Supabase)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/med-bot.git
   cd med-bot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys and database URLs
   ```

5. **Set up database schema**
   ```bash
   # Apply database migrations (see Database Setup section)
   python fix_missing_schema.py
   ```

6. **Verify setup**
   ```bash
   python check_status.py
   ```

## 🗄️ Database Setup

The project includes comprehensive database setup and management tools:

### Initial Setup
```bash
# Check current database status
python check_status.py

# Fix any missing schema components
python fix_missing_schema.py

# Set up storage buckets and verify configuration
python setup_storage_and_db.py
```

### Health Monitoring
```bash
# Run database health check
python health_check.py

# Analyze performance and optimization
python optimize_database.py
```

### Backup & Recovery
```bash
# Create backup
./backup_database.sh

# Restore from backup
./restore_database.sh backup_name
```

## 🚦 Usage

### Basic Chatbot
```bash
python chatbot.py
```

Available commands:
- `upload <file_path>` - Upload and process a document
- `list` - List all uploaded documents
- `ask <question>` - Ask a question about your documents
- `voice` - Ask a question using speech input
- `exit` - Exit the chatbot

### Exam Generator
```bash
python exam-feature/exam-generator.py
```

### RAG Pipeline (Direct)
```bash
# Process a PDF and ask a question
python rag_pipeline.py document.pdf "What is the main topic?"

# Process an image
python rag_pipeline.py image.png "Describe this image" --type image
```

## 📊 Database Schema

The database includes the following main tables:

- **profiles**: User profiles extending Supabase auth
- **notebooks**: Content organization containers
- **documents**: Document chunks with vector embeddings (768-dim)
- **conversations**: Chat conversation threads
- **messages**: Individual chat messages
- **user_settings**: User preferences and configurations

## 🔒 Security

- **Row Level Security (RLS)**: All tables secured with user-specific access
- **API Key Management**: Secure storage of credentials
- **Data Isolation**: Users can only access their own data
- **Encrypted Storage**: All data encrypted at rest and in transit

## 📈 Performance

- **Vector Search**: Optimized with HNSW indexing for fast similarity search
- **Connection Pooling**: Configured for optimal database performance
- **Caching**: Efficient embedding and content caching
- **Monitoring**: Built-in health checks and performance monitoring

## 🔧 Development

### Running Tests
```bash
# Test database connectivity
python test_db_connection.py

# Detailed Supabase testing
python test_supabase_detailed.py
```

### Database Management
```bash
# Performance optimization analysis
python optimize_database.py

# Manual schema fixes
python fix_missing_schema.py
```

## 📚 Documentation

- **Database Assessment**: See `DATABASE_ASSESSMENT_REPORT.md`
- **Backup Procedures**: See `BACKUP_RECOVERY.md`
- **API Documentation**: See individual module docstrings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the documentation files
2. Run `python check_status.py` for diagnostics
3. Review the database assessment report
4. Check the backup and recovery procedures

## 🔄 Recent Updates

- ✅ Complete database assessment and optimization
- ✅ Comprehensive backup and recovery system
- ✅ Performance monitoring and health checks
- ✅ Security audit and RLS implementation
- ✅ Storage bucket configuration for file uploads

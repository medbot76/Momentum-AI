// Configuration for API endpoints
const getApiBaseUrl = () => {
  // Check if we're running in Replit environment
  if (window.location.hostname.includes('replit.dev')) {
    // For Replit, the backend should be accessible on the same hostname but port 5000
    // The URL format is: https://hostname:5000
    const hostname = window.location.hostname;
    return `https://${hostname}:5000`;
  }
  
  // For local development
  return 'http://localhost:5000';
};

export const API_BASE_URL = getApiBaseUrl();
export const API_ENDPOINTS = {
  CHAT: `${API_BASE_URL}/api/chat`,
  CHAT_STREAM: `${API_BASE_URL}/api/chat/stream`,
  UPLOAD: `${API_BASE_URL}/api/upload`,
  DOCUMENTS: `${API_BASE_URL}/api/documents`,
  TTS: `${API_BASE_URL}/api/tts`,
  STT: `${API_BASE_URL}/api/stt`,
  HEALTH: `${API_BASE_URL}/api/health`,
  EXAM_PDF: `${API_BASE_URL}/api/exam-pdf`,
  FLASHCARDS: `${API_BASE_URL}/api/flashcards`,
  MODELS: `${API_BASE_URL}/api/models`, // List available models
  SET_MODEL: `${API_BASE_URL}/api/model`, // Set current model
  STUDYPLAN: `${API_BASE_URL}/api/studyplan`,
  STUDYPLAN_UPLOAD: `${API_BASE_URL}/api/studyplan-upload`,
  STUDYPLAN_ADD_TO_CALENDAR: `${API_BASE_URL}/api/studyplan/add-to-calendar`,
}; 
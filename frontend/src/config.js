// Configuration for API endpoints and Replit handoff
const getReplitAppUrl = () => {
  return process.env.REACT_APP_REPLIT_APP_URL || '';
};

const getApiBaseUrl = () => {
  // Check for environment variable first (for production/EC2)
  if (import.meta?.env?.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  
  // Check if we're running in Replit environment
  if (window.location.hostname.includes('replit.dev')) {
    // For Replit, the backend should be accessible on the same hostname but port 5000
    // The URL format is: https://hostname:5000
    const hostname = window.location.hostname;
    return `https://${hostname}:5000`;
  }
  
  // For EC2/production: use same origin (assuming backend is proxied or on same domain)
  // If backend is on a different port, you'll need to set VITE_API_BASE_URL env var
  if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
    // For production, assume backend is on same origin or proxied
    return window.location.origin;
  }
  
  // For local development
  return 'http://localhost:5000';
};

export const API_BASE_URL = getApiBaseUrl();
export { getReplitAppUrl };
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
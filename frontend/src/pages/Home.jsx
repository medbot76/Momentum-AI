"use client"

import { Maximize2, Mic, MicOff, Plus, Settings, X } from 'lucide-react';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import '../App.css';

import { useNavigate } from 'react-router-dom';
import { API_ENDPOINTS } from '../config';
import supabase from '../helper/supabaseClient';
import Chatbot from './Chatbot';

// Placeholder chat history grouped by date
const chatHistory = [
    {
      date: 'Today',
      chats: [
        { id: 1, title: 'Biology Q&A' },
        { id: 2, title: 'Chemistry Notes' },
      ],
    },
    {
      date: 'Yesterday',
      chats: [
        { id: 3, title: 'Physics Flashcards' },
      ],
    },
  ];
  
  // Enhanced Avatar Component - Apple-inspired Recording Animation
  const AnimatedAvatar = ({ isLoading, onVoiceInput, isMuted, setIsMuted, isFullscreen, setIsFullscreen, onClose }) => {
    const [isRecording, setIsRecording] = useState(false);
    const [isExiting, setIsExiting] = useState(false);
    const [showPersonas, setShowPersonas] = useState(false);
    const [selectedPersona, setSelectedPersona] = useState(0);
    
    // Text cycling state
    const [currentTextIndex, setCurrentTextIndex] = useState(0);
    const [isTextTransitioning, setIsTextTransitioning] = useState(false);
    
    // Array of messages to cycle through
    const voiceMessages = [
      "Talk to me with your voice",
      "Ask me anything out loud", 
      "Voice chat with AI",
      "Record lectures & chat",
      "Speak naturally to learn",
      "Voice-powered conversations"
    ];

    // Text cycling effect
    useEffect(() => {
      if (!isRecording && !isMuted) {
        const interval = setInterval(() => {
          setIsTextTransitioning(true);
          
          setTimeout(() => {
            setCurrentTextIndex((prev) => (prev + 1) % voiceMessages.length);
            setIsTextTransitioning(false);
          }, 500); // Increased from 300ms to 500ms for smoother transition
          
        }, 4500); // Increased from 3000ms to 4500ms for slower cycling

        return () => clearInterval(interval);
      }
    }, [isRecording, isMuted, voiceMessages.length]);
  
    // Add refs for strict centering control
    const sliderRef = useRef(null);
    const containerRef = useRef(null);
    const scrollTimeoutRef = useRef(null);
  
    // Persona data with names and profile pictures
    const personas = [
      {
        id: 0,
        name: "Dr. Sarah Chen",
        specialty: "Cardiothoracic Surgery",
        avatar: "👩‍⚕️",
        color: "from-blue-500 to-blue-600"
      },
      {
        id: 1,
        name: "Dr. Michael Rodriguez",
        specialty: "Neurology",
        avatar: "👨‍🔬",
        color: "from-green-500 to-green-600"
      },
      {
        id: 2,
        name: "Dr. Emily Watson",
        specialty: "Pediatrics",
        avatar: "👩‍💼",
        color: "from-purple-500 to-purple-600"
      },
      {
        id: 3,
        name: "Dr. James Kim",
        specialty: "Orthopedics",
        avatar: "👨‍⚕️",
        color: "from-orange-500 to-orange-600"
      },
      {
        id: 4,
        name: "Dr. Lisa Thompson",
        specialty: "Dermatology",
        avatar: "👩‍🎓",
        color: "from-pink-500 to-pink-600"
      }
    ];
  
    // Strict centering function
    const centerCurrentSlide = useCallback(() => {
      if (!sliderRef.current || !containerRef.current) return;
      
      const container = containerRef.current;
      const slider = sliderRef.current;
      const slides = slider.children;
      
      if (slides.length === 0) return;
      
      const containerRect = container.getBoundingClientRect();
      const containerCenter = containerRect.width / 2;
      
      let closestSlide = null;
      let closestDistance = Infinity;
      let closestIndex = 0;
      
      // Find the slide closest to center
      Array.from(slides).forEach((slide, index) => {
        const slideRect = slide.getBoundingClientRect();
        const slideCenter = slideRect.left - containerRect.left + slideRect.width / 2;
        const distance = Math.abs(slideCenter - containerCenter);
        
        if (distance < closestDistance) {
          closestDistance = distance;
          closestSlide = slide;
          closestIndex = index;
        }
      });
      
      if (closestSlide) {
        // Calculate the exact scroll position to center this slide
        const slideRect = closestSlide.getBoundingClientRect();
        const slideCenter = slideRect.left - containerRect.left + slideRect.width / 2;
        const offsetToCenter = slideCenter - containerCenter;
        
        // Smooth scroll to center position
        container.scrollBy({
          left: offsetToCenter,
          behavior: 'smooth'
        });
        
        // Update selected persona
        if (closestIndex !== selectedPersona) {
          setSelectedPersona(closestIndex);
        }
      }
    }, [selectedPersona]);
  
    // Handle scroll events with debouncing for strict centering
    const handleScroll = useCallback(() => {
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
      
      scrollTimeoutRef.current = setTimeout(() => {
        centerCurrentSlide();
      }, 150); // Short delay to allow smooth scrolling
    }, [centerCurrentSlide]);
  
    // Auto-center on mount and when personas change
    useEffect(() => {
      if (showPersonas && sliderRef.current && containerRef.current) {
        // Small delay to ensure DOM is ready
        const timer = setTimeout(() => {
          centerCurrentSlide();
        }, 100);
        
        return () => clearTimeout(timer);
      }
    }, [showPersonas, centerCurrentSlide]);
  
    // Cleanup scroll timeout
    useEffect(() => {
      return () => {
        if (scrollTimeoutRef.current) {
          clearTimeout(scrollTimeoutRef.current);
        }
      };
    }, []);
  
    const handleRecordingToggle = () => {
      if (isRecording) {
        // Immediately stop recording state for button
        setIsRecording(false);
        // Start unified exit animation for visual elements
        setIsExiting(true);
        // After exit animation completes, reset exit state
        setTimeout(() => {
          setIsExiting(false);
        }, 600); // Match animation duration
      } else {
        setIsRecording(true);
      }
    };
  
    const handleMicToggle = () => {
      setIsMuted(!isMuted);
    };
  
    const handleFullscreenToggle = () => {
      setIsFullscreen(!isFullscreen);
    };
  
    const handleClose = () => {
      onClose();
    };
  
    const handlePersonaClick = () => {
      setShowPersonas(true);
    };
  
    const handlePersonaSelect = (personaId) => {
      // Find the index of the selected persona
      const personaIndex = personas.findIndex(p => p.id === personaId);
      if (personaIndex !== -1) {
        setSelectedPersona(personaIndex);
        // Programmatically scroll to center the selected persona
        scrollToPersona(personaIndex);
      }
      setShowPersonas(false);
    };
  
    // Function to scroll to a specific persona by index
    const scrollToPersona = useCallback((index) => {
      if (!containerRef.current || !sliderRef.current) return;
      
      const container = containerRef.current;
      const slideWidth = container.clientWidth; // Each slide is 100vw
      const targetScrollLeft = index * slideWidth;
      
      container.scrollTo({
        left: targetScrollLeft,
        behavior: 'smooth'
      });
    }, []);
  
    if (showPersonas) {
      return (
        <div className={`relative flex flex-col items-center justify-center h-full w-full overflow-hidden ${isFullscreen ? 'fixed inset-0 z-50 bg-white' : ''}`}>
          {/* Top Controls */}
          <div className="absolute top-6 left-0 right-0 flex justify-between items-center px-6 z-10">
            {/* Back Button - Top Left */}
            <button 
              onClick={() => setShowPersonas(false)}
              className="w-10 h-10 bg-white/90 backdrop-blur-sm rounded-full shadow-lg flex items-center justify-center hover:shadow-xl transition-all duration-200 border border-gray-100/50 hover:scale-105 transform apple-button-hover group"
              title="Back to voice panel"
            >
              <svg className="w-5 h-5 text-gray-700 group-hover:text-gray-900 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
  
            {/* Close Button - Top Right */}
            <button 
              onClick={handleClose}
              className="w-10 h-10 bg-white/90 backdrop-blur-sm rounded-full shadow-lg flex items-center justify-center hover:shadow-xl transition-all duration-200 border border-gray-100/50 hover:scale-105 transform apple-button-hover group"
              title="Close voice panel"
            >
              <X className="w-5 h-5 text-gray-700 group-hover:text-gray-900 transition-colors" />
            </button>
          </div>
  
          {/* Persona Slider Container - Strict Centering */}
          <div className="w-full flex-1 flex items-center justify-center">
            <div 
              ref={containerRef}
              className="overflow-x-auto scrollbar-hide persona-slider-container-strict"
              style={{ 
                scrollbarWidth: 'none', 
                msOverflowStyle: 'none',
                width: '100vw',
                maxWidth: '100%'
              }}
              onScroll={handleScroll}
            >
              <div 
                ref={sliderRef}
                className="flex gap-0 persona-slider-track"
                style={{
                  width: `${personas.length * 100}vw`,
                  paddingLeft: '50vw',
                  paddingRight: '50vw'
                }}
              >
                {personas.map((persona, index) => (
                  <div
                    key={persona.id}
                    data-persona-id={persona.id}
                    className="flex-shrink-0 flex flex-col items-center cursor-pointer justify-center persona-slide-item-strict"
                    style={{ 
                      width: '100vw',
                      maxWidth: '100vw'
                    }}
                    onClick={() => handlePersonaSelect(persona.id)}
                  >
                    {/* Persona Circle - Always perfectly centered */}
                    <div className={`w-48 h-48 rounded-full bg-black flex items-center justify-center shadow-lg transition-all duration-300 relative persona-circle-large ${
                      index === selectedPersona 
                        ? 'ring-4 ring-red-500 ring-offset-2' 
                        : ''
                    }`}>
                      {/* Profile picture placeholder - black circle for now */}
                      <div className="w-44 h-44 rounded-full bg-black"></div>
                    </div>
                    
                    {/* Title underneath - positioned same as voice panel controls */}
                    <div className="mt-8 text-center">
                      <h3 className="font-semibold text-gray-800 text-lg mb-2">{persona.name}</h3>
                      <p className="text-gray-500 text-sm">{persona.specialty}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
  
          {/* Bottom section with indicators and instructions */}
          <div className="mb-8">
            {/* Scroll Indicators */}
            <div className="flex gap-2 justify-center mb-4">
              {personas.map((_, index) => (
                <div
                  key={index}
                  className={`w-2 h-2 rounded-full transition-all duration-300 ${
                    index === selectedPersona ? 'bg-red-400 scale-125' : 'bg-gray-200'
                  }`}
                ></div>
              ))}
            </div>
  
            {/* Instructions */}
            <p className="text-gray-400 text-xs text-center">
              Scroll horizontally to browse personas • Tap to select
            </p>
          </div>
        </div>
      );
    }
  
    return (
      <div className={`relative flex flex-col items-center justify-center h-full w-full ${isFullscreen ? 'fixed inset-0 z-50 bg-white' : ''}`}>
        {/* Top Controls - Only show when not fullscreen or when fullscreen */}
        <div className="absolute top-6 left-0 right-0 flex justify-between items-center px-6 z-10">
          {/* Close Button - Top Left */}
          <button 
            onClick={handleClose}
            className="w-10 h-10 bg-white/90 backdrop-blur-sm rounded-full shadow-lg flex items-center justify-center hover:shadow-xl transition-all duration-200 border border-gray-100/50 hover:scale-105 transform apple-button-hover group"
            title="Close voice panel"
          >
            <X className="w-5 h-5 text-gray-700 group-hover:text-gray-900 transition-colors" />
          </button>
  
          {/* Fullscreen Button - Top Right */}
          <button 
            onClick={handleFullscreenToggle}
            className="w-10 h-10 bg-white/90 backdrop-blur-sm rounded-full shadow-lg flex items-center justify-center hover:shadow-xl transition-all duration-200 border border-gray-100/50 hover:scale-105 transform apple-button-hover group"
            title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
          >
            <Maximize2 className="w-5 h-5 text-gray-700 group-hover:text-gray-900 transition-colors" />
          </button>
        </div>
  
        {/* Main Avatar Circle - Apple-inspired bouncy animation */}
        <div className={`w-48 h-48 rounded-full flex items-center justify-center transform transition-all duration-500 ${
          isExiting
            ? 'bg-red-500 scale-100 recording-circle-exit'
            : isRecording 
            ? 'bg-red-500 scale-110 recording-bounce recording-glow' 
            : 'bg-black scale-100'
        } apple-butter-smooth`}>
          {/* White circle in the middle when recording - perfectly centered with unified timing */}
          {(isRecording || isExiting) && (
            <div className={`w-6 h-6 bg-white rounded-full absolute ${
              isExiting ? 'recording-dot-exit' : 'recording-dot-bounce'
            }`}></div>
          )}
        </div>
  
        {/* Control buttons - positioned below the circle */}
        <div className="mt-8 flex gap-4">        
          {/* Left Button - Personas */}
          <button 
            onClick={handlePersonaClick}
            className="w-12 h-12 bg-white rounded-full shadow-lg flex items-center justify-center hover:shadow-xl transition-all duration-200 border border-gray-100 hover:scale-105 transform apple-button-hover"
            title="Choose Persona"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" className="text-gray-700">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" stroke="currentColor" strokeWidth="1.5"/>
              <circle cx="12" cy="7" r="4" stroke="currentColor" strokeWidth="1.5" fill="none"/>
            </svg>
          </button>
          
          {/* Recording Button - Center with Apple-style bounce */}
          <button 
            onClick={handleRecordingToggle}
            className={`w-12 h-12 rounded-full shadow-lg flex items-center justify-center transition-all duration-200 border transform ${
              isRecording 
                ? 'bg-red-500 border-red-300 shadow-red-200 scale-110 recording-button-active' 
                : 'bg-white border-gray-100 shadow-gray-200 hover:scale-110 apple-button-hover'
            }`}
            title={isRecording ? 'Stop recording lecture' : 'Start recording lecture'}
          >
            {isRecording ? (
              <div className="w-3 h-3 bg-white rounded-sm recording-stop-icon"></div>
            ) : (
              <div className="w-3 h-3 bg-red-500 rounded-full recording-start-icon"></div>
            )}
          </button>
          
          {/* Mic Button - Right (was Settings) */}
          <button 
            onClick={handleMicToggle}
            className={`w-12 h-12 rounded-full shadow-lg flex items-center justify-center hover:shadow-xl transition-all duration-200 border hover:scale-105 transform apple-button-hover ${
              isMuted 
                ? 'bg-red-500 border-red-300 text-white shadow-red-200' 
                : 'bg-white border-gray-100 text-gray-700 shadow-gray-200'
            }`}
            title={isMuted ? 'Unmute microphone' : 'Mute microphone'}
          >
            {isMuted ? (
              <MicOff className="w-5 h-5" />
            ) : (
              <Mic className="w-5 h-5" />
            )}
          </button>
        </div>
  
        {/* Status text with bouncy animation */}
        <div className="mt-4 text-center">
          {isRecording ? (
            <p className="text-red-500 text-sm font-medium recording-text-bounce">
              🎤 Recording lecture...
            </p>
          ) : isMuted ? (
            <p className="text-red-500 text-sm font-medium">
              🔇 Microphone muted
            </p>
          ) : (
            <p className={`text-gray-500 text-sm font-medium transition-all duration-1000 ease-in-out hover:text-gray-700 ${
              isTextTransitioning ? 'opacity-0 transform translate-y-3 scale-95' : 'opacity-100 transform translate-y-0 scale-100'
            }`}>
              {voiceMessages[currentTextIndex]}
            </p>
          )}
        </div>
      </div>
    );
  };
  
  function Home() {
    const [messages, setMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const fileInputRef = useRef(null);
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const messagesEndRef = useRef(null);
    const [selectedModel, setSelectedModel] = useState('Claude 4 Sonnet');
    const [selectedMode, setSelectedMode] = useState('drag-drop');
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);
    const [isDropdownClosing, setIsDropdownClosing] = useState(false);
    const [voiceEnabled, setVoiceEnabled] = useState(true); // Enable voice by default
    
    // Voice panel state
    const [voicePanelOpen, setVoicePanelOpen] = useState(false);
    const [voicePanelClosing, setVoicePanelClosing] = useState(false);
    const [voicePanelOpening, setVoicePanelOpening] = useState(false);
    const [voiceButtonClicked, setVoiceButtonClicked] = useState(false);
    const [isMuted, setIsMuted] = useState(false);
    const [isFullscreen, setIsFullscreen] = useState(false);
    
    // Files management state
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [isFilesDropdownOpen, setIsFilesDropdownOpen] = useState(false);
    const [isFilesDropdownClosing, setIsFilesDropdownClosing] = useState(false);
    const filesInputRef = useRef(null);
  
    // Stop functionality state
    const [abortController, setAbortController] = useState(null);
    const [currentAudio, setCurrentAudio] = useState(null);
  
    // Exam settings state
    const [isExamSettingsOpen, setIsExamSettingsOpen] = useState(false);
    const [isExamSettingsClosing, setIsExamSettingsClosing] = useState(false);
    const [isExamFullscreen, setIsExamFullscreen] = useState(false);
  
    const navigate = useNavigate();
    const signOut = async () => {
      const {error} = await supabase.auth.signOut();
      if (error) throw error;
      navigate("/login");
    }

    const modes = [
      { value: 'drag-drop', label: 'Drag & Drop', emoji: '📋' },
      { value: 'mkat', label: 'MKAT Mode', emoji: '🩺' },
      { value: 'lsat', label: 'LSAT Mode', emoji: '⚖️' }
    ];
  
    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
      // Clear isNew flags after animation completes
      const timer = setTimeout(() => {
        setMessages(prev => prev.map(msg => ({ ...msg, isNew: false })));
      }, 500);
      
      return () => clearTimeout(timer);
    }, [messages]);
  
    // Debug dropdown states
    useEffect(() => {
      console.log('Dropdown states changed:', { isDropdownOpen, isFilesDropdownOpen });
    }, [isDropdownOpen, isFilesDropdownOpen]);
  
    const handleVoiceInput = (transcribedText) => {
      if (transcribedText.trim()) {
        setInputMessage(transcribedText);
        // Add immediate visual feedback that voice was processed
        const voiceProcessedMessage = {
          type: 'system',
          content: `🎤 Voice input: "${transcribedText}"`,
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, voiceProcessedMessage]);
        
        // Automatically send the message after voice input
        setTimeout(() => {
          handleSendMessage(transcribedText);
        }, 500);
      }
    };
  
    const handleVoicePanelClose = () => {
      setVoicePanelClosing(true);
      setTimeout(() => {
        setVoicePanelOpen(false);
        setVoicePanelClosing(false);
        setIsFullscreen(false); // Exit fullscreen when closing
      }, 250); // Match ultra-smooth pop-out animation duration
    };
  
    const handleVoicePanelOpen = () => {
      // Add click animation to button
      setVoiceButtonClicked(true);
      setTimeout(() => setVoiceButtonClicked(false), 80);
      
      // Start opening animation
      setVoicePanelOpening(true);
      setVoicePanelOpen(true);
      
      // Reset opening state after animation completes
      setTimeout(() => {
        setVoicePanelOpening(false);
      }, 400); // Match ultra-smooth pop-in animation duration
    };
  
    const handleStop = () => {
      // Stop ongoing API request
      if (abortController) {
        abortController.abort();
        setAbortController(null);
      }
  
      // Stop current audio playback
      if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
        setCurrentAudio(null);
      }
  
      // Reset loading state
      setIsLoading(false);
  
      // Remove any temporary messages (voice generation/playing indicators)
      setMessages(prev => prev.filter(msg => !msg.temporary));
  
      // Add stop message
      const stopMessage = {
        type: 'system',
        content: '⏹️ Generation stopped',
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, stopMessage]);
    };
  
    const handleSendMessage = async (messageText = null) => {
      const messageToSend = messageText || inputMessage;
      if (!messageToSend.trim()) return;
  
      // Add user message to chat (skip if it's from voice input since we already showed it)
      if (!messageText) {
        const userMessage = {
          type: 'user',
          content: messageToSend,
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, userMessage]);
      } else {
        // For voice input, convert the system message to user message
        setMessages(prev => prev.map((msg, index) => 
          index === prev.length - 1 && msg.type === 'system' && msg.content.includes('🎤 Voice input:')
            ? { ...msg, type: 'user', content: messageToSend }
            : msg
        ));
      }
      
      setInputMessage('');
      setIsLoading(true);
  
      // Create abort controller for this request
      const controller = new AbortController();
      setAbortController(controller);
  
      try {
        const response = await fetch(API_ENDPOINTS.CHAT, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message: messageToSend }),
          signal: controller.signal
        });
  
        const data = await response.json();
  
        if (response.ok) {
          // Add bot response to chat with typing indicator
          const botMessage = {
            type: 'bot',
            content: data.response,
            timestamp: new Date().toISOString(),
            isNew: true // Flag for animation
          };
          setMessages(prev => [...prev, botMessage]);
  
          // If voice is enabled, generate TTS for the response
          if (voiceEnabled) {
            // Add visual indicator that voice is being generated
            const voiceGeneratingMessage = {
              type: 'system',
              content: '🔊 Generating voice response...',
              timestamp: new Date().toISOString(),
              temporary: true
            };
            setMessages(prev => [...prev, voiceGeneratingMessage]);
  
            try {
              const ttsResponse = await fetch(API_ENDPOINTS.TTS, {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: data.response }),
                signal: controller.signal
              });
  
              if (ttsResponse.ok) {
                // Remove the "generating voice" message
                setMessages(prev => prev.filter(msg => !msg.temporary));
                
                const audioBlob = await ttsResponse.blob();
                const audioUrl = URL.createObjectURL(audioBlob);
                const audio = new Audio(audioUrl);
                
                // Track current audio
                setCurrentAudio(audio);
                
                // Add visual indicator that voice is playing
                const voicePlayingMessage = {
                  type: 'system',
                  content: '🎵 Playing voice response...',
                  timestamp: new Date().toISOString(),
                  temporary: true
                };
                setMessages(prev => [...prev, voicePlayingMessage]);
                
                audio.onended = () => {
                  // Remove the "playing voice" message when audio ends
                  setMessages(prev => prev.filter(msg => !msg.temporary));
                  setCurrentAudio(null);
                };
                
                audio.play().catch(e => {
                  console.log('Audio play failed:', e);
                  // Remove the "playing voice" message on error
                  setMessages(prev => prev.filter(msg => !msg.temporary));
                  setCurrentAudio(null);
                });
              } else {
                // Remove the "generating voice" message on error
                setMessages(prev => prev.filter(msg => !msg.temporary));
              }
            } catch (ttsError) {
              if (ttsError.name !== 'AbortError') {
                console.error('TTS Error:', ttsError);
                // Remove the "generating voice" message on error
                setMessages(prev => prev.filter(msg => !msg.temporary));
              }
            }
          }
        } else {
          throw new Error(data.error || 'Failed to get response');
        }
      } catch (error) {
        if (error.name !== 'AbortError') {
          console.error('Error:', error);
          // Add error message to chat
          const errorMessage = {
            type: 'error',
            content: 'Sorry, I encountered an error. Please try again.',
            timestamp: new Date().toISOString()
          };
          setMessages(prev => [...prev, errorMessage]);
        }
      } finally {
        setIsLoading(false);
        setAbortController(null);
      }
    };
  
    const handleFileUpload = async (event) => {
      const file = event.target.files[0];
      if (!file) return;
  
      // Add loading message
      const loadingMessage = {
        type: 'system',
        content: `📤 Uploading "${file.name}"...`,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, loadingMessage]);
  
      const formData = new FormData();
      formData.append('file', file);
  
      try {
        const response = await fetch(API_ENDPOINTS.UPLOAD, {
          method: 'POST',
          body: formData,
        });
  
        const data = await response.json();
  
        if (response.ok) {
          // Remove loading message and add success message
          setMessages(prev => prev.filter(msg => msg !== loadingMessage));
          const successMessage = {
            type: 'system',
            content: `✅ File "${file.name}" uploaded successfully! You can now ask questions about it.`,
            timestamp: new Date().toISOString()
          };
          setMessages(prev => [...prev, successMessage]);
        } else {
          throw new Error(data.error || 'Failed to upload file');
        }
      } catch (error) {
        console.error('Error:', error);
        // Remove loading message and add error message
        setMessages(prev => prev.filter(msg => msg !== loadingMessage));
        const errorMessage = {
          type: 'error',
          content: `❌ Failed to upload "${file.name}". Please try again.`,
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, errorMessage]);
      }
  
      // Reset the file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    };
  
    const handleKeyPress = (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (isLoading) {
          handleStop();
        } else {
          handleSendMessage();
        }
      }
    };
  
    // Auto-resize textarea function
    const handleTextareaChange = (e) => {
      setInputMessage(e.target.value);
      
      // Auto-resize textarea
      const textarea = e.target;
      textarea.style.height = 'auto';
      const scrollHeight = textarea.scrollHeight;
      const maxHeight = 200; // Max height in pixels
      
      if (scrollHeight <= maxHeight) {
        textarea.style.height = scrollHeight + 'px';
        textarea.style.overflowY = 'hidden';
      } else {
        textarea.style.height = maxHeight + 'px';
        textarea.style.overflowY = 'auto';
      }
    };
  
    const handleDropdownClose = () => {
      setIsDropdownClosing(true);
      // Wait for exit animation to complete before hiding
      setTimeout(() => {
        setIsDropdownOpen(false);
        setIsDropdownClosing(false);
      }, 200); // Match animation duration
    };
  
    const handleFilesDropdownClose = () => {
      setIsFilesDropdownClosing(true);
      // Wait for exit animation to complete before hiding
      setTimeout(() => {
        setIsFilesDropdownOpen(false);
        setIsFilesDropdownClosing(false);
      }, 200); // Match animation duration
    };
  
    const handleFilesUpload = async (event) => {
      const files = Array.from(event.target.files);
      if (files.length === 0) return;
  
      for (const file of files) {
        // Add file to uploaded files list
        const fileObj = {
          id: Date.now() + Math.random(),
          name: file.name,
          size: file.size,
          type: file.type,
          uploadedAt: new Date().toISOString()
        };
        
        setUploadedFiles(prev => [...prev, fileObj]);
  
        // Show upload message
        const uploadMessage = {
          type: 'system',
          content: `📎 Uploaded "${file.name}" (${(file.size / 1024).toFixed(1)} KB)`,
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, uploadMessage]);
      }
  
      // Reset the file input
      if (filesInputRef.current) {
        filesInputRef.current.value = '';
      }
    };
  
    const handleRemoveFile = (fileId) => {
      setUploadedFiles(prev => prev.filter(file => file.id !== fileId));
      
      const removeMessage = {
        type: 'system',
        content: `🗑️ File removed from uploads`,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, removeMessage]);
    };
  
    return (
      <div className="flex h-screen bg-gray-50 text-gray-900">
        {/* Sidebar Toggle Button - Hide when dropdowns are open */}
        {!(isDropdownOpen || isFilesDropdownOpen || isDropdownClosing || isFilesDropdownClosing || isExamSettingsOpen || isExamSettingsClosing || isExamFullscreen) && (
          <button
            className="fixed top-8 w-6 h-6 bg-white border border-gray-200 rounded-full shadow-lg hover:shadow-xl hover:bg-gray-50 transition-all duration-200 flex items-center justify-center z-[100]"
            style={{
              left: sidebarOpen ? '309px' : '56px', // Shifted 8px left from previous values
              transition: 'left 0.3s',
            }}
            onClick={() => {
              console.log('Toggle button clicked, dropdown states:', { isDropdownOpen, isFilesDropdownOpen });
              setSidebarOpen((open) => !open);
            }}
          >
            <svg width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" className="text-gray-500">
              <path d={sidebarOpen ? 'M15 19l-7-7 7-7' : 'M9 5l7 7-7 7'} />
            </svg>
          </button>
        )}
  
        {/* Frosty Blur Backdrop - Covers entire screen when dropdown is open */}
        {(isDropdownOpen || isDropdownClosing) && (
          <div 
            className="fixed inset-0 bg-white/20 z-[110] transition-all duration-200 ease-out"
            style={{
              backdropFilter: 'blur(0px)',
              animation: isDropdownClosing 
                ? 'backdropFadeOut 200ms ease-out forwards' 
                : 'backdropFadeIn 200ms ease-out forwards'
            }}
            onClick={handleDropdownClose} 
          />
        )}
  
        {/* Files Dropdown Backdrop */}
        {(isFilesDropdownOpen || isFilesDropdownClosing) && (
          <div 
            className="fixed inset-0 bg-white/20 z-[110] transition-all duration-200 ease-out"
            style={{
              backdropFilter: 'blur(0px)',
              animation: isFilesDropdownClosing 
                ? 'backdropFadeOut 200ms ease-out forwards' 
                : 'backdropFadeIn 200ms ease-out forwards'
            }}
            onClick={handleFilesDropdownClose} 
          />
        )}
  
        {/* Dropdown Menu - Rendered at root level to appear above backdrop */}
        {(isDropdownOpen || isDropdownClosing) && (
          <div 
            className="fixed top-1/2 left-1/2 w-[320px] bg-white border border-gray-200/40 rounded-3xl shadow-2xl z-[120] overflow-hidden p-2"
            style={{
              transform: 'translate(-50%, -50%) scale(0.95)',
              opacity: 0,
              animation: isDropdownClosing 
                ? 'dropdownExit 200ms ease-out forwards' 
                : 'dropdownEnter 200ms ease-out forwards'
            }}
          >
            {modes.map((mode, index) => (
              <button
                key={mode.value}
                onClick={() => {
                  setSelectedMode(mode.value);
                  handleDropdownClose();
                }}
                className={`w-full flex items-center gap-4 px-6 py-4 text-left hover:bg-gray-50/60 transition-all duration-200 text-sm rounded-2xl ${
                  selectedMode === mode.value 
                    ? 'bg-gray-100/80 text-gray-900' 
                    : 'text-gray-700 hover:text-gray-900'
                } ${index !== modes.length - 1 ? 'mb-1' : ''}`}
              >
                <div className="flex items-center justify-center w-10 h-10 bg-gray-100/60 rounded-xl">
                  <span className="text-xl">{mode.emoji}</span>
                </div>
                <div className="flex-1">
                  <span className="font-semibold text-base">{mode.label}</span>
                </div>
                {selectedMode === mode.value && (
                  <div className="flex items-center justify-center w-6 h-6 bg-black rounded-full">
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" className="text-white">
                      <path d="M20 6L9 17l-5-5" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                )}
              </button>
            ))}
          </div>
        )}
  
        {/* Files Dropdown Menu - Apple-inspired design */}
        {(isFilesDropdownOpen || isFilesDropdownClosing) && (
          <div 
            className="fixed top-1/2 left-1/2 w-[480px] bg-white border border-gray-200/40 rounded-3xl shadow-2xl z-[120] overflow-hidden"
            style={{
              transform: 'translate(-50%, -50%) scale(0.95)',
              opacity: 0,
              animation: isFilesDropdownClosing 
                ? 'dropdownExit 200ms ease-out forwards' 
                : 'dropdownEnter 200ms ease-out forwards'
            }}
          >
            {/* Header */}
            <div className="px-6 py-4 border-b border-gray-100/60 bg-gray-50/30">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="flex items-center justify-center w-8 h-8 bg-blue-100 rounded-lg">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-blue-600">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke="currentColor" strokeWidth="1.5"/>
                      <polyline points="14,2 14,8 20,8" stroke="currentColor" strokeWidth="1.5"/>
                    </svg>
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900 text-base">Files</h3>
                    <p className="text-xs text-gray-500">{uploadedFiles.length} file{uploadedFiles.length !== 1 ? 's' : ''} uploaded</p>
                  </div>
                </div>
                <button
                  onClick={() => filesInputRef.current?.click()}
                  className="flex items-center gap-2 px-4 py-2 bg-black text-white rounded-lg hover:bg-gray-800 transition-all duration-200 text-sm font-medium"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="text-white">
                    <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                  Upload
                </button>
              </div>
            </div>
  
            {/* Files List */}
            <div className="max-h-80 overflow-y-auto">
              {uploadedFiles.length === 0 ? (
                <div className="px-6 py-8 text-center">
                  <div className="flex items-center justify-center w-12 h-12 bg-gray-100 rounded-xl mx-auto mb-3">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-gray-400">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke="currentColor" strokeWidth="1.5"/>
                      <polyline points="14,2 14,8 20,8" stroke="currentColor" strokeWidth="1.5"/>
                    </svg>
                  </div>
                  <p className="text-gray-500 text-sm font-medium mb-1">No files uploaded</p>
                  <p className="text-gray-400 text-xs">Upload files to get started with document analysis</p>
                </div>
              ) : (
                <div className="p-3">
                  {uploadedFiles.map((file, index) => (
                    <div
                      key={file.id}
                      className={`flex items-center gap-4 px-4 py-3 rounded-2xl hover:bg-gray-50/60 transition-all duration-200 group ${
                        index !== uploadedFiles.length - 1 ? 'mb-2' : ''
                      }`}
                    >
                      <div className="flex items-center justify-center w-10 h-10 bg-blue-50 rounded-xl">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-blue-600">
                          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke="currentColor" strokeWidth="1.5"/>
                          <polyline points="14,2 14,8 20,8" stroke="currentColor" strokeWidth="1.5"/>
                        </svg>
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-gray-900 text-sm truncate">{file.name}</p>
                        <p className="text-xs text-gray-500">
                          {(file.size / 1024).toFixed(1)} KB • {new Date(file.uploadedAt).toLocaleDateString()}
                        </p>
                      </div>
                      <button
                        onClick={() => handleRemoveFile(file.id)}
                        className="opacity-0 group-hover:opacity-100 p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-all duration-200"
                        title="Remove file"
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="text-current">
                          <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                        </svg>
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
  
            {/* Hidden file input */}
            <input
              type="file"
              ref={filesInputRef}
              onChange={handleFilesUpload}
              className="hidden"
              accept=".pdf,.txt,.docx,.png,.jpg,.jpeg"
              multiple
            />
          </div>
        )}
  
        {/* Sidebar - Apple-level polish */}
        <aside className={`transition-all duration-300 ease-in-out ${sidebarOpen ? 'w-80' : 'w-16'} bg-white/80 backdrop-blur-xl flex flex-col h-full shadow-sm relative`}>
          {/* Logo/Header */}
          <div className={`flex items-center h-24 border-b border-gray-100/50 ${sidebarOpen ? 'justify-center' : 'justify-center'}`}>
            <div className="flex items-center gap-3">
              <img 
                src="/medbotlogonew.jpg" 
                alt="Company Logo" 
                className={`rounded-xl object-cover transition-all duration-300 ${sidebarOpen ? 'w-8 h-8' : 'w-10 h-10'}`}
              />
              {sidebarOpen && <span className="font-medium text-xl text-gray-900">Momentum AI</span>}
            </div>
          </div>
  
          {/* New Chat Button */}
          {sidebarOpen && (
            <div className="p-6">
              <button className="w-full flex items-center gap-3 px-4 py-3 bg-black text-white rounded-xl hover:bg-gray-800 transition-all duration-200 font-medium">
                <Plus className="w-4 h-4" />
                <span>New Chat</span>
              </button>
            </div>
          )}
  
          {/* Chat History */}
          <div className={`flex-1 overflow-y-auto py-2 ${sidebarOpen ? 'px-6' : 'px-2'}`}>
            {chatHistory.map((group) => (
              <div key={group.date} className="mb-8">
                {sidebarOpen && <div className="text-xs font-semibold text-gray-400 mb-4 px-2 uppercase tracking-wider">{group.date}</div>}
                <ul className="space-y-2">
                  {group.chats.map((chat) => (
                    <li key={chat.id} className={`flex items-center gap-3 px-3 py-3 rounded-xl cursor-pointer hover:bg-gray-50 transition-all duration-200 ${sidebarOpen ? '' : 'justify-center'}`}> 
                      {sidebarOpen && <span className="truncate text-gray-700 text-sm font-medium">{chat.title}</span>}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
  
          {/* Voice Panel and Settings Buttons */}
          <div className={`border-t border-gray-100/50 ${sidebarOpen ? 'p-6' : 'p-4'} space-y-2`}>
            {/* Voice Panel Toggle Button */}
            <button 
              onClick={handleVoicePanelOpen}
              className={`flex items-center gap-3 text-gray-500 hover:text-gray-700 transition-all duration-200 hover:bg-gray-50 rounded-xl p-3 ${sidebarOpen ? 'w-full' : 'justify-center'} ${
                !voicePanelOpen 
                  ? `voice-button-pulse ${voiceButtonClicked ? 'voice-button-click' : ''}` 
                  : 'opacity-50 cursor-not-allowed'
              }`}
              disabled={voicePanelOpen}
              title={voicePanelOpen ? 'Voice panel is open' : 'Open voice panel'}
            >
              {/* Custom 5-bar equalizer circular icon */}
              <span className={`transition-all duration-300 ${sidebarOpen ? 'w-4 h-4' : 'w-5 h-5'}`} style={{display:'inline-flex',alignItems:'center',justifyContent:'center'}}>
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="10" cy="10" r="9" stroke="currentColor" strokeWidth="1.5" fill="none"/>
                  <rect x="4.5" y="10" width="1" height="3" rx="0.5" fill="currentColor"/>
                  <rect x="7" y="7.5" width="1" height="7" rx="0.5" fill="currentColor"/>
                  <rect x="9.5" y="5.5" width="1" height="11" rx="0.5" fill="currentColor"/>
                  <rect x="12" y="7.5" width="1" height="7" rx="0.5" fill="currentColor"/>
                  <rect x="14.5" y="10" width="1" height="3" rx="0.5" fill="currentColor"/>
                </svg>
              </span>
              {sidebarOpen && <span className="font-medium">Voice Panel</span>}
            </button>
            
            {/* Settings Button */}
            <button className={`flex items-center gap-3 text-gray-500 hover:text-gray-700 transition-all duration-200 hover:bg-gray-50 rounded-xl p-3 ${sidebarOpen ? 'w-full' : 'justify-center'}`}>
              <Settings className={`transition-all duration-300 ${sidebarOpen ? 'w-4 h-4' : 'w-5 h-5'}`} />
              {sidebarOpen && <span className="font-medium">Settings</span>}
            </button>
            
            <button 
              className={`flex items-center gap-3 text-gray-500 hover:text-gray-700 transition-all duration-200 hover:bg-gray-50 rounded-xl p-3 ${sidebarOpen ? 'w-full' : 'justify-center'}`}
              onClick={signOut}>
              {sidebarOpen && <span className="font-medium">Sign Out</span>}
            </button>
          </div>
        </aside>
  
        {/* Voice Panel - Absolute Positioned Overlay */}
        {(voicePanelOpen || voicePanelClosing) && (
          <section 
            className={`voice-panel-container fixed top-0 h-full w-80 bg-white border-l border-gray-200/60 z-30 ${
              voicePanelClosing 
                ? 'voice-panel-pop-out' 
                : voicePanelOpening 
                  ? 'voice-panel-pop-in'
                  : 'transform scale-100 opacity-100'
            }`}
            style={{
              left: sidebarOpen ? '0px' : '0px' // More aggressive overlap for open sidebar
            }}
          >
            <div className="flex flex-col items-center justify-center h-full">
              <AnimatedAvatar 
                isLoading={isLoading} 
                onVoiceInput={handleVoiceInput} 
                isMuted={isMuted} 
                setIsMuted={setIsMuted} 
                isFullscreen={isFullscreen} 
                setIsFullscreen={setIsFullscreen} 
                onClose={handleVoicePanelClose} 
              />
            </div>
          </section>
        )}
  
        {/* Main Chat Area - Apple-level polish */}
        <div className="flex-1">
          <Chatbot
            messages={messages}
            inputMessage={inputMessage}
            isLoading={isLoading}
            fileInputRef={fileInputRef}
            filesInputRef={filesInputRef}
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
            selectedMode={selectedMode}
            setSelectedMode={setSelectedMode}
            isDropdownOpen={isDropdownOpen}
            setIsDropdownOpen={setIsDropdownOpen}
            voiceEnabled={voiceEnabled}
            setVoiceEnabled={setVoiceEnabled} 
            handleTextareaChange={handleTextareaChange}
            handleKeyPress={handleKeyPress}
            handleSendMessage={handleSendMessage}
            handleStop={handleStop}
            handleFileUpload={handleFileUpload}
            messagesEndRef={messagesEndRef}
            uploadedFiles={uploadedFiles}
            setUploadedFiles={setUploadedFiles}
            isFilesDropdownOpen={isFilesDropdownOpen}
            setIsFilesDropdownOpen={setIsFilesDropdownOpen}
            isExamSettingsOpen={isExamSettingsOpen}
            setIsExamSettingsOpen={setIsExamSettingsOpen}
            isExamSettingsClosing={isExamSettingsClosing}
            setIsExamSettingsClosing={setIsExamSettingsClosing}
            isExamFullscreen={isExamFullscreen}
            setIsExamFullscreen={setIsExamFullscreen}
          />
        </div>
      </div>
    );
  }
  
  export default Home;
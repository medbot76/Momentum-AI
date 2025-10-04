import { AnimatePresence, motion } from 'framer-motion';
import { Calendar, Clock, FileText, Paperclip, Upload } from 'lucide-react';
import React, { useEffect, useRef, useState, useMemo } from 'react';
import { API_ENDPOINTS } from '../config';
import Exam from './Exam';
import Flashcards from './Flashcards';
import StudyPlanner from './StudyPlanner';
import clsx from 'clsx';
import supabase from '../helper/supabaseClient';
//hi
// Unified Input Component - moved outside to prevent recreation on every render
const UnifiedInputComponent = ({ 
  placeholder, 
  showExamSettings = false,
  inputMessage,
  setInputMessage,
  handleSendMessage,
  isLoading,
  examGenerating,
  fileInputRef,
  handleFileUpload,
  selectedModel,
  setSelectedModel,
  tabs,
  activeTab,
  handleTabChange,
  isExamSettingsOpen,
  handleExamSettingsClose,
  setIsExamSettingsOpen,
  // Study Planner specific props
  studyPlannerStep,
  setStudyPlannerStep,
  studyPlannerFiles,
  setStudyPlannerFiles,
  studyPlannerDragOver,
  setStudyPlannerDragOver,
  handleGenerateStudyPlan,
  handleStudyPlannerFileUpload,
  // Streaming props
  streamingEnabled = true,
  setStreamingEnabled = () => {}
}) => {
  const textareaRef = useRef(null);
  
  // Add state for model selector dropdown
  const [isModelSelectorOpen, setIsModelSelectorOpen] = useState(false);
  const [isModelSelectorClosing, setIsModelSelectorClosing] = useState(false);

  // Model options with icons
  const modelOptions = [
    { value: 'Claude 4 Sonnet', label: 'Claude 4 Sonnet' },
    { value: 'GPT-4o', label: 'GPT-4o' },
    { value: 'Gemini 2.0 Flash', label: 'Gemini 2.0 Flash' }
  ];

  // Handle model selector dropdown close
  const handleModelSelectorClose = () => {
    setIsModelSelectorClosing(true);
    setTimeout(() => {
      setIsModelSelectorOpen(false);
      setIsModelSelectorClosing(false);
    }, 200);
  };

  // Study Planner handlers
  const handleStudyPlannerDragOver = (e) => {
    e.preventDefault();
    setStudyPlannerDragOver(true);
  };

  const handleStudyPlannerDragLeave = (e) => {
    e.preventDefault();
    setStudyPlannerDragOver(false);
  };

  const handleStudyPlannerDrop = (e) => {
    e.preventDefault();
    setStudyPlannerDragOver(false);
    const files = e.dataTransfer.files;
    handleStudyPlannerFileUpload(files);
  };

  const resetStudyPlanner = () => {
    // First animate to inputToOriginal state (matches original input box)
    setStudyPlannerStep('inputToOriginal');
    setStudyPlannerFiles([]);
    // Wait for the smoother animation to complete before switching tabs
    setTimeout(() => {
      handleTabChange('chatbot');
    }, 250); // Match the inputToOriginal animation duration (0.25s)
  };

  // Animation variants for Study Planner
  const containerVariants = {
    input: {
      width: '100%',
      maxWidth: '768px', // max-w-3xl = 48rem = 768px
      height: '140px', // Match the minHeight from original
      borderRadius: '24px', // rounded-3xl = 24px
      transition: {
        duration: 0.3, // Faster
        ease: [0.25, 0.1, 0.25, 1] // Apple easing
      }
    },
    inputToOriginal: {
      width: '100%',
      maxWidth: '768px', // Exact match to max-w-3xl
      height: '140px', // Exact match to original minHeight
      borderRadius: '24px', // Exact match to rounded-3xl
      transition: {
        duration: 0.25, // Faster for back transition
        ease: [0.25, 0.1, 0.25, 1] // Apple easing
      }
    },
    upload: {
      width: '100%',
      maxWidth: '500px',
      height: '450px',
      borderRadius: '20px',
      transition: {
        duration: 0.3, // Faster
        ease: [0.25, 0.1, 0.25, 1] // Apple easing
      }
    },
    loading: {
      width: '100%',
      maxWidth: '500px',
      height: '400px',
      borderRadius: '20px',
      transition: {
        duration: 0.25, // Faster
        ease: [0.25, 0.1, 0.25, 1] // Apple easing
      }
    },
    calendar: {
      width: '100%',
      maxWidth: '1200px',
      height: '700px',
      borderRadius: '20px',
      transition: {
        duration: 0.4, // Faster
        ease: [0.25, 0.1, 0.25, 1] // Apple easing
      }
    }
  };

  const contentVariants = {
    enter: {
      opacity: 0,
      y: 6, // Even smaller movement
      scale: 0.99, // More subtle scale
      transition: {
        duration: 0.2, // Faster
        ease: [0.25, 0.1, 0.25, 1] // Apple easing
      }
    },
    center: {
      opacity: 1,
      y: 0,
      scale: 1,
      transition: {
        duration: 0.25, // Faster
        delay: 0.1, // Shorter delay
        ease: [0.25, 0.1, 0.25, 1] // Apple easing
      }
    },
    exit: {
      opacity: 0,
      y: -6, // Even smaller movement
      scale: 1.01, // More subtle scale
      transition: {
        duration: 0.2, // Faster
        ease: [0.25, 0.1, 0.25, 1] // Apple easing
      }
    }
  };

  // Handle Study Planner tab click to start morphing
  const handleStudyPlannerTabClick = () => {
    handleTabChange('study-planner');
    // Always trigger morphing to upload when Study Planner tab is clicked
    setStudyPlannerStep('upload');
  };

  // If Study Planner is active, render the morphing component
  if (activeTab === 'study-planner') {
    return (
      <motion.div
        className="bg-white border-2 shadow-lg relative overflow-hidden transition-all duration-300 hover:shadow-xl group"
        variants={containerVariants}
        animate={studyPlannerStep}
        initial="input"
        style={{
          borderColor: studyPlannerStep === 'upload' && studyPlannerDragOver ? '#ef4444' : '#e5e7eb',
          boxShadow: studyPlannerStep === 'upload' && studyPlannerDragOver 
            ? '0 0 20px rgba(239, 68, 68, 0.15), 0 0 40px rgba(239, 68, 68, 0.08)' 
            : studyPlannerStep === 'upload'
            ? '0 10px 25px -5px rgba(0, 0, 0, 0.15)'
            : undefined
        }}
        onMouseEnter={() => {
          if (studyPlannerStep === 'upload') {
            // Add subtle red glow on hover
            document.querySelector('.study-planner-container')?.style.setProperty(
              'box-shadow', 
              '0 0 15px rgba(239, 68, 68, 0.12), 0 10px 25px -5px rgba(0, 0, 0, 0.15)'
            );
          }
        }}
        onMouseLeave={() => {
          if (studyPlannerStep === 'upload') {
            // Remove red glow
            document.querySelector('.study-planner-container')?.style.setProperty(
              'box-shadow', 
              '0 10px 25px -5px rgba(0, 0, 0, 0.15)'
            );
          }
        }}
      >
        <div className="study-planner-container">
          <AnimatePresence mode="wait">
            {studyPlannerStep === 'input' && (
              <motion.div
                key="input"
                variants={contentVariants}
                initial="enter"
                animate="center"
                exit="exit"
                className="h-full flex flex-col justify-center p-6"
              >
                <textarea
                  placeholder="Ask me anything about your studies..."
                  className="w-full h-full resize-none text-lg placeholder-gray-400 focus:outline-none bg-transparent"
                  onClick={() => setStudyPlannerStep('upload')}
                  readOnly
                />
                <div className="flex items-center justify-between mt-4">
                  <div className="flex items-center gap-4">
                    <button className="p-1.5 text-gray-400 hover:text-gray-600 transition-colors rounded-lg hover:bg-gray-50">
                      <FileText className="w-4 h-4" />
                    </button>
                    <div className="flex space-x-1 bg-gray-100 rounded-xl p-1">
                      {tabs.map((tab) => (
                        <button
                          key={tab.id}
                          onClick={() => tab.id === 'study-planner' ? handleStudyPlannerTabClick() : handleTabChange(tab.id)}
                          className={`flex items-center gap-2 px-2.5 py-1.5 rounded-lg font-medium text-sm transition-all duration-200 ${
                            activeTab === tab.id
                              ? 'bg-white text-gray-900 shadow-sm'
                              : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                          }`}
                        >
                          {tab.icon}
                          <span className="hidden sm:inline">{tab.label}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                  <button className="p-2.5 rounded-full bg-black hover:bg-gray-800 text-white transition-all duration-200 shadow-sm">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-white">
                      <path d="M7 11l5-5m0 0l5 5m-5-5v12" stroke="currentColor" strokeWidth="2"/>
                    </svg>
                  </button>
                </div>
              </motion.div>
            )}

            {studyPlannerStep === 'inputToOriginal' && (
              <motion.div
                key="inputToOriginal"
                variants={contentVariants}
                initial="enter"
                animate="center"
                exit="exit"
                className="h-full flex flex-col justify-center p-6"
              >
                <textarea
                  placeholder="Ask me anything..."
                  className="w-full h-full resize-none text-lg placeholder-gray-400 focus:outline-none bg-transparent"
                  readOnly
                />
                <div className="flex items-center justify-between mt-4">
                  <div className="flex items-center gap-4">
                    <button className="p-1.5 text-gray-400 hover:text-gray-600 transition-colors rounded-lg hover:bg-gray-50">
                      <FileText className="w-4 h-4" />
                    </button>
                    <div className="flex space-x-1 bg-gray-100 rounded-xl p-1">
                      {tabs.map((tab) => (
                        <button
                          key={tab.id}
                          onClick={() => tab.id === 'study-planner' ? handleStudyPlannerTabClick() : handleTabChange(tab.id)}
                          className={`flex items-center gap-2 px-2.5 py-1.5 rounded-lg font-medium text-sm transition-all duration-200 ${
                            activeTab === tab.id
                              ? 'bg-white text-gray-900 shadow-sm'
                              : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                          }`}
                        >
                          {tab.icon}
                          <span className="hidden sm:inline">{tab.label}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                  <button className="p-2.5 rounded-full bg-black hover:bg-gray-800 text-white transition-all duration-200 shadow-sm">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-white">
                      <path d="M7 11l5-5m0 0l5 5m-5-5v12" stroke="currentColor" strokeWidth="2"/>
                    </svg>
                  </button>
                </div>
              </motion.div>
            )}

            {studyPlannerStep === 'upload' && (
              <motion.div
                key="upload"
                variants={contentVariants}
                initial="enter"
                animate="center"
                exit="exit"
                className="flex flex-col min-h-0"
                onDragOver={handleStudyPlannerDragOver}
                onDragLeave={handleStudyPlannerDragLeave}
                onDrop={handleStudyPlannerDrop}
              >
                {/* Header Section */}
                <div className="text-center px-8 pt-6 pb-4 flex-shrink-0">
                  <div className={`flex items-center justify-center w-20 h-20 rounded-3xl mx-auto mb-6 transition-all duration-300 ${
                    studyPlannerDragOver 
                      ? 'bg-gray-100 scale-105' 
                      : 'bg-gray-50'
                  }`}>
                    <Upload className={`w-10 h-10 transition-colors duration-300 ${
                      studyPlannerDragOver ? 'text-gray-700' : 'text-gray-400'
                    }`} />
                  </div>
                  <h3 className="text-2xl font-bold text-gray-900 mb-3">Upload Your Syllabi</h3>
                  <p className="text-gray-500 text-lg">Drag and drop your semester syllabi here</p>
                </div>

                {/* Drop Zone */}
                <div className="px-8 flex-shrink-0">
                  <div 
                    className={`relative w-full h-32 border-2 border-dashed rounded-3xl flex items-center justify-center transition-all duration-300 cursor-pointer group mb-6 ${
                      studyPlannerDragOver 
                        ? 'border-gray-500 bg-gray-50/90 scale-[1.02]' 
                        : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50/60'
                    }`}
                    onClick={() => {
                      // Add file input click functionality
                      const input = document.createElement('input');
                      input.type = 'file';
                      input.multiple = true;
                      input.accept = '.pdf,.txt,.docx,.png,.jpg,.jpeg';
                      input.onchange = (e) => {
                        if (e.target.files) {
                          handleStudyPlannerFileUpload(e.target.files);
                        }
                      };
                      input.click();
                    }}
                  >
                    <div className="text-center">
                      <div className={`flex items-center justify-center w-12 h-12 rounded-2xl mx-auto mb-4 transition-all duration-300 ${
                        studyPlannerDragOver 
                          ? 'bg-gray-200 scale-110' 
                          : 'bg-gray-100 group-hover:bg-gray-200 group-hover:scale-105'
                      }`}>
                        <Upload className={`w-6 h-6 transition-all duration-300 ${
                          studyPlannerDragOver 
                            ? 'text-gray-700' 
                            : 'text-gray-500 group-hover:text-gray-700'
                        }`} />
                      </div>
                      <p className={`text-base font-semibold transition-colors duration-300 ${
                        studyPlannerDragOver 
                          ? 'text-gray-800' 
                          : 'text-gray-600 group-hover:text-gray-800'
                      }`}>
                        {studyPlannerDragOver ? 'Drop files here' : 'Drop files or click to browse'}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Scrollable Files Area */}
                <div className="px-8 flex-1 overflow-y-auto min-h-0">
                  {studyPlannerFiles.length > 0 && (
                    <div className="flex flex-col">
                      <div className="flex items-center justify-between mb-4 flex-shrink-0">
                        <h4 className="text-lg font-semibold text-gray-900">Uploaded Files</h4>
                        <span className="text-sm text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
                          {studyPlannerFiles.length} file{studyPlannerFiles.length !== 1 ? 's' : ''}
                        </span>
                      </div>
                      <div className="space-y-3 pb-4">
                        {studyPlannerFiles.map((file, index) => (
                          <motion.div 
                            key={file.id}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.05, duration: 0.2 }}
                            className="flex items-center gap-4 p-4 bg-gray-50/80 rounded-2xl border border-gray-100 hover:bg-gray-100/80 transition-all duration-200 group flex-shrink-0"
                          >
                            <div className="flex items-center justify-center w-10 h-10 bg-blue-50 rounded-xl flex-shrink-0">
                              <FileText className="w-5 h-5 text-blue-600" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-semibold text-gray-900 truncate">{file.name}</p>
                              <p className="text-xs text-gray-500">{(file.size / 1024).toFixed(1)} KB</p>
                            </div>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setStudyPlannerFiles(prev => prev.filter(f => f.id !== file.id));
                              }}
                              className="opacity-0 group-hover:opacity-100 p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-xl transition-all duration-200 flex-shrink-0"
                            >
                              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-current">
                                <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                              </svg>
                            </button>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Action Buttons - Integrated into the card */}
                <div className="flex items-center justify-between px-8 py-4 flex-shrink-0">
                  <button
                    onClick={resetStudyPlanner}
                    className="px-6 py-3 text-gray-500 hover:text-gray-700 font-medium rounded-2xl hover:bg-gray-50 transition-all duration-200"
                  >
                    Back
                  </button>
                  <button
                    onClick={handleGenerateStudyPlan}
                    disabled={studyPlannerFiles.length === 0}
                    className={`px-8 py-3 font-semibold rounded-2xl transition-all duration-300 transform ${
                      studyPlannerFiles.length > 0
                        ? 'bg-black text-white hover:bg-gray-800 hover:scale-105 shadow-lg hover:shadow-xl'
                        : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                    }`}
                  >
                    Generate Study Plan
                  </button>
                </div>
              </motion.div>
            )}

            {studyPlannerStep === 'loading' && (
              <motion.div
                key="loading"
                variants={contentVariants}
                initial="enter"
                animate="center"
                exit="exit"
                className="h-full flex flex-col items-center justify-center p-10"
              >
                <div className="text-center">
                  {/* Loading Icon */}
                  <div className="flex items-center justify-center w-24 h-24 bg-gradient-to-br from-gray-50 to-gray-100 rounded-3xl mx-auto mb-8 shadow-sm">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                    >
                      <Clock className="w-12 h-12 text-gray-600" />
                    </motion.div>
                  </div>
                  
                  {/* Loading Text */}
                  <h3 className="text-2xl font-semibold text-gray-900 mb-4">Generating Your Study Plan</h3>
                  <p className="text-gray-500 mb-10 max-w-md leading-relaxed">
                    Analyzing your syllabi and creating a personalized schedule optimized for your success...
                  </p>
                  
                  {/* Progress Dots */}
                  <div className="flex items-center justify-center gap-3">
                    {[0, 1, 2].map((i) => (
                      <motion.div
                        key={i}
                        className="w-3 h-3 bg-gray-400 rounded-full"
                        animate={{ 
                          scale: [1, 1.4, 1],
                          opacity: [0.5, 1, 0.5]
                        }}
                        transition={{
                          duration: 1.2,
                          repeat: Infinity,
                          delay: i * 0.2,
                          ease: [0.25, 0.1, 0.25, 1]
                        }}
                      />
                    ))}
                  </div>
                  
                  {/* Progress Steps */}
                  <div className="mt-12 space-y-4 max-w-sm">
                    {[
                      { text: "Reading syllabi content", delay: 0 },
                      { text: "Identifying key dates", delay: 1000 },
                      { text: "Creating optimal schedule", delay: 2000 }
                    ].map((step, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: step.delay / 1000, duration: 0.5 }}
                        className="flex items-center gap-3 text-sm text-gray-600"
                      >
                        <div className="w-2 h-2 bg-gray-300 rounded-full"></div>
                        <span>{step.text}</span>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    );
  }

  // Regular input component for other tabs
  return (
    <div className="w-full max-w-3xl mb-6">
      {/* Main Input Area */}
      <div className="relative bg-white border border-gray-200 rounded-3xl shadow-lg focus-within:ring-2 focus-within:ring-black focus-within:border-transparent">
        <textarea
          ref={textareaRef}
          placeholder={placeholder}
          className="w-full px-6 pt-5 pb-19 text-lg bg-transparent resize-none placeholder-gray-400 focus:outline-none"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSendMessage();
            }
          }}
          disabled={isLoading || examGenerating}
          rows={1}
          style={{ minHeight: '140px' }}
        />
        {/* Bottom controls inside the input box */}
        <div className="absolute bottom-0 left-0 right-0 bg-white rounded-b-3xl border-t border-gray-100 px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="p-1.5 text-gray-400 hover:text-gray-600 transition-colors rounded-lg hover:bg-gray-50"
              title="Attach file"
            >
              <Paperclip className="w-4 h-4" />
            </button>
            {/* Exam Settings Button - only show when specified */}
            {showExamSettings && (
              <button
                onClick={() => {
                  if (isExamSettingsOpen) {
                    handleExamSettingsClose();
                  } else {
                    setIsExamSettingsOpen(true);
                  }
                }}
                className="p-2 text-gray-400 hover:text-gray-600 transition-colors rounded-lg hover:bg-gray-50"
                title="Exam settings"
              >
                <span className="text-base">⚙️</span>
              </button>
            )}
            <button
              onClick={() => {
                if (isModelSelectorOpen) {
                  handleModelSelectorClose();
                } else {
                  setIsModelSelectorOpen(true);
                }
              }}
              className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-gray-700 bg-white border border-gray-200 rounded-full hover:bg-gray-50 hover:border-gray-300 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-black/10 shadow-sm"
            >
              <span className="font-medium">{selectedModel}</span>
              <svg width="8" height="8" viewBox="0 0 24 24" fill="none" className={`text-gray-500 transition-transform duration-200 ${isModelSelectorOpen ? 'rotate-180' : ''}`}>
                <path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="2"/>
              </svg>
            </button>
            {/* Tab buttons inline with other controls */}
            <div className="flex space-x-1 bg-gray-100 rounded-xl p-1">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => tab.id === 'study-planner' ? handleStudyPlannerTabClick() : handleTabChange(tab.id)}
                  className={`flex items-center gap-2 px-2.5 py-1.5 rounded-lg font-medium text-sm transition-all duration-200 ${
                    activeTab === tab.id
                      ? 'bg-white text-gray-900 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`}
                >
                  {tab.icon}
                  <span className="hidden sm:inline">{tab.label}</span>
                </button>
              ))}
            </div>
          </div>
          
          {/* Streaming Toggle Button - Only show in chatbot tab */}
          {activeTab === 'chatbot' && (
            <button
              onClick={() => setStreamingEnabled(!streamingEnabled)}
              className={`p-2.5 rounded-full transition-all duration-200 shadow-sm ml-2 ${
                streamingEnabled
                  ? 'bg-blue-500 hover:bg-blue-600 text-white' 
                  : 'bg-gray-200 hover:bg-gray-300 text-gray-600'
              }`}
              title={streamingEnabled ? 'Disable real-time streaming' : 'Enable real-time streaming'}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-current">
                <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              </svg>
            </button>
          )}
          
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() && !examGenerating}
            className={`p-2.5 rounded-full transition-all duration-200 shadow-sm ml-4 ${
              (isLoading || examGenerating)
                ? 'bg-red-500 hover:bg-red-600 text-white' 
                : 'bg-black hover:bg-gray-800 text-white disabled:opacity-50 disabled:cursor-not-allowed'
            }`}
          >
            {(isLoading || examGenerating) ? (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-white">
                <rect x="6" y="6" width="12" height="12" rx="2" stroke="currentColor" strokeWidth="2" fill="currentColor"/>
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-white">
                <path d="M7 11l5-5m0 0l5 5m-5-5v12" stroke="currentColor" strokeWidth="2"/>
              </svg>
            )}
          </button>
        </div>
        {/* File upload input */}
        <input
          type="file"
          ref={fileInputRef}
          onChange={(e) => {
            if (e.target.files) {
              const file = e.target.files[0];
              handleFileUpload(file);
            }
          }}
          className="hidden"
          accept=".pdf,.txt,.docx,.png,.jpg,.jpeg"
          multiple={false}
        />

        {/* Model Selector Dropdown Menu */}
        {(isModelSelectorOpen || isModelSelectorClosing) && (
          <div 
            className="fixed top-1/2 left-1/2 w-[240px] bg-white border border-gray-200/40 rounded-2xl shadow-2xl z-[120] overflow-hidden p-2"
            style={{
              transform: 'translate(-50%, -50%) scale(0.95)',
              opacity: 0,
              animation: isModelSelectorClosing 
                ? 'dropdownExit 200ms ease-out forwards' 
                : 'dropdownEnter 200ms ease-out forwards'
            }}
          >
            {modelOptions.map((model, index) => (
              <button
                key={model.value}
                onClick={() => {
                  setSelectedModel(model.value);
                  handleModelSelectorClose();
                }}
                className={`w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-gray-50/60 transition-all duration-200 text-sm rounded-xl ${
                  selectedModel === model.value 
                    ? 'bg-gray-100/80 text-gray-900' 
                    : 'text-gray-700 hover:text-gray-900'
                } ${index !== modelOptions.length - 1 ? 'mb-1' : ''}`}
              >
                <div className="flex-1">
                  <span className="font-medium text-sm">{model.label}</span>
                </div>
                {selectedModel === model.value && (
                  <div className="flex items-center justify-center w-5 h-5 bg-black rounded-full">
                    <svg width="8" height="8" viewBox="0 0 24 24" fill="none" className="text-white">
                      <path d="M20 6L9 17l-5-5" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                )}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

function Chatbot({ 
  messages: propMessages, 
  inputMessage: propInputMessage, 
  isLoading: propIsLoading, 
  fileInputRef,
  filesInputRef,
  selectedModel,
  setSelectedModel,
  selectedMode,
  setSelectedMode,
  isDropdownOpen,
  setIsDropdownOpen,
  voiceEnabled,
  setVoiceEnabled,
  handleTextareaChange,
  handleKeyPress,
  handleSendMessage: propHandleSendMessage, 
  handleStop: propHandleStop, 
  handleFileUpload: propHandleFileUpload, 
  messagesEndRef,
  uploadedFiles: propUploadedFiles, 
  setUploadedFiles,
  isFilesDropdownOpen,
  setIsFilesDropdownOpen,
  isExamSettingsOpen: propIsExamSettingsOpen,
  setIsExamSettingsOpen: propSetIsExamSettingsOpen,
  isExamSettingsClosing: propIsExamSettingsClosing,
  setIsExamSettingsClosing: propSetIsExamSettingsClosing,
  isExamFullscreen: propIsExamFullscreen,
  setIsExamFullscreen: propSetIsExamFullscreen
}) {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isDropdownClosing, setIsDropdownClosing] = useState(false);
  const [isFilesDropdownClosing, setIsFilesDropdownClosing] = useState(false);
  const [isModelSelectorOpen, setIsModelSelectorOpen] = useState(false);
  const [isModelSelectorClosing, setIsModelSelectorClosing] = useState(false);
  const [abortController, setAbortController] = useState(null);
  const [currentAudio, setCurrentAudio] = useState(null);
  const [activeTab, setActiveTab] = useState('chatbot');
  const [flashcards, setFlashcards] = useState([]);
  const [examPrompt, setExamPrompt] = useState('');
  const [examConfig, setExamConfig] = useState({
    difficulty: 'medium',
    numQuestions: 10,
    topic: '',
    types: 'Mixed',
    format: 'PDF',
  });
  const [examGenerating, setExamGenerating] = useState(false);
  const [stopExamGeneration, setStopExamGeneration] = useState(null);
  
  // Use prop directly instead of local state
  const uploadedFiles = propUploadedFiles || [];
  
  // State for files fetched from Supabase
  const [supabaseFiles, setSupabaseFiles] = useState([]);
  const [loadingFiles, setLoadingFiles] = useState(false);
  
  // State for notebook management
  const [currentNotebookId, setCurrentNotebookId] = useState(null);
  const [currentNotebookName, setCurrentNotebookName] = useState('Default Notebook');
  const [isCreatingNewChat, setIsCreatingNewChat] = useState(false);
  const [newChatNotebookName, setNewChatNotebookName] = useState('');
  const [availableNotebooks, setAvailableNotebooks] = useState([]);
  const [isNewChatDropdownOpen, setIsNewChatDropdownOpen] = useState(false);
  
  // Study Planner states
  const [studyPlannerStep, setStudyPlannerStep] = useState('input'); // 'input', 'upload', 'loading', 'calendar'
  const [studyPlannerFiles, setStudyPlannerFiles] = useState([]);
  const [studyPlannerDragOver, setStudyPlannerDragOver] = useState(false);
  
  // Use props for exam settings states if provided, otherwise use local state
  const isExamSettingsOpen = propIsExamSettingsOpen !== undefined ? propIsExamSettingsOpen : false;
  const setIsExamSettingsOpen = propSetIsExamSettingsOpen || (() => {});
  const isExamSettingsClosing = propIsExamSettingsClosing !== undefined ? propIsExamSettingsClosing : false;
  const setIsExamSettingsClosing = propSetIsExamSettingsClosing || (() => {});
  
  // Use props for exam fullscreen states if provided, otherwise use local state
  const isExamFullscreen = propIsExamFullscreen !== undefined ? propIsExamFullscreen : false;
  const setIsExamFullscreen = propSetIsExamFullscreen || (() => {});

  // Example Exam upload state
  const [exampleExamFile, setExampleExamFile] = useState(null);
  const [exampleExamFilename, setExampleExamFilename] = useState("");
  const exampleExamInputRef = useRef(null);
  const [uploadingExampleExam, setUploadingExampleExam] = useState(false);

  // Add state for study plan results
  const [studyPlanPdfUrl, setStudyPlanPdfUrl] = useState("");
  const [studyPlanEvents, setStudyPlanEvents] = useState([]);
  const [studyPlanError, setStudyPlanError] = useState("");
  const [calendarEmail, setCalendarEmail] = useState("");
  const [calendarType, setCalendarType] = useState("gmail");
  const [isAddingToCalendar, setIsAddingToCalendar] = useState(false);
  const [calendarResult, setCalendarResult] = useState(null);
  const studyPlannerWeeksRef = useRef(16); // For future: allow user to set weeks
  
  // Streaming state
  const [streamingUpdates, setStreamingUpdates] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingEnabled, setStreamingEnabled] = useState(true); // Toggle for streaming
  
  // Inline Study Plan Calendar component (simple month grid)
  const StudyPlanCalendar = ({ events }) => {
    const [currentDate, setCurrentDate] = useState(() => new Date());

    const year = currentDate.getFullYear();
    const month = currentDate.getMonth();

    const startOfMonth = new Date(year, month, 1);
    const endOfMonth = new Date(year, month + 1, 0);
    const startDay = startOfMonth.getDay(); // 0 (Sun) - 6 (Sat)
    const totalDays = endOfMonth.getDate();

    const daysArray = useMemo(() => {
      const cells = [];
      // Add leading blanks
      for (let i = 0; i < startDay; i += 1) cells.push(null);
      // Add days of month
      for (let d = 1; d <= totalDays; d += 1) cells.push(new Date(year, month, d));
      return cells;
    }, [startDay, totalDays, year, month]);

    const eventsByDay = useMemo(() => {
      const map = new Map();
      (events || []).forEach((evt) => {
        const dtStr = evt.start_datetime || evt.start || evt.date;
        if (!dtStr) return;
        const dt = new Date(dtStr);
        if (Number.isNaN(dt.getTime())) return;
        if (dt.getMonth() !== month || dt.getFullYear() !== year) return;
        const key = dt.getDate();
        const list = map.get(key) || [];
        list.push(evt);
        map.set(key, list);
      });
      return map;
    }, [events, month, year]);

    const goPrevMonth = () => {
      const prev = new Date(year, month - 1, 1);
      setCurrentDate(prev);
    };
    const goNextMonth = () => {
      const next = new Date(year, month + 1, 1);
      setCurrentDate(next);
    };

    const monthName = currentDate.toLocaleString(undefined, { month: 'long', year: 'numeric' });
    const weekDays = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

    return (
      <div className="mt-8">
        <div className="flex items-center justify-between mb-3">
          <button onClick={goPrevMonth} className="px-3 py-1.5 rounded-lg border text-sm hover:bg-gray-50">Prev</button>
          <div className="text-lg font-semibold">{monthName}</div>
          <button onClick={goNextMonth} className="px-3 py-1.5 rounded-lg border text-sm hover:bg-gray-50">Next</button>
        </div>
        <div className="grid grid-cols-7 gap-2 text-xs font-medium text-gray-500 mb-2">
          {weekDays.map((d) => (
            <div key={d} className="text-center">{d}</div>
          ))}
        </div>
        <div className="grid grid-cols-7 gap-2">
          {daysArray.map((cellDate, idx) => {
            if (!cellDate) return <div key={`blank-${idx}`} className="h-28 rounded-xl border border-dashed border-gray-200 bg-gray-50" />;
            const day = cellDate.getDate();
            const todaysEvents = eventsByDay.get(day) || [];
            return (
              <div key={`day-${day}-${idx}`} className="h-28 rounded-xl border border-gray-200 p-2 flex flex-col overflow-hidden">
                <div className="text-xs font-semibold text-gray-700 mb-1">{day}</div>
                <div className="flex-1 overflow-y-auto space-y-1 pr-1">
                  {todaysEvents.slice(0, 3).map((evt, i) => (
                    <div key={i} className="text-[10px] bg-blue-50 text-blue-700 border border-blue-100 rounded px-1 py-0.5 truncate">
                      {evt.summary || evt.title || 'Event'}
                    </div>
                  ))}
                  {todaysEvents.length > 3 && (
                    <div className="text-[10px] text-gray-500">+{todaysEvents.length - 3} more</div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const handleDownloadStudyPlanPDF = async () => {
    if (studyPlanPdfUrl) {
      try {
        console.log('Starting PDF download...');
        console.log('Original URL:', studyPlanPdfUrl);
        
        // Construct full URL if it's a relative path
        const fullUrl = studyPlanPdfUrl.startsWith('http') 
          ? studyPlanPdfUrl 
          : `${window.location.origin}${studyPlanPdfUrl}`;
        
        console.log('Full URL:', fullUrl);
        
        // Fetch the PDF from the backend
        const response = await fetch(fullUrl);
        console.log('Response status:', response.status);
        console.log('Response headers:', Object.fromEntries(response.headers.entries()));
        
        if (response.ok) {
          const blob = await response.blob();
          console.log('Blob size:', blob.size);
          console.log('Blob type:', blob.type);
          
          // Check if the blob has content
          if (blob.size === 0) {
            console.error('Downloaded blob is empty');
            alert('Error: The PDF file is empty. Please try generating the study plan again.');
            return;
          }
          
          // Check if the blob is actually a PDF
          if (blob.type !== 'application/pdf') {
            console.warn('Downloaded file is not a PDF:', blob.type);
            console.log('Response content-type:', response.headers.get('content-type'));
          }
          
          // Try to get the filename from the response headers
          const contentDisposition = response.headers.get('content-disposition');
          let filename = `study_plan_${Date.now()}.pdf`;
          
          if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
            if (filenameMatch && filenameMatch[1]) {
              filename = filenameMatch[1].replace(/['"]/g, '');
            }
          }
          
          console.log('Downloading as:', filename);
          
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = filename;
          a.style.display = 'none';
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
          
          console.log('PDF download completed');
        } else {
          console.error('Failed to fetch PDF:', response.status, response.statusText);
          // Try to get error details
          const errorText = await response.text();
          console.error('Error response:', errorText);
          alert(`Error downloading PDF: ${response.status} ${response.statusText}`);
        }
      } catch (error) {
        console.error('Error downloading PDF:', error);
        alert(`Error downloading PDF: ${error.message}`);
      }
    } else {
      console.error('No PDF URL available');
      alert('No PDF URL available. Please generate a study plan first.');
    }
  };

  const tabs = [
    { id: 'chatbot', label: 'Chatbot', icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-current">
        <circle cx="11" cy="11" r="8" stroke="currentColor" strokeWidth="1.5"/>
        <path d="m21 21-4.35-4.35" stroke="currentColor" strokeWidth="1.5"/>
      </svg>
    )},
    { id: 'flashcards', label: 'Flashcards', icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-current">
        <rect x="3" y="5" width="18" height="12" rx="3" stroke="currentColor" strokeWidth="1.5" fill="none"/>
        <path d="M7 9h10M7 13h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    )},
    { id: 'exam', label: 'Exam', icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-current">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke="currentColor" strokeWidth="1.5" fill="none"/>
        <polyline points="14,2 14,8 20,8" stroke="currentColor" strokeWidth="1.5"/>
        <line x1="16" y1="13" x2="8" y2="13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        <line x1="16" y1="17" x2="8" y2="17" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    )},
    { id: 'study-planner', label: 'Study Planner', icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-current">
        <rect x="3" y="4" width="18" height="18" rx="2" ry="2" stroke="currentColor" strokeWidth="1.5" fill="none"/>
        <line x1="16" y1="2" x2="16" y2="6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        <line x1="8" y1="2" x2="8" y2="6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        <line x1="3" y1="10" x2="21" y2="10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    )}
  ];

  const modes = [
    { value: 'drag-drop', label: 'Drag & Drop', emoji: '📋' },
    { value: 'mkat', label: 'MKAT Mode', emoji: '🩺' },
    { value: 'lsat', label: 'LSAT Mode', emoji: '⚖️' }
  ];

  const modelOptions = [
    { value: 'Claude 4 Sonnet', label: 'Claude 4 Sonnet' },
    { value: 'GPT-4o', label: 'GPT-4o' },
    { value: 'Gemini 2.0 Flash', label: 'Gemini 2.0 Flash' }
  ];

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    // Clear isNew flags after animation completes
    const timer = setTimeout(() => {
      setMessages(prev => prev.map(msg => ({ ...msg, isNew: false })));
    }, 500);
    
    return () => clearTimeout(timer);
  }, [messages]);

  // Get or create notebook for the current chat
  const getOrCreateNotebook = async (notebookName = 'Default Notebook') => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      
      if (!user?.id) {
        return null;
      }

      // Try to find existing notebook with this name
      const { data: notebooks } = await supabase
        .from('notebooks')
        .select('id, name')
        .eq('user_id', user.id)
        .eq('name', notebookName)
        .limit(1);
      
      if (notebooks && notebooks.length > 0) {
        return notebooks[0];
      }

      // Create new notebook if it doesn't exist
      const { data: newNotebook, error } = await supabase
        .from('notebooks')
        .insert([{
          user_id: user.id,
          name: notebookName,
          description: `Notebook: ${notebookName}`,
          color: '#4285f4'
        }])
        .select('id, name')
        .single();

      if (error) {
        console.error('Error creating notebook:', error);
        return null;
      }

      return newNotebook;
    } catch (error) {
      console.error('Error getting/creating notebook:', error);
      return null;
    }
  };

  // Fetch user's files from Supabase
  const fetchUserFiles = async () => {
    try {
      setLoadingFiles(true);
      const { data: { user } } = await supabase.auth.getUser();
      
      if (!user?.id) {
        setSupabaseFiles([]);
        return;
      }

      // Use current notebook or get default
      let notebookId = currentNotebookId;
      if (!notebookId) {
        const notebook = await getOrCreateNotebook(currentNotebookName);
        if (notebook) {
          notebookId = notebook.id;
          setCurrentNotebookId(notebookId);
        }
      }

      if (!notebookId) {
        setSupabaseFiles([]);
        return;
      }

      // Fetch documents for this user and notebook
      const { data: documents, error } = await supabase
        .from('documents')
        .select('id, filename, original_filename, file_type, file_size, created_at, processing_status')
        .eq('user_id', user.id)
        .eq('notebook_id', notebookId)
        .order('created_at', { ascending: false });

      if (error) {
        console.error('Error fetching documents:', error);
        setSupabaseFiles([]);
        return;
      }

      // Transform documents to match the expected format
      const transformedFiles = documents.map(doc => ({
        id: doc.id,
        name: doc.original_filename || doc.filename,
        filename: doc.original_filename || doc.filename, // Add filename property for flashcards
        size: doc.file_size || 0,
        type: doc.file_type || 'unknown',
        supabaseDocument: true,
        status: doc.processing_status,
        uploadedAt: doc.created_at
      }));

      console.log('Fetched files from Supabase:', transformedFiles);
      setSupabaseFiles(transformedFiles);
    } catch (error) {
      console.error('Error fetching user files:', error);
      setSupabaseFiles([]);
    } finally {
      setLoadingFiles(false);
    }
  };

  // Fetch files when component mounts
  useEffect(() => {
    fetchUserFiles();
  }, []);

  // Load available notebooks
  const loadAvailableNotebooks = async () => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user?.id) return;

      const { data: notebooks } = await supabase
        .from('notebooks')
        .select('id, name, created_at')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

      setAvailableNotebooks(notebooks || []);
    } catch (error) {
      console.error('Error loading notebooks:', error);
    }
  };

  // Initialize default notebook on mount
  useEffect(() => {
    const initializeDefaultNotebook = async () => {
      const notebook = await getOrCreateNotebook('Default Notebook');
      if (notebook) {
        setCurrentNotebookId(notebook.id);
        setCurrentNotebookName(notebook.name);
      }
      // Load all available notebooks
      await loadAvailableNotebooks();
    };
    initializeDefaultNotebook();
  }, []);

  // Handle new chat creation
  const handleNewChat = () => {
    if (availableNotebooks.length > 0) {
      setIsNewChatDropdownOpen(true);
    } else {
      setIsCreatingNewChat(true);
      setNewChatNotebookName('');
    }
  };

  // Create new notebook and start new chat
  const createNewChat = async () => {
    if (!newChatNotebookName.trim()) {
      alert('Please enter a notebook name');
      return;
    }

    try {
      const notebook = await getOrCreateNotebook(newChatNotebookName.trim());
      if (notebook) {
        // Reset chat state
        setMessages([]);
        setInputMessage('');
        setCurrentNotebookId(notebook.id);
        setCurrentNotebookName(notebook.name);
        setSupabaseFiles([]);
        setUploadedFiles([]);
        
        // Close new chat modal
        setIsCreatingNewChat(false);
        setNewChatNotebookName('');
        
        // Fetch files for new notebook
        await fetchUserFiles();
        
        // Refresh available notebooks list
        await loadAvailableNotebooks();
      }
    } catch (error) {
      console.error('Error creating new chat:', error);
      alert('Failed to create new chat');
    }
  };

  // Cancel new chat creation
  const cancelNewChat = () => {
    setIsCreatingNewChat(false);
    setNewChatNotebookName('');
  };

  // Switch to a different notebook
  const switchToNotebook = async (notebook) => {
    try {
      // Reset chat state
      setMessages([]);
      setInputMessage('');
      setCurrentNotebookId(notebook.id);
      setCurrentNotebookName(notebook.name);
      setSupabaseFiles([]);
      setUploadedFiles([]);
      
      // Fetch files for the selected notebook
      await fetchUserFiles();
    } catch (error) {
      console.error('Error switching to notebook:', error);
      alert('Failed to switch notebook');
    }
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

    // Handle exam tab differently
    if (activeTab === 'exam') {
      // If exam is generating, stop it
      if (examGenerating && stopExamGeneration) {
        stopExamGeneration();
        return;
      }
      
      // For exam, pass the prompt and config to the Exam component
      setInputMessage('');
      const promptWithConfig = {
        prompt: messageToSend,
        config: examConfig
      };
      setExamPrompt(promptWithConfig);
      return;
    }

    // Handle flashcards tab differently
    if (activeTab === 'flashcards') {
      setInputMessage('');
      const placeholderFlashcard = {
        id: Date.now(),
        question: 'Generating flashcards...',
        answer: 'Please wait while we create your flashcards.',
        topic: messageToSend,
        isLoading: true
      };
      setFlashcards(prev => [...prev, placeholderFlashcard]);
      setIsLoading(true);

      // Use the most recently uploaded file from Supabase
      console.log('Available Supabase files:', supabaseFiles);
      const latestFile = supabaseFiles.length > 0 ? supabaseFiles[supabaseFiles.length - 1].filename : null;
      console.log('Latest file for flashcards:', latestFile);
      
      if (!latestFile) {
        setFlashcards(prev => prev.map(card =>
          card.isLoading ? { ...card, question: 'No file uploaded.', answer: 'Please upload a document first.', isLoading: false } : card
        ));
        setIsLoading(false);
        return;
      }

      try {
        // Get user and notebook info
        let userId = 'web-user';
        let notebookId = currentNotebookId;
        try {
          const { data: { user } } = await supabase.auth.getUser();
          if (user?.id) {
            userId = user.id;
            // Use current notebook ID, or get default if not set
            if (!notebookId) {
              const notebook = await getOrCreateNotebook('Default Notebook');
              notebookId = notebook?.id || null;
            }
          }
        } catch (_) {}

        const response = await fetch(API_ENDPOINTS.FLASHCARDS, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            topic: messageToSend,
            filename: latestFile,
            num_cards: 8,
            notebook_id: notebookId,
            user_id: userId
          })
        });
        const data = await response.json();
        if (response.ok && Array.isArray(data.flashcards)) {
          setFlashcards(data.flashcards.map((fc, idx) => ({
            id: Date.now() + idx,
            question: fc.question,
            answer: fc.answer,
            topic: messageToSend,
            isLoading: false
          })));
        } else {
          setFlashcards(prev => prev.map(card =>
            card.isLoading ? { ...card, question: 'Error generating flashcards.', answer: data.error || 'Unknown error.', isLoading: false } : card
          ));
        }
      } catch (error) {
        setFlashcards(prev => prev.map(card =>
          card.isLoading ? { ...card, question: 'Error generating flashcards.', answer: error.message, isLoading: false } : card
        ));
      }
      setIsLoading(false);
      return;
    }

    // For chatbot tab (or no specific tab), handle as regular chat
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

    // Initialize streaming state
    if (streamingEnabled) {
      setStreamingUpdates([]);
      setIsStreaming(true);
    }

    // Create abort controller for this request
    const controller = new AbortController();
    setAbortController(controller);

    try {
      let response, data;
      
      if (streamingEnabled) {
        // Use streaming endpoint
        // Ensure we pass user_id and notebook_id expected by backend
        let userId = 'web-user';
        let notebookId = currentNotebookId;
        try {
          const { data: { user } } = await supabase.auth.getUser();
          if (user?.id) {
            userId = user.id;
            // Use current notebook ID, or get default if not set
            if (!notebookId) {
              const notebook = await getOrCreateNotebook('Default Notebook');
              notebookId = notebook?.id || null;
            }
          }
        } catch (_) {}
        response = await fetch(API_ENDPOINTS.CHAT_STREAM, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message: messageToSend, user_id: userId, notebook_id: notebookId }),
          signal: controller.signal
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Handle Server-Sent Events stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let finalAnswer = '';
        let finalVideos = [];
        let thinkingSteps = [];

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const update = JSON.parse(line.slice(6));
                  
                  if (update.type === 'stream_complete') {
                    // Stream is complete, break out of the loop
                    break;
                  } else if (update.type === 'error') {
                    throw new Error(update.message);
                  } else if (update.type === 'query_complete') {
                    finalAnswer = update.answer;
                    
                    // Add final completion thinking message
                    const completionMessage = {
                      type: 'thinking',
                      content: `🎯 **Answer Complete**: Generated response successfully`,
                      timestamp: new Date().toISOString(),
                      temporary: true
                    };
                    setMessages(prev => [...prev, completionMessage]);
                    thinkingSteps.push(completionMessage);
                  } else if (update.type === 'multihop_detected') {
                    // Add initial thinking message
                    const thinkingMessage = {
                      type: 'thinking',
                      content: `🚀 **Multi-hop Question Detected**: Breaking down into sub-questions...`,
                      timestamp: new Date().toISOString(),
                      temporary: true
                    };
                    setMessages(prev => [...prev, thinkingMessage]);
                    thinkingSteps.push(thinkingMessage);
                  } else if (update.type === 'single_hop_detected') {
                    // Add initial thinking message for single-hop
                    const thinkingMessage = {
                      type: 'thinking',
                      content: `📝 **Single-hop Question**: Processing your question...`,
                      timestamp: new Date().toISOString(),
                      temporary: true
                    };
                    setMessages(prev => [...prev, thinkingMessage]);
                    thinkingSteps.push(thinkingMessage);
                  } else if (update.type === 'query_start') {
                    // Add initial thinking message
                    const thinkingMessage = {
                      type: 'thinking',
                      content: `🤔 **Analyzing Question**: Understanding what you're asking...`,
                      timestamp: new Date().toISOString(),
                      temporary: true
                    };
                    setMessages(prev => [...prev, thinkingMessage]);
                    thinkingSteps.push(thinkingMessage);
                  } else if (update.type === 'step_start' || update.type === 'step_complete' || update.type === 'synthesis_start' || update.type === 'query_complete') {
                    // Add thinking steps to chat
                    let stepMessage = '';
                    if (update.type === 'step_start') {
                      stepMessage = `🔍 **Step ${update.step}**: Analyzing "${update.sub_question}"`;
                    } else if (update.type === 'step_complete') {
                      stepMessage = `✅ **Step ${update.step} Complete**: Found ${update.chunks_found} relevant chunks`;
                    } else if (update.type === 'synthesis_start') {
                      stepMessage = `🧠 **Synthesizing** final answer from all steps...`;
                    } else if (update.type === 'query_complete') {
                      stepMessage = `🎯 **Complete**: Generated final answer (${update.total_steps} steps, ${update.total_chunks} chunks)`;
                    }
                    
                    if (stepMessage) {
                      const thinkingMessage = {
                        type: 'thinking',
                        content: stepMessage,
                        timestamp: new Date().toISOString(),
                        temporary: true
                      };
                      setMessages(prev => [...prev, thinkingMessage]);
                      thinkingSteps.push(thinkingMessage);
                    }
                  }
                  
                  // Add update to streaming state
                  setStreamingUpdates(prev => [...prev, update]);
                } catch (e) {
                  console.warn('Failed to parse SSE update:', e);
                }
              }
            }
          }
        } finally {
          reader.releaseLock();
        }

        // Keep thinking messages but mark them as completed
        setMessages(prev => prev.map(msg => 
          msg.temporary && msg.type === 'thinking' 
            ? { ...msg, temporary: false, type: 'thinking_complete' }
            : msg
        ));

        // Create final response object
        data = {
          answer: finalAnswer,
          videos: finalVideos,
          is_streaming: true
        };
      } else {
        // Use regular endpoint
        // Ensure we pass user_id and notebook_id expected by backend
        let userId = 'web-user';
        let notebookId = currentNotebookId;
        try {
          const { data: { user } } = await supabase.auth.getUser();
          if (user?.id) {
            userId = user.id;
            // Use current notebook ID, or get default if not set
            if (!notebookId) {
              const notebook = await getOrCreateNotebook('Default Notebook');
              notebookId = notebook?.id || null;
            }
          }
        } catch (_) {}
        response = await fetch(API_ENDPOINTS.CHAT, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message: messageToSend, user_id: userId, notebook_id: notebookId }),
          signal: controller.signal
        });

        data = await response.json();
      }

      // Check if we have a valid answer
      if (!data.answer && !data.error) {
        data.answer = "I'm sorry, I couldn't generate a response. Please try again.";
      }

      if (response.ok) {
        // Add bot response to chat with typing indicator
        const botMessage = {
          type: 'bot',
          content: data.answer || data.response || '',
          timestamp: new Date().toISOString(),
          isNew: true,
          videos: Array.isArray(data.videos) ? data.videos : []
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
      // Clean up streaming state
      setIsStreaming(false);
    }
  };

  const handleFileUpload = async (file) => {
    if (!file) return;

    // Add loading message
    const loadingMessage = {
      type: 'system',
      content: `📤 Uploading "${file.name}"...`,
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, loadingMessage]);

    let supabaseSuccess = false;
    let backendSuccess = false;
    let fileObj = null;
    let backendError = null;
    let supabaseError = null;

    // 1. Upload to Supabase Storage and insert metadata
    try {
      const { data: { user } } = await supabase.auth.getUser();
      // Store files under a per-user folder to satisfy common storage RLS policies
      const filePath = `${user.id}/${Date.now()}_${file.name}`;
      console.log('Current user:', user);
      console.log('User ID:', user?.id);
      console.log('User email:', user?.email);
      
      // Ensure user exists in the profiles table
      const { data: existingUser, error: userCheckError, status } = await supabase
        .from('profiles')
        .select('id')
        .eq('user_id', user.id)
        .single();
      
      // If user not found (status 406), insert the user
      if (!existingUser || userCheckError?.status === 406) {
        const { error: userInsertError } = await supabase
          .from('profiles')
          .insert([{
            user_id: user.id,
            email: user.email,
            full_name: user.user_metadata?.full_name || user.email.split('@')[0]
          }]);
        if (userInsertError) {
          console.error('Error creating user:', userInsertError);
          throw userInsertError;
        }
        console.log('Created user in profiles table');
      }
      
      const { data: storageData, error: storageError } = await supabase
        .storage
        .from('documents')
        .upload(filePath, file);

      console.log('Storage upload result:', { storageData, storageError });
      console.log('Storage error details:', JSON.stringify(storageError, null, 2));
      if (storageError) throw storageError;

      const { data: publicUrlData } = supabase
        .storage
        .from('documents')
        .getPublicUrl(filePath);

      const fileUrl = publicUrlData.publicUrl;

      // Use current notebook or create default
      let notebookId = currentNotebookId;
      if (!notebookId) {
        const notebook = await getOrCreateNotebook('Default Notebook');
        if (notebook) {
          notebookId = notebook.id;
          setCurrentNotebookId(notebookId);
          setCurrentNotebookName(notebook.name);
        }
      }

      // Use notebookId in your insert
      const { data: insertData, error: insertError } = await supabase
        .from('documents')
        .insert([{
          user_id: user.id,
          notebook_id: notebookId,
          filename: file.name,
          original_filename: file.name,
          file_type: file.type,
          file_size: file.size,
          storage_path: filePath,
          file_path: fileUrl,
          processing_status: 'completed',
          status: 'completed'
        }])
        .select()
        .single();

      if (insertError) throw insertError;

      if (!insertData) {
        throw new Error('File insert did not return data');
      }

      fileObj = {
        id: insertData.id,
        name: file.name,
        size: file.size,
        type: file.type,
        url: fileUrl,
        uploadedAt: new Date().toISOString()
      };
      supabaseSuccess = true;
    } catch (error) {
      supabaseError = error;
      console.error('Supabase upload error:', error);
    }

    // 2. Send file to backend API as before
    try {
      const formData = new FormData();
      formData.append('file', file);
      try {
        // Include user_id so backend can write rows under correct user for RLS
        const { data: { user: backendUser } } = await supabase.auth.getUser();
        if (backendUser?.id) {
          formData.append('user_id', backendUser.id);
        }
      } catch (_) {}
      const response = await fetch(API_ENDPOINTS.UPLOAD, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Failed to upload file to backend');
      backendSuccess = true;
    } catch (error) {
      backendError = error;
      console.error('Backend upload error:', error);
    }

    // 3. UI update logic
    setMessages(prev => prev.filter(msg => msg !== loadingMessage));
    if (supabaseSuccess && backendSuccess) {
      const successMessage = {
        type: 'system',
        content: `✅ File "${file.name}" uploaded successfully! You can now ask questions about it.`,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, successMessage]);
      if (fileObj) setUploadedFiles(prev => [...prev, fileObj]);
      
      // Refresh the Supabase files list
      await fetchUserFiles();
    } else {
      let errorMsg = `❌ Failed to upload "${file.name}".`;
      if (supabaseError) errorMsg += ` Supabase: ${supabaseError.message || supabaseError}`;
      if (backendError) errorMsg += ` Backend: ${backendError.message || backendError}`;
      const errorMessage = {
        type: 'error',
        content: errorMsg,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    }

    // Reset the file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDropdownClose = () => {
    setIsDropdownClosing(true);
    setTimeout(() => {
      setIsDropdownOpen(false);
      setIsDropdownClosing(false);
    }, 200); // Match animation duration
  };

  const handleFilesDropdownClose = () => {
    setIsFilesDropdownClosing(true);
    setTimeout(() => {
      setIsFilesDropdownOpen(false);
      setIsFilesDropdownClosing(false);
    }, 200); // Match animation duration
  };

  const handleExamSettingsClose = () => {
    setIsExamSettingsClosing(true);
    setTimeout(() => {
      setIsExamSettingsOpen(false);
      setIsExamSettingsClosing(false);
    }, 200); // Match animation duration
  };

  const handleModelSelectorClose = () => {
    setIsModelSelectorClosing(true);
    setTimeout(() => {
      setIsModelSelectorOpen(false);
      setIsModelSelectorClosing(false);
    }, 200); // Match animation duration
  };

  const handleTabChange = (tabId) => {
    // Reset exam prompt when switching away from exam tab
    if (activeTab === 'exam' && tabId !== 'exam') {
      setExamPrompt('');
    }
    
    // Reset study planner state when switching away from study planner tab
    // Only reset if we're currently on study planner and switching to something else
    if (activeTab === 'study-planner' && tabId !== 'study-planner') {
      setStudyPlannerStep('input');
      setStudyPlannerFiles([]);
      setStudyPlannerDragOver(false);
    }
    
    setActiveTab(tabId);
  };

  // Handle example exam file upload
  const handleExampleExamUpload = async (file) => {
    if (!file) return;
    setUploadingExampleExam(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      // Upload to the backend example_exams directory (custom endpoint)
      const response = await fetch("/api/upload-example-exam", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (response.ok && data.filename) {
        setExampleExamFile(file);
        setExampleExamFilename(data.filename);
      } else {
        alert(data.error || "Failed to upload example exam file.");
      }
    } catch (err) {
      alert("Error uploading example exam file.");
    }
    setUploadingExampleExam(false);
  };

  // Exam Configuration Buttons Component
  const ExamConfigButtons = () => (
    <div className="w-full max-w-3xl">
      <div className="flex flex-wrap justify-center gap-3 items-center">
        {/* Difficulty Button */}
        <button
          onClick={() => {
            const difficulties = ['easy', 'medium', 'hard'];
            const currentIndex = difficulties.indexOf(examConfig.difficulty);
            const nextIndex = (currentIndex + 1) % difficulties.length;
            setExamConfig(prev => ({ ...prev, difficulty: difficulties[nextIndex] }));
          }}
          className="px-4 py-2 bg-black text-white rounded-full hover:bg-gray-800 transition-all duration-200 text-sm font-medium"
        >
          Difficulty: {examConfig.difficulty.charAt(0).toUpperCase() + examConfig.difficulty.slice(1)}
        </button>
        {/* Number of Questions Button */}
        <button
          onClick={() => {
            const options = [5, 10, 15, 20];
            const currentIndex = options.indexOf(examConfig.numQuestions);
            const nextIndex = (currentIndex + 1) % options.length;
            setExamConfig(prev => ({ ...prev, numQuestions: options[nextIndex] }));
          }}
          className="px-4 py-2 bg-black text-white rounded-full hover:bg-gray-800 transition-all duration-200 text-sm font-medium"
        >
          Questions: {examConfig.numQuestions}
        </button>
        {/* Question Types Button */}
        <button
          onClick={() => {
            const types = ['Mixed', 'Multiple Choice', 'Short Answer', 'Problem Solving'];
            const currentIndex = types.indexOf(examConfig.types);
            const nextIndex = (currentIndex + 1) % types.length;
            setExamConfig(prev => ({ ...prev, types: types[nextIndex] }));
          }}
          className="px-4 py-2 bg-black text-white rounded-full hover:bg-gray-800 transition-all duration-200 text-sm font-medium"
        >
          {examConfig.types} Types
        </button>
        {/* Format Button */}
        <button
          onClick={() => {
            const formats = ['PDF', 'Text', 'HTML'];
            const currentIndex = formats.indexOf(examConfig.format);
            const nextIndex = (currentIndex + 1) % formats.length;
            setExamConfig(prev => ({ ...prev, format: formats[nextIndex] }));
          }}
          className="px-4 py-2 bg-black text-white rounded-full hover:bg-gray-800 transition-all duration-200 text-sm font-medium"
        >
          {examConfig.format} Format
        </button>
        {/* Upload Example Exam Button */}
        <button
          type="button"
          className={clsx(
            "px-3 py-1.5 rounded-full border border-gray-200 bg-white text-gray-700 hover:bg-gray-100 transition-all duration-200 text-xs font-medium",
            uploadingExampleExam && "opacity-50 cursor-not-allowed"
          )}
          onClick={() => exampleExamInputRef.current && exampleExamInputRef.current.click()}
          disabled={uploadingExampleExam}
          title="Upload Example Exam Format"
        >
          {uploadingExampleExam ? "Uploading..." : "Upload Example Exam"}
        </button>
        <input
          type="file"
          accept=".pdf,.docx,.txt"
          ref={exampleExamInputRef}
          style={{ display: "none" }}
          onChange={e => {
            if (e.target.files && e.target.files[0]) {
              handleExampleExamUpload(e.target.files[0]);
            }
          }}
        />
        {exampleExamFilename && (
          <span className="ml-2 text-xs text-green-600 font-semibold">
            Current Example Exam: 
            <a
              href={`/exam_generation_feature/example_exams/${encodeURIComponent(exampleExamFilename)}`}
              target="_blank"
              rel="noopener noreferrer"
              className="underline hover:text-green-800 ml-1"
            >
              {exampleExamFilename}
            </a>
          </span>
        )}
      </div>
    </div>
  );

  const handleGenerateStudyPlan = async () => {
    if (!studyPlannerFiles.length) return;
    setStudyPlannerStep('loading');
    setStudyPlanPdfUrl("");
    setStudyPlanEvents([]);
    setStudyPlanError("");
    try {
      const formData = new FormData();
      const fileObj = studyPlannerFiles[0];
      const fileToSend = fileObj.file instanceof File ? fileObj.file : fileObj;
      formData.append('file', fileToSend);
      formData.append('weeks', studyPlannerWeeksRef.current);
      
      console.log('Uploading file for study plan generation:', fileToSend.name);
      
      const response = await fetch(API_ENDPOINTS.STUDYPLAN, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      
      console.log('Study plan response:', data);
      
      if (!response.ok) {
        console.error('Study plan generation failed:', data.error);
        setStudyPlanError(data.error || 'Failed to generate study plan.');
        setStudyPlannerStep('upload');
        return;
      }
      
      console.log('PDF URL received:', data.pdf_url);
      console.log('Events received:', data.events);
      console.log('Number of events:', data.events?.length || 0);
      
      setStudyPlanPdfUrl(data.pdf_url);
      setStudyPlanEvents(data.events || []);
      console.log('Study plan generated, transitioning to calendar step with events:', data.events?.length || 0);
      setStudyPlannerStep('calendar');
    } catch (err) {
      console.error('Study plan generation error:', err);
      setStudyPlanError('An error occurred while generating the study plan.');
      setStudyPlannerStep('upload');
    }
  };

  const handleStudyPlannerFileUpload = (files) => {
    const fileList = Array.from(files).map((file, index) => ({
      id: Date.now() + index,
      name: file.name,
      size: file.size,
      type: file.type,
      file, // store the actual File object
    }));
    setStudyPlannerFiles(prev => [...prev, ...fileList]);
  };

  const handleStudyPlannerSimpleUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch('/api/studyplan-upload', {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();
    if (response.ok) {
      setStudyPlanPdfUrl(data.pdf_url);
      setStudyPlanEvents(data.events || []);
      console.log('Study plan generated (simple upload), transitioning to calendar step with events:', data.events?.length || 0);
      setStudyPlannerStep('calendar');
    } else {
      setStudyPlanError(data.error || 'Failed to generate study plan.');
    }
  };

  const handleAddToCalendar = async () => {
    if (!calendarEmail || !studyPlanEvents.length) return;
    
    setIsAddingToCalendar(true);
    setCalendarResult(null);
    
    try {
      const response = await fetch(API_ENDPOINTS.STUDYPLAN_ADD_TO_CALENDAR, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          events: studyPlanEvents,
          calendar_type: calendarType,
          email: calendarEmail,
        }),
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setCalendarResult({
          success: true,
          message: data.message,
          successful_adds: data.successful_adds,
          total_events: data.total_events,
          failed_events: data.failed_events,
        });
      } else {
        setCalendarResult({
          success: false,
          message: data.error || 'Failed to add events to calendar',
        });
      }
    } catch (error) {
      setCalendarResult({
        success: false,
        message: 'An error occurred while adding events to calendar',
      });
    } finally {
      setIsAddingToCalendar(false);
    }
  };

  return (
    <div className="flex flex-col h-full w-full">
      {/* Top bar: Menu Button and Mode Selector */}
      <div className="px-8 py-4 bg-white/50 backdrop-blur-sm border-b border-gray-100/50">
        <div className="max-w-4xl mx-auto flex justify-between items-center">
          {/* Left side: New Chat button */}
          <button
            onClick={handleNewChat}
            className="flex items-center gap-2 px-4 py-2 rounded-xl border transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-black/10 shadow-sm text-sm bg-white/90 border-gray-200/60 text-gray-700 hover:bg-white hover:border-gray-300/80 hover:shadow-md"
          >
            <div className="flex items-center justify-center w-6 h-6 bg-blue-100/60 rounded-lg">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="text-blue-500">
                <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              </svg>
            </div>
            <span className="font-medium text-sm">New Chat</span>
          </button>
          
          {/* Center: Mode Selector */}
          <div className="flex items-center gap-4">
          {/* Mode selector button - always visible */}
          <button
            onClick={() => {
              if (isDropdownOpen) {
                handleDropdownClose();
              } else {
                setIsDropdownOpen(true);
              }
            }}
            className="flex items-center gap-2 px-4 py-2 rounded-xl border transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-black/10 shadow-sm text-sm bg-white/90 border-gray-200/60 text-gray-700 hover:bg-white hover:border-gray-300/80 hover:shadow-md"
          >
            <div className="flex items-center justify-center w-6 h-6 bg-gray-100/60 rounded-lg">
              <span className="text-sm">{modes.find(mode => mode.value === selectedMode)?.emoji || '🎓'}</span>
            </div>
            <span className="font-medium text-sm">
              {modes.find(mode => mode.value === selectedMode)?.label || 'General Mode'}
            </span>
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" className={`text-gray-500 transition-transform duration-200 ${isDropdownOpen ? 'rotate-180' : ''}`}>
              <path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="1.5"/>
            </svg>
          </button>
          
          {/* Files Button - Show in drag-drop mode, always visible */}
          {selectedMode === 'drag-drop' && (
            <button
              onClick={() => {
                if (isFilesDropdownOpen) {
                  handleFilesDropdownClose();
                } else {
                  setIsFilesDropdownOpen(true);
                }
              }}
              className="flex items-center gap-2 px-4 py-2 rounded-xl border transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-black/10 shadow-sm text-sm bg-white/90 border-gray-200/60 text-gray-700 hover:bg-white hover:border-gray-300/80 hover:shadow-md"
            >
              <div className="flex items-center justify-center w-6 h-6 bg-red-100/60 rounded-lg">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="text-red-500">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke="currentColor" strokeWidth="1.5"/>
                  <polyline points="14,2 14,8 20,8" stroke="currentColor" strokeWidth="1.5"/>
                </svg>
              </div>
              <span className="font-medium text-sm">Files</span>
                {(uploadedFiles.length > 0 || supabaseFiles.length > 0) && (
                <div className="flex items-center justify-center w-5 h-5 bg-red-500 text-white rounded-full text-xs font-medium">
                    {uploadedFiles.length + supabaseFiles.length}
                </div>
              )}
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" className={`text-gray-500 transition-transform duration-200 ${isFilesDropdownOpen ? 'rotate-180' : ''}`}>
                <path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="1.5"/>
              </svg>
            </button>
          )}
          </div>
          
          {/* Right side: Current notebook name */}
          <div className="text-sm text-gray-600 font-medium">
            {currentNotebookName}
          </div>
        </div>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto px-8 py-2" style={{ maxHeight: 'calc(100vh - 160px)' }}>
        <div className="max-w-4xl mx-auto">
          {activeTab === 'exam' ? (
            examPrompt ? (
              <div className="h-full">
                <Exam 
                  initialPrompt={examPrompt} 
                  difficulty={examConfig.difficulty}
                  numQuestions={examConfig.numQuestions}
                  types={examConfig.types}
                  format={examConfig.format}
                  uploadedFiles={uploadedFiles}
                  exampleExamFilename={exampleExamFilename}
                  onPromptProcessed={() => {
                    // Optionally handle when prompt is processed
                  }}
                  onLoadingChange={(loading, stopFn) => {
                    setExamGenerating(loading);
                    setStopExamGeneration(() => stopFn);
                  }}
                  onFullscreenChange={(fullscreen) => {
                    setIsExamFullscreen(fullscreen);
                  }}
                />
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center min-h-[calc(100vh-320px)] text-center py-8 mt-16">
                <AnimatePresence mode="wait">
                  <motion.h1 
                    key={activeTab === 'study-planner' 
                      ? (studyPlannerStep === 'calendar' ? 'study-plan' : 'plan-schedule')
                      : 'general'
                    }
                    initial={{ 
                      opacity: 0, 
                      y: 12, // Smaller movement
                      scale: 0.98, // Less dramatic scale
                      filter: 'blur(0.8px)', // Less blur
                      letterSpacing: '0.015em' // Less spacing
                    }}
                    animate={{ 
                      opacity: 1, 
                      y: 0, 
                      scale: 1,
                      filter: 'blur(0px)',
                      letterSpacing: '0em'
                    }}
                    exit={{ 
                      opacity: 0, 
                      y: -12, // Smaller movement
                      scale: 1.02, // Less dramatic scale
                      filter: 'blur(0.8px)', // Less blur
                      letterSpacing: '-0.015em' // Less spacing
                    }}
                    transition={{ 
                      duration: 0.5, // Quicker
                      ease: [0.25, 0.1, 0.25, 1],
                      opacity: { duration: 0.4 }, // Quicker opacity
                      filter: { duration: 0.45 }, // Quicker filter
                      letterSpacing: { duration: 0.5 } // Quicker spacing
                    }}
                    className="text-3xl md:text-4xl font-bold text-gray-900 mb-8"
                  >
                    {activeTab === 'study-planner' 
                      ? (studyPlannerStep === 'calendar' ? 'Your Study Plan' : 'Plan Your Study Schedule')
                      : 'How can I help you learn today?'
                    }
                  </motion.h1>
                </AnimatePresence>
                <UnifiedInputComponent 
                  placeholder={`Enter a topic to generate ${examConfig.numQuestions} ${examConfig.difficulty} exam questions (e.g., 'Cardiology', 'Neurology')...`}
                  inputMessage={inputMessage}
                  setInputMessage={setInputMessage}
                  handleSendMessage={handleSendMessage}
                  isLoading={isLoading}
                  examGenerating={examGenerating}
                  fileInputRef={fileInputRef}
                  handleFileUpload={handleFileUpload}
                  selectedModel={selectedModel}
                  setSelectedModel={setSelectedModel}
                  tabs={tabs}
                  activeTab={activeTab}
                  handleTabChange={handleTabChange}
                  isExamSettingsOpen={isExamSettingsOpen}
                  handleExamSettingsClose={handleExamSettingsClose}
                  setIsExamSettingsOpen={setIsExamSettingsOpen}
                  // Study Planner specific props (not used in exam mode)
                  studyPlannerStep={studyPlannerStep}
                  setStudyPlannerStep={setStudyPlannerStep}
                  studyPlannerFiles={studyPlannerFiles}
                  setStudyPlannerFiles={setStudyPlannerFiles}
                  studyPlannerDragOver={studyPlannerDragOver}
                  setStudyPlannerDragOver={setStudyPlannerDragOver}
                  handleGenerateStudyPlan={handleGenerateStudyPlan}
                  handleStudyPlannerFileUpload={handleStudyPlannerFileUpload}
                />
                {/* Exam Configuration Buttons - Directly under input box */}
                <div className="mt-3">
                  <ExamConfigButtons />
                      </div>
                    </div>
            )
          ) : activeTab === 'flashcards' ? (
            flashcards.length === 0 ? (
              <div className="flex flex-col items-center justify-center min-h-[calc(100vh-320px)] text-center py-8 mt-16">
                <AnimatePresence mode="wait">
                  <motion.h1 
                    key="flashcards"
                    initial={{ 
                      opacity: 0, 
                      y: 12, // Smaller movement
                      scale: 0.98, // Less dramatic scale
                      filter: 'blur(0.8px)', // Less blur
                      letterSpacing: '0.015em' // Less spacing
                    }}
                    animate={{ 
                      opacity: 1, 
                      y: 0, 
                      scale: 1,
                      filter: 'blur(0px)',
                      letterSpacing: '0em'
                    }}
                    exit={{ 
                      opacity: 0, 
                      y: -12, // Smaller movement
                      scale: 1.02, // Less dramatic scale
                      filter: 'blur(0.8px)', // Less blur
                      letterSpacing: '-0.015em' // Less spacing
                    }}
                    transition={{ 
                      duration: 0.5, // Quicker
                      ease: [0.25, 0.1, 0.25, 1],
                      opacity: { duration: 0.4 }, // Quicker opacity
                      filter: { duration: 0.45 }, // Quicker filter
                      letterSpacing: { duration: 0.5 } // Quicker spacing
                    }}
                    className="text-3xl md:text-4xl font-bold text-gray-900 mb-8"
                  >
                    What topic would you like flashcards for?
                  </motion.h1>
                </AnimatePresence>
                <UnifiedInputComponent 
                  placeholder="Enter a topic to generate flashcards..."
                  inputMessage={inputMessage}
                  setInputMessage={setInputMessage}
                  handleSendMessage={handleSendMessage}
                  isLoading={isLoading}
                  examGenerating={examGenerating}
                  fileInputRef={fileInputRef}
                  handleFileUpload={handleFileUpload}
                  selectedModel={selectedModel}
                  setSelectedModel={setSelectedModel}
                  tabs={tabs}
                  activeTab={activeTab}
                  handleTabChange={handleTabChange}
                  isExamSettingsOpen={isExamSettingsOpen}
                  handleExamSettingsClose={handleExamSettingsClose}
                  setIsExamSettingsOpen={setIsExamSettingsOpen}
                  // Study Planner specific props (not used in flashcard input mode)
                  studyPlannerStep={studyPlannerStep}
                  setStudyPlannerStep={setStudyPlannerStep}
                  studyPlannerFiles={studyPlannerFiles}
                  setStudyPlannerFiles={setStudyPlannerFiles}
                  studyPlannerDragOver={studyPlannerDragOver}
                  setStudyPlannerDragOver={setStudyPlannerDragOver}
                  handleGenerateStudyPlan={handleGenerateStudyPlan}
                  handleStudyPlannerFileUpload={handleStudyPlannerFileUpload}
                />
              </div>
            ) : (
              <div className="py-2">
                <Flashcards flashcards={flashcards} />
                {isLoading && (
                  <div className="flex items-center justify-center py-2">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                    </div>
                    <span className="text-sm text-gray-600 ml-3">Generating flashcards...</span>
                  </div>
                )}
                  </div>
            )
          ) : activeTab === 'study-planner' ? (
            studyPlannerStep === 'calendar' ? (
              <div className="w-full max-w-6xl mx-auto px-4 py-8">
                {console.log('Rendering calendar step with events:', studyPlanEvents.length)}
                {console.log('PDF URL:', studyPlanPdfUrl)}
                {console.log('Events data:', studyPlanEvents)}
                
                {/* PDF Download Section */}
                <div className="text-center mb-8">
                  <h2 className="text-2xl font-bold text-gray-900 mb-6">Your Study Plan PDF</h2>
                  {studyPlanPdfUrl && (
                    <div className="mb-6">
                      <button
                        onClick={handleDownloadStudyPlanPDF}
                        className="inline-flex items-center gap-2 px-6 py-3 bg-black text-white rounded-lg hover:bg-gray-800 transition-colors"
                      >
                        <FileText className="w-5 h-5" />
                        Download Study Plan PDF
                      </button>
                      <p className="text-xs text-gray-500 mt-2">
                        PDF URL: {studyPlanPdfUrl}
                      </p>
                    </div>
                  )}
                </div>

                {/* Main Content Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  
                  {/* Calendar Integration Section */}
                  <div className="bg-white rounded-xl border border-gray-200 p-6">
                    <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
                      <Calendar className="w-5 h-5" />
                      Add to Calendar
                    </h3>
                    <p className="text-gray-600 mb-6">Connect your calendar to automatically add study events</p>
                    
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Calendar Type</label>
                        <select
                          value={calendarType}
                          onChange={(e) => setCalendarType(e.target.value)}
                          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                          <option value="gmail">Gmail Calendar</option>
                          <option value="outlook">Outlook Calendar</option>
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Email Address</label>
                        <input
                          type="email"
                          value={calendarEmail}
                          onChange={(e) => setCalendarEmail(e.target.value)}
                          placeholder={`Enter your ${calendarType === 'gmail' ? 'Gmail' : 'Outlook'} email`}
                          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      </div>
                      
                      <button
                        onClick={handleAddToCalendar}
                        disabled={!calendarEmail || isAddingToCalendar}
                        className="w-full px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-colors"
                      >
                        {isAddingToCalendar ? (
                          <>
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                            Adding to Calendar...
                          </>
                        ) : (
                          <>
                            <Calendar className="w-4 h-4" />
                            Add {studyPlanEvents.length} Events to {calendarType === 'gmail' ? 'Gmail' : 'Outlook'} Calendar
                          </>
                        )}
                      </button>
                    </div>
                    
                    {/* Calendar Result */}
                    {calendarResult && (
                      <div className={`mt-4 p-4 rounded-lg ${
                        calendarResult.success 
                          ? 'bg-green-50 border border-green-200 text-green-800' 
                          : 'bg-red-50 border border-red-200 text-red-800'
                      }`}>
                        <p className="font-medium">{calendarResult.message}</p>
                        {calendarResult.success && calendarResult.successful_adds !== undefined && (
                          <p className="text-sm mt-1">
                            Successfully added {calendarResult.successful_adds} out of {calendarResult.total_events} events.
                            {calendarResult.failed_events && calendarResult.failed_events.length > 0 && (
                              <span className="block mt-1">
                                Failed events: {calendarResult.failed_events.join(', ')}
                              </span>
                            )}
                          </p>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Study Events List */}
                  <div className="bg-white rounded-xl border border-gray-200 p-6">
                    <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
                      <Clock className="w-5 h-5" />
                      Study Events ({studyPlanEvents.length})
                    </h3>
                    {studyPlanEvents.length > 0 ? (
                      <div className="max-h-96 overflow-y-auto space-y-3">
                        {studyPlanEvents.map((event, idx) => (
                          <div key={idx} className="p-3 bg-gray-50 rounded-lg border border-gray-100">
                            <div className="font-medium text-gray-900">
                              {event.summary || event.title || `Study Event ${idx + 1}`}
                            </div>
                            {event.start_datetime && (
                              <div className="text-sm text-blue-600 mt-1">
                                📅 {new Date(event.start_datetime).toLocaleDateString()} at {new Date(event.start_datetime).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                              </div>
                            )}
                            {event.description && (
                              <div className="text-sm text-gray-600 mt-1">
                                {event.description}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        <Clock className="w-12 h-12 mx-auto mb-2 opacity-50" />
                        <p className="mb-2">No events found in your study plan.</p>
                        <p className="text-sm">The study plan PDF may still contain valuable information.</p>
                        {studyPlanPdfUrl && (
                          <p className="text-sm mt-2">Check the PDF download above for your complete study plan.</p>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                {/* Calendar View */}
                {studyPlanEvents.length > 0 && (
                  <div className="mt-8 bg-white rounded-xl border border-gray-200 p-6">
                    <h3 className="text-xl font-semibold text-gray-900 mb-4">Calendar View</h3>
                    <StudyPlanCalendar events={studyPlanEvents} />
                  </div>
                )}

                {/* Action Buttons */}
                <div className="mt-8 flex gap-4 justify-center">
                  <button
                    className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                    onClick={() => {
                      setStudyPlannerStep('input');
                      setStudyPlannerFiles([]);
                      setStudyPlanPdfUrl('');
                      setStudyPlanEvents([]);
                      setCalendarEmail('');
                      setCalendarResult(null);
                    }}
                  >
                    Generate Another Study Plan
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center min-h-[calc(100vh-320px)] text-center py-8 mt-16">
                <AnimatePresence mode="wait">
                  <motion.h1 
                    key={studyPlannerStep === 'calendar' ? 'study-plan' : 'plan-schedule'}
                    initial={{ 
                      opacity: 0, 
                      y: 12, // Smaller movement
                      scale: 0.98, // Less dramatic scale
                      filter: 'blur(0.8px)', // Less blur
                      letterSpacing: '0.015em' // Less spacing
                    }}
                    animate={{ 
                      opacity: 1, 
                      y: 0, 
                      scale: 1,
                      filter: 'blur(0px)',
                      letterSpacing: '0em'
                    }}
                    exit={{ 
                      opacity: 0, 
                      y: -12, // Smaller movement
                      scale: 1.02, // Less dramatic scale
                      filter: 'blur(0.8px)', // Less blur
                      letterSpacing: '-0.015em' // Less spacing
                    }}
                    transition={{ 
                      duration: 0.5, // Quicker
                      ease: [0.25, 0.1, 0.25, 1],
                      opacity: { duration: 0.4 }, // Quicker opacity
                      filter: { duration: 0.45 }, // Quicker filter
                      letterSpacing: { duration: 0.5 } // Quicker spacing
                    }}
                    className="text-3xl md:text-4xl font-bold text-gray-900 mb-8"
                  >
                    {studyPlannerStep === 'calendar' ? 'Your Study Plan' : 'Plan Your Study Schedule'}
                  </motion.h1>
                </AnimatePresence>
                {studyPlannerStep !== 'calendar' && studyPlannerStep !== 'loading' && (
                <UnifiedInputComponent 
                  placeholder="Ask me anything about your studies..."
                  inputMessage={inputMessage}
                  setInputMessage={setInputMessage}
                  handleSendMessage={handleSendMessage}
                  isLoading={isLoading}
                  examGenerating={examGenerating}
                  fileInputRef={fileInputRef}
                  handleFileUpload={handleFileUpload}
                  selectedModel={selectedModel}
                  setSelectedModel={setSelectedModel}
                  tabs={tabs}
                  activeTab={activeTab}
                  handleTabChange={handleTabChange}
                  isExamSettingsOpen={isExamSettingsOpen}
                  handleExamSettingsClose={handleExamSettingsClose}
                  setIsExamSettingsOpen={setIsExamSettingsOpen}
                  // Study Planner specific props
                  studyPlannerStep={studyPlannerStep}
                  setStudyPlannerStep={setStudyPlannerStep}
                  studyPlannerFiles={studyPlannerFiles}
                  setStudyPlannerFiles={setStudyPlannerFiles}
                  studyPlannerDragOver={studyPlannerDragOver}
                  setStudyPlannerDragOver={setStudyPlannerDragOver}
                  handleGenerateStudyPlan={handleGenerateStudyPlan}
                  handleStudyPlannerFileUpload={handleStudyPlannerFileUpload}
                />
                )}
              </div>
            )
          ) : messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center min-h-[calc(100vh-320px)] text-center py-8 mt-16">
              <AnimatePresence mode="wait">
                <motion.h1 
                  key="general"
                  initial={{ 
                    opacity: 0, 
                    y: 12, // Smaller movement
                    scale: 0.98, // Less dramatic scale
                    filter: 'blur(0.8px)', // Less blur
                    letterSpacing: '0.015em' // Less spacing
                  }}
                  animate={{ 
                    opacity: 1, 
                    y: 0, 
                    scale: 1,
                    filter: 'blur(0px)',
                    letterSpacing: '0em'
                  }}
                  exit={{ 
                    opacity: 0, 
                    y: -12, // Smaller movement
                    scale: 1.02, // Less dramatic scale
                    filter: 'blur(0.8px)', // Less blur
                    letterSpacing: '-0.015em' // Less spacing
                  }}
                  transition={{ 
                    duration: 0.5, // Quicker
                    ease: [0.25, 0.1, 0.25, 1],
                    opacity: { duration: 0.4 }, // Quicker opacity
                    filter: { duration: 0.45 }, // Quicker filter
                    letterSpacing: { duration: 0.5 } // Quicker spacing
                  }}
                  className="text-3xl md:text-4xl font-bold text-gray-900 mb-8"
                >
                  How can I help you learn today?
                </motion.h1>
              </AnimatePresence>
              <UnifiedInputComponent 
                placeholder="Ask me anything..."
                inputMessage={inputMessage}
                setInputMessage={setInputMessage}
                handleSendMessage={handleSendMessage}
                isLoading={isLoading}
                examGenerating={examGenerating}
                fileInputRef={fileInputRef}
                handleFileUpload={handleFileUpload}
                selectedModel={selectedModel}
                setSelectedModel={setSelectedModel}
                tabs={tabs}
                activeTab={activeTab}
                handleTabChange={handleTabChange}
                isExamSettingsOpen={isExamSettingsOpen}
                handleExamSettingsClose={handleExamSettingsClose}
                setIsExamSettingsOpen={setIsExamSettingsOpen}
                // Study Planner specific props (not used in chat mode)
                studyPlannerStep={studyPlannerStep}
                setStudyPlannerStep={setStudyPlannerStep}
                studyPlannerFiles={studyPlannerFiles}
                setStudyPlannerFiles={setStudyPlannerFiles}
                studyPlannerDragOver={studyPlannerDragOver}
                setStudyPlannerDragOver={setStudyPlannerDragOver}
                handleGenerateStudyPlan={handleGenerateStudyPlan}
                handleStudyPlannerFileUpload={handleStudyPlannerFileUpload}
              />
              {/* Exam Configuration Buttons - Directly under input box when exam tab is selected */}
              {activeTab === 'exam' && (
                <div className="mt-3">
                  <ExamConfigButtons />
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-8">
              {messages.map((message, index) => (
                <div key={index} className={`${
                  message.isNew ? 'message-enter' : ''
                }`}>
                  {message.type === 'user' ? (
                    <div className={`flex justify-end ${message.isNew ? 'user-message-enter' : ''}`}>
                      <div className="max-w-[80%]">
                        <div className="bg-black text-white rounded-3xl px-6 py-4 shadow-sm">
                          <div className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</div>
                        </div>
                        <div className="text-xs text-gray-400 mt-2 text-right">
                          {new Date(message.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  ) : message.type === 'bot' ? (
                    <div className={`w-full ${
                      message.type === 'bot' && message.isNew ? 'bot-message-enter' : ''
                    }`}>
                      <div className="w-full bg-gray-50/50 border-y border-gray-100/50 py-6 -mx-8 px-8">
                        <div className="max-w-3xl mx-auto">
                          <div className="text-sm leading-relaxed whitespace-pre-wrap text-gray-900">{message.content}</div>
                          
                          {/* YouTube Videos Section */}
                          {message.videos && message.videos.length > 0 && (
                            <div className="mt-6 pt-4 border-t border-gray-200">
                              <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-red-500">
                                  <path d="M23 7l-7 5 7 5V7z" fill="currentColor"/>
                                  <rect x="1" y="5" width="15" height="14" rx="2" ry="2" stroke="currentColor" strokeWidth="2" fill="none"/>
                                </svg>
                                Related Videos
                              </h4>
                              <div className="grid gap-3">
                                {message.videos.map((video, index) => (
                                  <div key={index} className="flex gap-3 p-3 bg-white rounded-lg border border-gray-200 hover:border-gray-300 transition-colors">
                                    {/* Thumbnail */}
                                    <div className="flex-shrink-0">
                                      <img 
                                        src={video.thumbnail} 
                                        alt="Video thumbnail"
                                        className="w-20 h-15 object-cover rounded-md"
                                        onError={(e) => {
                                          e.target.style.display = 'none';
                                          e.target.nextSibling.style.display = 'flex';
                                        }}
                                      />
                                      <div className="w-20 h-15 bg-gray-100 rounded-md flex items-center justify-center" style={{display: 'none'}}>
                                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="text-gray-400">
                                          <path d="M23 7l-7 5 7 5V7z" fill="currentColor"/>
                                          <rect x="1" y="5" width="15" height="14" rx="2" ry="2" stroke="currentColor" strokeWidth="2" fill="none"/>
                                        </svg>
                                      </div>
                                    </div>
                                    
                                    {/* Video Info */}
                                    <div className="flex-1 min-w-0">
                                      <a 
                                        href={video.link} 
                                        target="_blank" 
                                        rel="noopener noreferrer"
                                        className="text-sm font-medium text-blue-600 hover:text-blue-800 hover:underline block truncate"
                                      >
                                        Video {index + 1}
                                      </a>
                                      
                                      {video.timestamp && (
                                        <div className="text-xs text-gray-500 mt-1">
                                          📍 {Math.floor(video.timestamp.start / 60)}:{(video.timestamp.start % 60).toString().padStart(2, '0')} - {Math.floor(video.timestamp.end / 60)}:{(video.timestamp.end % 60).toString().padStart(2, '0')}
                                        </div>
                                      )}
                                      
                                      {video.transcript_snippet && (
                                        <div className="text-xs text-gray-600 mt-2 line-clamp-2">
                                          {video.transcript_snippet.length > 150 
                                            ? `${video.transcript_snippet.substring(0, 150)}...` 
                                            : video.transcript_snippet
                                          }
                                        </div>
                                      )}
                                      
                                      {!video.timestamp && !video.transcript_snippet && (
                                        <div className="text-xs text-gray-500 mt-1">
                                          No transcript available
                                        </div>
                                      )}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                          
                          <div className="text-xs text-gray-400 mt-3">
                            {new Date(message.timestamp).toLocaleTimeString()}
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : message.type === 'thinking' || message.type === 'thinking_complete' ? (
                    <div className="flex justify-start">
                      <div className="max-w-[80%]">
                        <div className={`rounded-3xl px-6 py-4 shadow-sm ${
                          message.type === 'thinking_complete' 
                            ? 'bg-gradient-to-r from-green-50 to-emerald-50 text-green-800 border border-green-200' 
                            : 'bg-gradient-to-r from-blue-50 to-indigo-50 text-blue-800 border border-blue-200'
                        }`}>
                          <div className="text-sm leading-relaxed whitespace-pre-wrap font-medium">
                            {message.content}
                            {message.type === 'thinking_complete' && ' ✅'}
                          </div>
                        </div>
                        <div className="text-xs text-gray-400 mt-2">
                          {new Date(message.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className={`flex justify-start ${
                      message.type === 'system' ? 'system-message-enter' : ''
                    }`}>
                      <div className="max-w-[80%]">
                        <div className={`rounded-3xl px-6 py-4 shadow-sm ${
                          message.type === 'error'
                            ? 'bg-red-50 text-red-700 border border-red-100'
                            : message.type === 'system'
                            ? message.content.includes('🎤 Voice input:')
                              ? 'voice-processing text-white'
                              : message.content.includes('🔊 Generating voice')
                              ? 'voice-generating text-white voice-indicator'
                              : message.content.includes('🎵 Playing voice')
                              ? 'voice-playing text-white voice-indicator'
                              : 'bg-gray-50 text-gray-700 border border-gray-100'
                            : 'bg-gray-50 text-gray-900 border border-gray-100'
                        }`}>
                          <div className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</div>
                        </div>
                        <div className="text-xs text-gray-400 mt-2">
                          {new Date(message.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
              {isLoading && (
                <div className="w-full">
                  <div className="w-full bg-gray-50/50 border-y border-gray-100/50 py-6 -mx-8 px-8">
                    <div className="max-w-3xl mx-auto">
                      <div className="flex items-center gap-3">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                        </div>
                        <span className="text-sm text-gray-600">Thinking...</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* AI Disclaimer */}
      {messages.length === 0 && activeTab === 'chatbot' && (
        <div className="px-8 py-2">
          <div className="max-w-4xl mx-auto">
            <p className="text-xs text-gray-500 text-center">
              Momentum AI can make mistakes. Please verify important information and consult professionals for medical, legal, or financial advice.
            </p>
          </div>
        </div>
      )}

      {/* Chat Input - Simplified for conversation mode */}
      {(messages.length > 0 && activeTab === 'chatbot') && (
        <div className="border-t border-gray-50 px-8 py-6">
          <div className="max-w-3xl mx-auto">
            <UnifiedInputComponent 
                placeholder="Ask me anything..."
              inputMessage={inputMessage}
              setInputMessage={setInputMessage}
              handleSendMessage={handleSendMessage}
              isLoading={isLoading}
              examGenerating={examGenerating}
              fileInputRef={fileInputRef}
              handleFileUpload={handleFileUpload}
              selectedModel={selectedModel}
              setSelectedModel={setSelectedModel}
              tabs={tabs}
              activeTab={activeTab}
              handleTabChange={handleTabChange}
              isExamSettingsOpen={isExamSettingsOpen}
              handleExamSettingsClose={handleExamSettingsClose}
              setIsExamSettingsOpen={setIsExamSettingsOpen}
              // Study Planner specific props (not used in chat mode)
              studyPlannerStep={studyPlannerStep}
              setStudyPlannerStep={setStudyPlannerStep}
              studyPlannerFiles={studyPlannerFiles}
              setStudyPlannerFiles={setStudyPlannerFiles}
              studyPlannerDragOver={studyPlannerDragOver}
              setStudyPlannerDragOver={setStudyPlannerDragOver}
              handleGenerateStudyPlan={handleGenerateStudyPlan}
              handleStudyPlannerFileUpload={handleStudyPlannerFileUpload}
            />
          </div>
        </div>
      )}

      {/* Flashcard Input - When flashcards exist */}
      {(flashcards.length > 0 && activeTab === 'flashcards') && (
        <div className="border-t border-gray-50 px-8 py-6">
          <div className="max-w-3xl mx-auto">
            <UnifiedInputComponent 
              placeholder="Generate more flashcards on another topic..."
              inputMessage={inputMessage}
              setInputMessage={setInputMessage}
              handleSendMessage={handleSendMessage}
              isLoading={isLoading}
              examGenerating={examGenerating}
              fileInputRef={fileInputRef}
              handleFileUpload={handleFileUpload}
              selectedModel={selectedModel}
              setSelectedModel={setSelectedModel}
              tabs={tabs}
              activeTab={activeTab}
              handleTabChange={handleTabChange}
              isExamSettingsOpen={isExamSettingsOpen}
              handleExamSettingsClose={handleExamSettingsClose}
              setIsExamSettingsOpen={setIsExamSettingsOpen}
              // Study Planner specific props (not used in flashcard input mode)
              studyPlannerStep={studyPlannerStep}
              setStudyPlannerStep={setStudyPlannerStep}
              studyPlannerFiles={studyPlannerFiles}
              setStudyPlannerFiles={setStudyPlannerFiles}
              studyPlannerDragOver={studyPlannerDragOver}
              setStudyPlannerDragOver={setStudyPlannerDragOver}
              handleGenerateStudyPlan={handleGenerateStudyPlan}
              handleStudyPlannerFileUpload={handleStudyPlannerFileUpload}
            />
          </div>
        </div>
      )}

      {/* Exam Input - When exam is displayed */}
      {(examPrompt && activeTab === 'exam') && (
        <div className="border-t border-gray-50 px-8 pb-6">
          <div className="max-w-3xl mx-auto">
            <UnifiedInputComponent 
              placeholder="Regenerate exam or ask for modifications..." 
              showExamSettings={true}
              inputMessage={inputMessage}
              setInputMessage={setInputMessage}
              handleSendMessage={handleSendMessage}
              isLoading={isLoading}
              examGenerating={examGenerating}
              fileInputRef={fileInputRef}
              handleFileUpload={handleFileUpload}
              selectedModel={selectedModel}
              setSelectedModel={setSelectedModel}
              tabs={tabs}
              activeTab={activeTab}
              handleTabChange={handleTabChange}
              isExamSettingsOpen={isExamSettingsOpen}
              handleExamSettingsClose={handleExamSettingsClose}
              setIsExamSettingsOpen={setIsExamSettingsOpen}
              // Study Planner specific props (not used in exam input mode)
              studyPlannerStep={studyPlannerStep}
              setStudyPlannerStep={setStudyPlannerStep}
              studyPlannerFiles={studyPlannerFiles}
              setStudyPlannerFiles={setStudyPlannerFiles}
              studyPlannerDragOver={studyPlannerDragOver}
              setStudyPlannerDragOver={setStudyPlannerDragOver}
              handleGenerateStudyPlan={handleGenerateStudyPlan}
              handleStudyPlannerFileUpload={handleStudyPlannerFileUpload}
            />
          </div>
        </div>
      )}

      {/* Exam Settings Dropdown Menu */}
      {(isExamSettingsOpen || isExamSettingsClosing) && (
        <div 
          className="fixed top-1/2 left-1/2 w-[360px] bg-white border border-gray-200/40 rounded-3xl shadow-2xl z-[100] overflow-hidden"
          style={{
            transform: 'translate(-50%, -50%) scale(0.95)',
            opacity: 0,
            animation: isExamSettingsClosing 
              ? 'dropdownExit 200ms ease-out forwards' 
              : 'dropdownEnter 200ms ease-out forwards'
          }}
        >
          {/* Header */}
          <div className="px-6 py-4 border-b border-gray-100/60 bg-gray-50/30">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-8 h-8 bg-gray-100 rounded-lg">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-gray-600">
                  <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.5"/>
                  <path d="M12 1v6m0 10v6m11-7h-6m-10 0H1m15.5-6.5l-4.24 4.24M7.76 7.76L3.52 3.52m12.96 12.96l4.24 4.24M7.76 16.24l-4.24 4.24" stroke="currentColor" strokeWidth="1.5"/>
                </svg>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 text-base">Exam Settings</h3>
                <p className="text-xs text-gray-500">Configure your exam preferences</p>
              </div>
            </div>
          </div>

          {/* Settings Options */}
          <div className="p-4 space-y-4">
            {/* Difficulty Setting */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Difficulty</label>
              <div className="flex gap-2">
                {['easy', 'medium', 'hard'].map((difficulty) => (
                  <button
                    key={difficulty}
                    onClick={() => setExamConfig(prev => ({ ...prev, difficulty }))}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                      examConfig.difficulty === difficulty
                        ? 'bg-black text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {difficulty.charAt(0).toUpperCase() + difficulty.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Number of Questions Setting */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Number of Questions</label>
              <div className="flex gap-2">
                {[5, 10, 15, 20].map((num) => (
                      <button
                    key={num}
                    onClick={() => setExamConfig(prev => ({ ...prev, numQuestions: num }))}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                      examConfig.numQuestions === num
                        ? 'bg-black text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {num}
                      </button>
                    ))}
                  </div>
                </div>

            {/* Question Types Setting */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Question Types</label>
              <button className="w-full px-4 py-2 bg-gray-100 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-200 transition-all duration-200">
                Mixed Types
              </button>
            </div>

            {/* Format Setting */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Format</label>
              <button className="w-full px-4 py-2 bg-gray-100 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-200 transition-all duration-200">
                PDF Format
              </button>
            </div>
          </div>

          {/* Footer */}
          <div className="px-6 py-4 border-t border-gray-100/60 bg-gray-50/30">
                <button
              onClick={handleExamSettingsClose}
              className="w-full px-4 py-2 bg-black text-white rounded-lg text-sm font-medium hover:bg-gray-800 transition-all duration-200"
            >
              Apply Settings
            </button>
          </div>
        </div>
      )}

      {/* Mode Selector Dropdown Menu */}
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

      {/* Files Dropdown Menu */}
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
                <div className="flex items-center justify-center w-8 h-8 bg-red-100 rounded-lg">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-red-500">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke="currentColor" strokeWidth="1.5"/>
                    <polyline points="14,2 14,8 20,8" stroke="currentColor" strokeWidth="1.5"/>
                  </svg>
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 text-base">Files</h3>
                  <p className="text-xs text-gray-500">{uploadedFiles.length + supabaseFiles.length} file{(uploadedFiles.length + supabaseFiles.length) !== 1 ? 's' : ''} uploaded</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => fetchUserFiles()}
                  className="flex items-center gap-2 px-3 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-all duration-200 text-sm font-medium"
                  disabled={loadingFiles}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className={`text-current ${loadingFiles ? 'animate-spin' : ''}`}>
                    <path d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  Refresh
                </button>
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
          </div>

          {/* Files List */}
          <div className="max-h-80 overflow-y-auto">
            {loadingFiles ? (
              <div className="px-6 py-8 text-center">
                <div className="flex items-center justify-center w-12 h-12 bg-gray-100 rounded-xl mx-auto mb-3">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-gray-400"></div>
                </div>
                <p className="text-gray-500 text-sm font-medium mb-1">Loading files...</p>
              </div>
            ) : uploadedFiles.length === 0 && supabaseFiles.length === 0 ? (
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
                {/* Combine current session files and Supabase files */}
                {[...uploadedFiles, ...supabaseFiles].map((file, index) => (
                  <div
                    key={`${file.supabaseDocument ? 'supabase' : 'session'}-${file.id}`}
                    className={`flex items-center gap-4 px-4 py-3 rounded-2xl hover:bg-gray-50/60 transition-all duration-200 group ${
                      index !== (uploadedFiles.length + supabaseFiles.length) - 1 ? 'mb-2' : ''
                    }`}
                  >
                    <div className="flex items-center justify-center w-10 h-10 bg-red-50 rounded-xl">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-red-500">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke="currentColor" strokeWidth="1.5"/>
                        <polyline points="14,2 14,8 20,8" stroke="currentColor" strokeWidth="1.5"/>
                    </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-gray-900 text-sm truncate">{file.name}</p>
                      <p className="text-xs text-gray-500">
                        {file.size > 0 ? `${(file.size / 1024).toFixed(1)} KB • ` : ''}
                        {new Date(file.uploadedAt).toLocaleDateString()}
                        {file.status && (
                          <span className={`ml-2 px-2 py-0.5 rounded-full text-xs ${
                            file.status === 'completed' ? 'bg-green-100 text-green-700' :
                            file.status === 'pending' ? 'bg-yellow-100 text-yellow-700' :
                            file.status === 'failed' ? 'bg-red-100 text-red-700' :
                            'bg-gray-100 text-gray-700'
                          }`}>
                            {file.status}
                          </span>
                        )}
                      </p>
                    </div>
                    {!file.supabaseDocument && (
                    <button
                      onClick={() => {
                        setUploadedFiles(prev => prev.filter(f => f.id !== file.id));
                      }}
                      className="opacity-0 group-hover:opacity-100 p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-all duration-200"
                      title="Remove file"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="text-current">
                        <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                      </svg>
                </button>
                    )}
              </div>
                ))}
            </div>
            )}
          </div>
        </div>
      )}

      {/* New Chat Dropdown */}
      {isNewChatDropdownOpen && (
        <div className="fixed inset-0 z-[130] flex items-center justify-center">
          <div 
            className="fixed inset-0 bg-black/20 backdrop-blur-sm"
            onClick={() => setIsNewChatDropdownOpen(false)}
          />
          <div className="relative bg-white rounded-2xl shadow-2xl border border-gray-200/40 p-6 w-full max-w-md mx-4">
            <div className="text-center">
              <div className="flex items-center justify-center w-12 h-12 bg-blue-50 rounded-xl mx-auto mb-4">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="text-blue-500">
                  <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Choose Notebook</h3>
              <p className="text-gray-600 mb-6">Select an existing notebook or create a new one.</p>
              
              <div className="space-y-3 mb-6 max-h-60 overflow-y-auto">
                {availableNotebooks.map((notebook) => (
                  <button
                    key={notebook.id}
                    onClick={() => {
                      setIsNewChatDropdownOpen(false);
                      switchToNotebook(notebook);
                    }}
                    className={`w-full text-left p-3 rounded-xl border transition-colors ${
                      notebook.id === currentNotebookId
                        ? 'border-blue-200 bg-blue-50 text-blue-900'
                        : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                    }`}
                  >
                    <div className="font-medium">{notebook.name}</div>
                    <div className="text-sm text-gray-500">
                      {new Date(notebook.created_at).toLocaleDateString()}
                    </div>
                  </button>
                ))}
              </div>
              
              <button
                onClick={() => {
                  setIsNewChatDropdownOpen(false);
                  setIsCreatingNewChat(true);
                  setNewChatNotebookName('');
                }}
                className="w-full px-4 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors font-medium"
              >
                Create New Notebook
              </button>
            </div>
          </div>
        </div>
      )}

      {/* New Chat Modal */}
      {isCreatingNewChat && (
        <div className="fixed inset-0 z-[130] flex items-center justify-center">
          <div 
            className="fixed inset-0 bg-black/20 backdrop-blur-sm"
            onClick={cancelNewChat}
          />
          <div className="relative bg-white rounded-2xl shadow-2xl border border-gray-200/40 p-6 w-full max-w-md mx-4">
            <div className="text-center">
              <div className="flex items-center justify-center w-12 h-12 bg-blue-50 rounded-xl mx-auto mb-4">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="text-blue-500">
                  <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Create New Chat</h3>
              <p className="text-gray-600 mb-6">Enter a name for your new notebook to start a fresh conversation.</p>
              
              <input
                type="text"
                value={newChatNotebookName}
                onChange={(e) => setNewChatNotebookName(e.target.value)}
                placeholder="e.g., Biology Study, Math Problems, History Notes"
                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    createNewChat();
                  }
                }}
                autoFocus
              />
              
              <div className="flex gap-3 mt-6">
                <button
                  onClick={cancelNewChat}
                  className="flex-1 px-4 py-3 text-gray-600 bg-gray-100 rounded-xl hover:bg-gray-200 transition-colors font-medium"
                >
                  Cancel
                </button>
                <button
                  onClick={createNewChat}
                  disabled={!newChatNotebookName.trim()}
                  className="flex-1 px-4 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium"
                >
                  Create Chat
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Dropdown Menu - Rendered at root level to appear above backdrop */}
      {(isDropdownOpen || isFilesDropdownOpen || isExamSettingsOpen || isModelSelectorOpen || isDropdownClosing || isFilesDropdownClosing || isModelSelectorClosing) && (
        <div 
          className="fixed inset-0 z-[90]"
          style={{
            backgroundColor: 'rgba(0, 0, 0, 0)',
            backdropFilter: 'blur(0px)',
            animation: (isDropdownClosing || isFilesDropdownClosing || isExamSettingsClosing || isModelSelectorClosing) 
              ? 'backdropExit 200ms ease-out forwards' 
              : 'backdropEnter 200ms ease-out forwards'
          }}
          onClick={() => {
            if (isDropdownOpen) handleDropdownClose();
            if (isFilesDropdownOpen) handleFilesDropdownClose();
            if (isExamSettingsOpen) handleExamSettingsClose();
            if (isModelSelectorOpen) handleModelSelectorClose();
          }}
        />
      )}

      {/* Hidden StudyPlanner for logic */}
      <StudyPlanner 
        onStepChange={setStudyPlannerStep}
        currentStep={studyPlannerStep}
        uploadedFiles={studyPlannerFiles}
        setUploadedFiles={setStudyPlannerFiles}
        isDragOver={studyPlannerDragOver}
        setIsDragOver={setStudyPlannerDragOver}
      />

      {/* Model Selector Dropdown Menu */}
      {(isModelSelectorOpen || isModelSelectorClosing) && (
        <div 
          className="fixed top-1/2 left-1/2 w-[240px] bg-white border border-gray-200/40 rounded-2xl shadow-2xl z-[120] overflow-hidden p-2"
          style={{
            transform: 'translate(-50%, -50%) scale(0.95)',
            opacity: 0,
            animation: isModelSelectorClosing 
              ? 'dropdownExit 200ms ease-out forwards' 
              : 'dropdownEnter 200ms ease-out forwards'
          }}
        >
          {modelOptions.map((model, index) => (
            <button
              key={model.value}
              onClick={() => {
                setSelectedModel(model.value);
                handleModelSelectorClose();
              }}
              className={`w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-gray-50/60 transition-all duration-200 text-sm rounded-xl ${
                selectedModel === model.value 
                  ? 'bg-gray-100/80 text-gray-900' 
                  : 'text-gray-700 hover:text-gray-900'
              } ${index !== modelOptions.length - 1 ? 'mb-1' : ''}`}
            >
              <div className="flex-1">
                <span className="font-medium text-sm">{model.label}</span>
              </div>
              {selectedModel === model.value && (
                <div className="flex items-center justify-center w-5 h-5 bg-black rounded-full">
                  <svg width="8" height="8" viewBox="0 0 24 24" fill="none" className="text-white">
                    <path d="M20 6L9 17l-5-5" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default Chatbot;
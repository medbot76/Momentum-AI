import {
  Download,
  Loader2,
  Maximize,
  Minimize,
  Minus,
  Plus,
  RotateCcw,
} from "lucide-react";
import React, { useEffect, useRef, useState } from "react";
import { API_ENDPOINTS } from "../config";
import clsx from "clsx";

function Exam({
  initialPrompt,
  onPromptProcessed,
  onLoadingChange,
  onFullscreenChange,
  difficulty,
  numQuestions,
  types,
  format,
  uploadedFiles,
  exampleExamFilename,
}) {
  const [examState, setExamState] = useState("initial"); // 'initial', 'generating', 'display'
  const [userPrompt, setUserPrompt] = useState("");
  const [examConfig, setExamConfig] = useState({
    difficulty: difficulty || "medium",
    numQuestions: numQuestions || 10,
    topic: "",
    types: types || "Mixed",
    format: format || "PDF",
  });
  const [generatedExam, setGeneratedExam] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [generationTimeout, setGenerationTimeout] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(0.85); // Default zoom level (85%)
  const [isFullscreen, setIsFullscreen] = useState(false);
  const fileInputRef = useRef(null);
  const [pdfUrl, setPdfUrl] = useState(null);
  const [examContent, setExamContent] = useState("");
  const exampleExamInputRef = useRef(null);
  const [uploadingExampleExam, setUploadingExampleExam] = useState(false);

  // Notify parent component of loading state changes
  useEffect(() => {
    if (onLoadingChange) {
      onLoadingChange(isLoading);
    }
  }, [isLoading, onLoadingChange]);

  // Expose stop generation method to parent
  useEffect(() => {
    if (onLoadingChange) {
      onLoadingChange(isLoading, () => {
        if (generationTimeout) {
          clearTimeout(generationTimeout);
          setGenerationTimeout(null);
          setIsLoading(false);
        }
      });
    }
  }, [isLoading, onLoadingChange, generationTimeout]);

  // Handle initial prompt from Chatbot
  useEffect(() => {
    if (initialPrompt && examState === "initial") {
      // Check if initialPrompt is an object with prompt and config
      if (
        typeof initialPrompt === "object" &&
        initialPrompt.prompt &&
        initialPrompt.config
      ) {
        // Use the configuration from Chatbot
        setExamConfig((prev) => ({
          ...prev,
          ...initialPrompt.config,
          topic: initialPrompt.prompt,
          difficulty: difficulty || initialPrompt.config.difficulty || "medium",
          numQuestions: numQuestions || initialPrompt.config.numQuestions || 10,
          types: types || initialPrompt.config.types || "Mixed",
          format: format || initialPrompt.config.format || "PDF",
        }));
        handleInitialSubmit(initialPrompt.prompt, initialPrompt.config);
      } else {
        // Fallback to old behavior for string prompts
        handleInitialSubmit(initialPrompt);
      }

      if (onPromptProcessed) {
        onPromptProcessed(initialPrompt);
      }
    }
  }, [initialPrompt, difficulty, numQuestions, types, format]);

  // Notify parent component of fullscreen state changes
  useEffect(() => {
    if (onFullscreenChange) {
      onFullscreenChange(isFullscreen);
    }
  }, [isFullscreen, onFullscreenChange]);

  const handleInitialSubmit = (promptText = null, config = null) => {
    const prompt = promptText || userPrompt;
    if (!prompt.trim()) return;

    let newConfig = { ...examConfig };

    if (config) {
      // Use config from Chatbot
      newConfig = { ...newConfig, ...config, topic: prompt };
    } else {
      // Parse the prompt for configuration hints (fallback)
      const promptLower = prompt.toLowerCase();
      newConfig.topic = prompt;

      if (promptLower.includes("easy")) newConfig.difficulty = "easy";
      else if (
        promptLower.includes("hard") ||
        promptLower.includes("difficult")
      )
        newConfig.difficulty = "hard";
      else if (
        promptLower.includes("medium") ||
        promptLower.includes("moderate")
      )
        newConfig.difficulty = "medium";

      const numberMatch = promptLower.match(/(\d+)\s*questions?/);
      if (numberMatch) {
        const num = parseInt(numberMatch[1]);
        if (num >= 1 && num <= 20) newConfig.numQuestions = num;
      }
    }

    setExamConfig(newConfig);

    // Jump directly to generating and show PDF immediately
    setExamState("generating");
    setIsLoading(true);
    handleGenerateExam(newConfig);
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
        // setExampleExamFile(file); // Removed
        // setExampleExamFilename(data.filename); // Removed
      } else {
        alert(data.error || "Failed to upload example exam file.");
      }
    } catch (err) {
      alert("Error uploading example exam file.");
    }
    setUploadingExampleExam(false);
  };

  const handleGenerateExam = async (config = examConfig) => {
    setIsLoading(true);
    // Show placeholder content
    const placeholderExam = `\nPRACTICE EXAM\nDifficulty: ${config.difficulty.toUpperCase()}\nNumber of Questions: ${config.numQuestions}\nTopic: ${config.topic}\n\n## Generating Your Exam...\n\nPlease wait while we create your personalized ${config.difficulty} level exam on ${config.topic}.\n\nThis exam will contain ${config.numQuestions} carefully crafted questions designed to test your knowledge and understanding.\n\nThe exam is being generated using advanced AI to ensure high-quality, relevant questions that match your specified difficulty level.\n\n---\n*AI Generated Practice Exam - ${config.difficulty.charAt(0).toUpperCase() + config.difficulty.slice(1)} Level*`;
    setGeneratedExam(placeholderExam);
    setExamContent(placeholderExam);
    setExamState("display");

    try {
      // First, get the exam content as text
      const contentResponse = await fetch(API_ENDPOINTS.EXAM_PDF, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: config.topic,
          difficulty: config.difficulty,
          num_questions: config.numQuestions,
          types: config.types,
          format: "Text", // Request text format for display
          files: uploadedFiles ? uploadedFiles.map((f) => f.name) : [],
          example_exam_filename: exampleExamFilename || undefined,
        }),
      });

      if (contentResponse.ok) {
        const examText = await contentResponse.text();
        setExamContent(examText);
        setGeneratedExam(examText);
      } else {
        const errorText = "Failed to generate exam content.";
        setExamContent(errorText);
        setGeneratedExam(errorText);
      }

      // Then, get the PDF for download
      const pdfResponse = await fetch(API_ENDPOINTS.EXAM_PDF, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: config.topic,
          difficulty: config.difficulty,
          num_questions: config.numQuestions,
          types: config.types,
          format: "PDF", // Request PDF format for download
          files: uploadedFiles ? uploadedFiles.map((f) => f.name) : [],
          example_exam_filename: exampleExamFilename || undefined,
        }),
      });

      if (pdfResponse.ok) {
        const blob = await pdfResponse.blob();
        const url = URL.createObjectURL(blob);
        setPdfUrl(url);
      }

      setIsLoading(false);
      setGenerationTimeout(null);
    } catch (err) {
      const errorText = "Error contacting backend for exam generation.";
      setExamContent(errorText);
      setGeneratedExam(errorText);
      setIsLoading(false);
      setGenerationTimeout(null);
    }
  };

  const handleRegenerate = () => {
    setIsLoading(true);
    handleGenerateExam();
  };

  const handleDownloadPDF = () => {
    if (pdfUrl) {
      // Download the actual PDF
      const a = document.createElement("a");
      a.href = pdfUrl;
      a.download = `exam_${examConfig.topic.replace(/\s+/g, "_")}_${Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } else {
      // Fallback to text download if PDF is not available
      const blob = new Blob([examContent], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `exam_${examConfig.topic.replace(/\s+/g, "_")}_${Date.now()}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  // Initial state - return null to let Chatbot handle the input interface
  if (examState === "initial") {
    return null;
  }

  // Display state - PDF-like exam view (now the only state after initial)
  if (examState === "display" || examState === "generating") {
    return (
      <div
        className={`flex flex-col h-full transition-all duration-700 ease-[cubic-bezier(0.32,0.72,0,1)] ${
          isFullscreen ? "fixed inset-0 z-50 bg-gray-50" : "bg-gray-50"
        }`}
      >
        {/* Show current example exam if present */}
        {exampleExamFilename && (
          <div className="w-full flex justify-center items-center py-2 bg-green-50 border-b border-green-200">
            <span className="text-xs text-green-700 font-semibold">
              Current Example Exam:&nbsp;
              <a
                href={`/exam_generation_feature/example_exams/${encodeURIComponent(exampleExamFilename)}`}
                target="_blank"
                rel="noopener noreferrer"
                className="underline hover:text-green-800"
              >
                {exampleExamFilename}
              </a>
            </span>
          </div>
        )}
        {/* Top Action Bar - Compact Rounded Style */}
        <div
          className={`sticky top-0 z-10 px-6 transition-all duration-400 ease-[cubic-bezier(0.32,0.72,0,1)] ${
            isFullscreen ? "py-6" : "py-3"
          }`}
        >
          <div
            className={`bg-white/95 backdrop-blur-sm border border-gray-200/60 rounded-full shadow-lg px-4 py-2 mx-auto w-fit transition-all duration-400 ease-[cubic-bezier(0.32,0.72,0,1)] ${
              isFullscreen
                ? "transform scale-110 shadow-2xl"
                : "transform scale-100"
            }`}
          >
            <div className="flex items-center justify-center gap-3">
              <button
                onClick={handleRegenerate}
                disabled={isLoading}
                className="flex items-center gap-1.5 px-3 py-1.5 text-gray-600 hover:text-gray-900 hover:bg-gray-100/60 rounded-full transition-all duration-200 ease-out disabled:opacity-50 text-xs font-medium"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Regenerate</span>
              </button>

              {/* Zoom Controls */}
              <div className="flex items-center border border-gray-200/60 rounded-full bg-gray-50/40 transition-all duration-200 ease-out">
                <button
                  onClick={() =>
                    setZoomLevel((prev) => Math.max(0.5, prev - 0.1))
                  }
                  disabled={zoomLevel <= 0.5}
                  className="p-1.5 text-gray-600 hover:text-gray-900 hover:bg-gray-100/60 rounded-full transition-all duration-200 ease-out disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Zoom out"
                >
                  <Minus className="w-3.5 h-3.5" />
                </button>
                <div className="px-2 py-1 text-xs font-medium text-gray-700 min-w-[45px] text-center transition-all duration-200 ease-out">
                  {Math.round(zoomLevel * 100)}%
                </div>
                <button
                  onClick={() =>
                    setZoomLevel((prev) => Math.min(1.5, prev + 0.1))
                  }
                  disabled={zoomLevel >= 1.5}
                  className="p-1.5 text-gray-600 hover:text-gray-900 hover:bg-gray-100/60 rounded-full transition-all duration-200 ease-out disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Zoom in"
                >
                  <Plus className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Fullscreen Toggle Button */}
              <button
                onClick={() => setIsFullscreen(!isFullscreen)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full transition-all duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] text-xs font-medium ${
                  isFullscreen
                    ? "text-red-500 hover:text-red-600 hover:bg-red-50/60 bg-red-50/40"
                    : "text-gray-600 hover:text-gray-900 hover:bg-gray-100/60"
                }`}
                title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
              >
                <div
                  className={`transition-transform duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] ${isFullscreen ? "rotate-90" : "rotate-0"}`}
                >
                  {isFullscreen ? (
                    <Minimize className="w-3.5 h-3.5" />
                  ) : (
                    <Maximize className="w-3.5 h-3.5" />
                  )}
                </div>
                <span className="transition-opacity duration-200 ease-out">
                  {isFullscreen ? "Exit" : "Fullscreen"}
                </span>
              </button>

              <button
                onClick={handleDownloadPDF}
                disabled={isLoading}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-black text-white rounded-full hover:bg-gray-800 transition-all duration-200 ease-out disabled:opacity-50 text-xs font-medium"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Download PDF</span>
              </button>
            </div>
          </div>
        </div>

        {/* PDF-like Exam Content */}
        <div
          className={`flex-1 overflow-y-auto transition-all duration-600 ease-[cubic-bezier(0.32,0.72,0,1)] ${
            isFullscreen ? "p-8 pb-32" : "p-6 pb-32"
          }`}
        >
          <div
            className={`mx-auto transition-all duration-600 ease-[cubic-bezier(0.32,0.72,0,1)] ${
              isFullscreen ? "max-w-5xl" : "max-w-4xl"
            }`}
          >
            <div
              className={`bg-white rounded-lg overflow-hidden transition-all duration-600 ease-[cubic-bezier(0.32,0.72,0,1)] ${
                isFullscreen ? "shadow-2xl" : "shadow-lg"
              }`}
              style={{
                transform: `scale(${zoomLevel}) ${isFullscreen ? "translateY(-8px)" : "translateY(0px)"}`,
                transformOrigin: "top center",
                marginBottom: `${(1 - zoomLevel) * 100}px`,
                transition: "all 0.6s cubic-bezier(0.32, 0.72, 0, 1)",
              }}
            >
              {/* Exam Header */}
              <div
                className={`bg-gray-50/60 px-8 border-b border-gray-200/60 transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] ${
                  isFullscreen ? "py-8" : "py-6"
                }`}
              >
                <div className="text-center">
                  <h1
                    className={`font-bold text-gray-900 mb-2 transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] ${
                      isFullscreen ? "text-3xl" : "text-2xl"
                    }`}
                  >
                    PRACTICE EXAM
                  </h1>
                  <div
                    className={`flex justify-center gap-8 text-gray-600 transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] ${
                      isFullscreen ? "text-base" : "text-sm"
                    }`}
                  >
                    <span>
                      Difficulty:{" "}
                      <strong className="text-gray-900">
                        {examConfig.difficulty.toUpperCase()}
                      </strong>
                    </span>
                    <span>
                      Questions:{" "}
                      <strong className="text-gray-900">
                        {examConfig.numQuestions}
                      </strong>
                    </span>
                    <span>
                      Topic:{" "}
                      <strong className="text-gray-900">
                        {examConfig.topic}
                      </strong>
                    </span>
                  </div>
                  {isLoading && (
                    <div className="mt-3 flex items-center justify-center gap-2 transition-all duration-300 ease-out">
                      <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />
                      <span className="text-sm text-blue-600">
                        Generating exam...
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Exam Content */}
              <div
                className={`transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] ${
                  isFullscreen ? "p-10" : "p-8"
                }`}
              >
                <div className="prose prose-gray max-w-none">
                  {!examContent ? (
                    <div className="text-center py-8">
                      <p className="text-gray-500">
                        No exam content available.
                      </p>
                    </div>
                  ) : (
                    examContent.split("\n").map((line, index) => {
                      const trimmedLine = line.trim();

                      if (!trimmedLine)
                        return <div key={index} className="h-4" />;

                      // Main headings
                      if (trimmedLine.startsWith("##")) {
                        return (
                          <h2
                            key={index}
                            className="text-xl font-bold text-gray-900 mt-8 mb-4 pb-2 border-b border-gray-200"
                          >
                            {trimmedLine.replace("##", "").trim()}
                          </h2>
                        );
                      }

                      // Questions
                      if (
                        trimmedLine.startsWith("**Question") ||
                        trimmedLine.startsWith("Question")
                      ) {
                        return (
                          <div
                            key={index}
                            className="mt-6 mb-3 p-4 bg-gray-50 rounded-lg border-l-4 border-blue-500"
                          >
                            <p className="font-semibold text-gray-900 leading-relaxed">
                              {trimmedLine.replace(/\*\*/g, "")}
                            </p>
                          </div>
                        );
                      }

                      // Multiple choice options
                      if (/^[A-D]\)/.test(trimmedLine)) {
                        return (
                          <div key={index} className="ml-6 mb-2">
                            <p className="text-gray-700 hover:bg-gray-100 p-2 rounded transition-colors duration-200">
                              {trimmedLine}
                            </p>
                          </div>
                        );
                      }

                      // Bold text
                      if (trimmedLine.includes("**")) {
                        return (
                          <p
                            key={index}
                            className="font-semibold text-gray-900 mt-4 mb-2"
                          >
                            {trimmedLine.replace(/\*\*/g, "")}
                          </p>
                        );
                      }

                      // Instructions
                      if (trimmedLine.startsWith("**Instructions:")) {
                        return (
                          <div
                            key={index}
                            className="mt-4 mb-3 p-3 bg-blue-50 border-l-4 border-blue-400 rounded"
                          >
                            <p className="text-blue-800 font-medium">
                              {trimmedLine.replace(/\*\*/g, "")}
                            </p>
                          </div>
                        );
                      }

                      // Section headers (like "Multiple Choice Questions")
                      if (trimmedLine.match(/^[A-Z][A-Z\s]+:$/)) {
                        return (
                          <h3
                            key={index}
                            className="text-lg font-semibold text-gray-800 mt-6 mb-3"
                          >
                            {trimmedLine}
                          </h3>
                        );
                      }

                      // Regular paragraphs
                      return (
                        <p
                          key={index}
                          className="text-gray-700 leading-relaxed mb-2"
                        >
                          {trimmedLine}
                        </p>
                      );
                    })
                  )}
                </div>
              </div>

              {/* Exam Footer */}
              <div
                className={`bg-gray-50/60 px-8 border-t border-gray-200/60 text-center transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] ${
                  isFullscreen ? "py-6" : "py-4"
                }`}
              >
                <p
                  className={`text-gray-500 transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] ${
                    isFullscreen ? "text-base" : "text-sm"
                  }`}
                >
                  AI Generated Practice Exam • Generated on{" "}
                  {new Date().toLocaleDateString()}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return null;
}

export default Exam;

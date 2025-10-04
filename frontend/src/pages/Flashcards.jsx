import React, { useState } from 'react';

function Flashcards({ flashcards = [] }) {
  const [current, setCurrent] = useState(0);
  const [showAnswer, setShowAnswer] = useState(false);
  const [isFlipping, setIsFlipping] = useState(false);
  const [isMounted, setIsMounted] = useState(false);
  const containerRef = React.useRef(null);

  // Ensure component is fully mounted and CSS is loaded
  React.useEffect(() => {
    const timer = setTimeout(() => {
      setIsMounted(true);
    }, 50); // Quick initialization for better UX
    
    return () => clearTimeout(timer);
  }, []);

  // Reset to first card and hide answer when flashcards change
  React.useEffect(() => {
    setCurrent(0);
    setShowAnswer(false);
  }, [flashcards]);

  if (flashcards.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center w-full h-full py-8">
        <div className="text-center">
          <div className="text-5xl mb-3">🃏</div>
          <h2 className="text-xl font-bold text-gray-800 mb-2">No flashcards yet</h2>
          <p className="text-gray-600">Type a topic below to generate flashcards!</p>
        </div>
      </div>
    );
  }

  const handlePrevious = () => {
    if (current > 0) {
      setCurrent(current - 1);
      setShowAnswer(false);
    }
  };

  const handleNext = () => {
    if (current < flashcards.length - 1) {
      setCurrent(current + 1);
      setShowAnswer(false);
    }
  };

  const handleFlip = () => {
    if (flashcards[current].isLoading || isFlipping || !isMounted || !containerRef.current) return;
    
    setIsFlipping(true);
    
    // Simplified animation approach
    setShowAnswer(prev => {
      const newShowAnswer = !prev;
      
      // Schedule the animation completion
      setTimeout(() => {
        setIsFlipping(false);
      }, 250);
      
      return newShowAnswer;
    });
  };

  return (
    <div className="flex flex-col items-center justify-start w-full h-full py-2 pt-16">
      {/* Navigation and Card - explicitly centered */}
      <div className="flex items-center justify-center gap-8 mb-3">
        {/* Previous Button */}
        <button
          onClick={handlePrevious}
          className={`w-12 h-12 rounded-full bg-black hover:bg-gray-800 flex items-center justify-center transition-all duration-200 shadow-sm ${
            current === 0 ? 'opacity-40 cursor-not-allowed' : 'hover:shadow-lg'
          }`}
          disabled={current === 0}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-white">
            <path d="M15 18l-6-6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>

        {/* Flashcard Container with 3D Perspective */}
        <div className="relative" style={{ perspective: '1000px' }}>
          <div 
            className={`flashcard-flip-container cursor-pointer ${isFlipping ? 'flipping' : ''} ${showAnswer ? 'flipped' : ''}`}
            onClick={handleFlip}
            style={{
              width: '600px',
              height: '360px'
            }}
            ref={containerRef}
          >
            {/* Front of Card (Question) */}
            <div 
              className="flashcard-face flashcard-front bg-white rounded-3xl shadow-lg border border-gray-100/50"
            >
              <div className="absolute inset-0 flex flex-col rounded-3xl p-8">
                {/* Card Counter - positioned with consistent top padding */}
                <div className="flex justify-center mb-6">
                  <div className="text-gray-400 text-base font-medium">
                    {current + 1} of {flashcards.length}
                  </div>
                </div>
                
                {flashcards[current].isLoading ? (
                  <div className="flex flex-col items-center justify-center flex-1">
                    <div className="flex space-x-1 mb-4">
                      <div className="w-3 h-3 bg-gray-400 rounded-full animate-pulse"></div>
                      <div className="w-3 h-3 bg-gray-400 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                      <div className="w-3 h-3 bg-gray-400 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                    </div>
                    <p className="text-xl font-medium text-gray-600 text-center leading-relaxed">
                      {flashcards[current].question}
                    </p>
                  </div>
                ) : (
                  <>
                    {/* Main content area - centered with proper spacing */}
                    <div className="flex-1 flex items-center justify-center px-4">
                      <div className="text-2xl font-semibold text-gray-900 text-center leading-relaxed">
                        {flashcards[current].question}
                      </div>
                    </div>
                    
                    {/* Bottom text - positioned with consistent bottom padding */}
                    <div className="flex items-center justify-center gap-2 text-gray-500 text-base font-medium mt-6">
                      <span>Tap to reveal answer</span>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" className="text-gray-400">
                        <path d="M7 13l3 3 7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Back of Card (Answer) */}
            <div 
              className="flashcard-face flashcard-back bg-white rounded-3xl shadow-lg border border-gray-100/50"
            >
              <div className="absolute inset-0 flex flex-col rounded-3xl p-8">
                {/* Card Counter - positioned with consistent top padding */}
                <div className="flex justify-center mb-6">
                  <div className="text-gray-400 text-base font-medium">
                    {current + 1} of {flashcards.length}
                  </div>
                </div>
                
                {!flashcards[current].isLoading && (
                  <div className="flex-1 flex items-center justify-center px-4">
                    <div className="text-2xl font-semibold text-gray-900 text-center leading-relaxed">
                      {flashcards[current].answer}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Next Button */}
        <button
          onClick={handleNext}
          className={`w-12 h-12 rounded-full bg-black hover:bg-gray-800 flex items-center justify-center transition-all duration-200 shadow-sm ${
            current === flashcards.length - 1 ? 'opacity-40 cursor-not-allowed' : 'hover:shadow-lg'
          }`}
          disabled={current === flashcards.length - 1}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-white">
            <path d="M9 18l6-6-6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>

      {/* Spaced Repetition Buttons - Show when answer is revealed */}
      {showAnswer && !flashcards[current].isLoading && (
        <div className="flex items-center justify-center gap-2 w-full max-w-xl">
          <button 
            className="flex flex-col items-center justify-center px-4 py-2 bg-gray-100 hover:bg-black hover:text-white text-gray-800 rounded-full font-medium text-sm transition-all duration-200 shadow-sm hover:shadow-md min-w-[80px] h-[50px]"
            onClick={() => {
              // Handle "Again" - card will repeat soon
              console.log('Again clicked');
            }}
          >
            <span className="text-xs mb-0.5 opacity-70">1m</span>
            <span className="text-sm font-semibold">Again</span>
          </button>
          <button 
            className="flex flex-col items-center justify-center px-4 py-2 bg-gray-100 hover:bg-black hover:text-white text-gray-800 rounded-full font-medium text-sm transition-all duration-200 shadow-sm hover:shadow-md min-w-[80px] h-[50px]"
            onClick={() => {
              // Handle "Hard" - card will repeat with shorter interval
              console.log('Hard clicked');
            }}
          >
            <span className="text-xs mb-0.5 opacity-70">10m</span>
            <span className="text-sm font-semibold">Hard</span>
          </button>
          <button 
            className="flex flex-col items-center justify-center px-4 py-2 bg-gray-100 hover:bg-black hover:text-white text-gray-800 rounded-full font-medium text-sm transition-all duration-200 shadow-sm hover:shadow-md min-w-[80px] h-[50px]"
            onClick={() => {
              // Handle "Good" - card will repeat with normal interval
              console.log('Good clicked');
            }}
          >
            <span className="text-xs mb-0.5 opacity-70">4d</span>
            <span className="text-sm font-semibold">Good</span>
          </button>
          <button 
            className="flex flex-col items-center justify-center px-4 py-2 bg-gray-100 hover:bg-black hover:text-white text-gray-800 rounded-full font-medium text-sm transition-all duration-200 shadow-sm hover:shadow-md min-w-[80px] h-[50px]"
            onClick={() => {
              // Handle "Easy" - card will repeat with longer interval
              console.log('Easy clicked');
            }}
          >
            <span className="text-xs mb-0.5 opacity-70">10d</span>
            <span className="text-sm font-semibold">Easy</span>
          </button>
        </div>
      )}
    </div>
  );
}

export default Flashcards;

const StudyPlanner = ({ 
  onStepChange, 
  currentStep, 
  uploadedFiles, 
  setUploadedFiles, 
  isDragOver, 
  setIsDragOver 
}) => {
  const handleFileUpload = (files) => {
    const fileList = Array.from(files).map((file, index) => ({
      id: Date.now() + index,
      name: file.name,
      size: file.size,
      type: file.type
    }));
    setUploadedFiles(prev => [...prev, ...fileList]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    handleFileUpload(files);
  };

  const handleGenerateStudyPlan = () => {
    onStepChange('loading');
    // Simulate loading for 2 seconds
    setTimeout(() => {
      onStepChange('calendar');
    }, 2000);
  };

  const resetToInput = () => {
    onStepChange('input');
    setUploadedFiles([]);
  };

  // This component doesn't render anything - it just provides the logic
  // The actual UI morphing happens in UnifiedInputComponent
  return null;
};

export default StudyPlanner;
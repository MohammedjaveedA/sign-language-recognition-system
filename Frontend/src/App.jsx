import React, { useState, useRef, useEffect } from 'react';
import io from 'socket.io-client';
import './App.css';

const App = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [prediction, setPrediction] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [handsDetected, setHandsDetected] = useState(false);
  const [videoFrame, setVideoFrame] = useState('');
  const [error, setError] = useState('');
  
  // Translation State
  const [targetLanguage, setTargetLanguage] = useState('en');
  const [languages, setLanguages] = useState({});
  const [translatedText, setTranslatedText] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);
  const [showTranslation, setShowTranslation] = useState(false);

  // Available Signs State
  const [availableSigns, setAvailableSigns] = useState([]);
  const [isLoadingSigns, setIsLoadingSigns] = useState(false);

  const socketRef = useRef(null);
  const videoRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    console.log('🔌 Connecting to WebSocket server...');
    
    socketRef.current = io('http://localhost:5000', {
      transports: ['websocket'],
      timeout: 5000,
      reconnectionAttempts: 3,
      reconnectionDelay: 1000
    });

    socketRef.current.on('connect', () => {
      console.log('✅ Connected to server');
      setIsConnected(true);
      setError('');
      // Load available signs when connected
      loadAvailableSigns();
    });

    socketRef.current.on('disconnect', (reason) => {
      console.log('❌ Disconnected from server:', reason);
      setIsConnected(false);
      setIsProcessing(false);
      setError('Disconnected from server');
    });

    socketRef.current.on('connected', (data) => {
      console.log('📡 Server:', data.message);
    });

    socketRef.current.on('recognition_started', () => {
      console.log('🎬 Recognition started - waiting for frames...');
      setIsProcessing(true);
      setError('');
      setPrediction('');
      setConfidence(0);
      setHandsDetected(false);
      // Reset translation when starting
      setTranslatedText('');
      setShowTranslation(false);
    });

    socketRef.current.on('recognition_stopped', () => {
      console.log('⏹️ Recognition stopped');
      setIsProcessing(false);
      setVideoFrame('');
    });

    socketRef.current.on('frame_processed', (data) => {
      // Update video frame
      if (data.frame) {
        setVideoFrame(data.frame);
      }

      // Update recognition data
      setPrediction(data.prediction || '');
      setConfidence(data.confidence || 0);
      setHandsDetected(data.hands_detected || false);
    });

    socketRef.current.on('language_changed', (data) => {
      if (data.success) {
        setTargetLanguage(data.language);
        console.log(`🌐 Language changed to: ${data.language_name}`);
        // Clear translation when language changes
        setTranslatedText('');
        setShowTranslation(false);
      }
    });

    socketRef.current.on('translation_result', (data) => {
      setIsTranslating(false);
      setTranslatedText(data.translated_text);
      setShowTranslation(true);
    });

    socketRef.current.on('available_signs_updated', (data) => {
      console.log('🔄 Available signs updated:', data.signs);
      setAvailableSigns(data.signs || []);
    });

    socketRef.current.on('error', (data) => {
      console.error('❌ Server error:', data.message);
      setError(data.message);
    });

    socketRef.current.on('connect_error', (error) => {
      console.error('❌ Connection error:', error);
      setError('Failed to connect to server. Make sure backend is running on port 5000.');
    });

    // Load supported languages on component mount
    loadSupportedLanguages();

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  const loadSupportedLanguages = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/languages');
      const data = await response.json();
      setLanguages(data.languages);
      setTargetLanguage(data.current_language);
    } catch (error) {
      console.error('Failed to load languages:', error);
    }
  };

  const loadAvailableSigns = async () => {
    try {
      setIsLoadingSigns(true);
      const response = await fetch('http://localhost:5000/api/available-signs');
      const data = await response.json();
      setAvailableSigns(data.signs || []);
    } catch (error) {
      console.error('Failed to load available signs:', error);
      // Fallback to default signs if API fails
      setAvailableSigns(['Hello', 'Thank You', 'Yes', 'No', 'Water','1','2','3','4']);
    } finally {
      setIsLoadingSigns(false);
    }
  };

  const changeLanguage = async (languageCode) => {
    if (socketRef.current && isConnected) {
      socketRef.current.emit('set_language', { language: languageCode });
    } else {
      // Fallback to HTTP API
      try {
        const response = await fetch('http://localhost:5000/api/set-language', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ language: languageCode }),
        });
        const data = await response.json();
        if (data.success) {
          setTargetLanguage(data.language);
        }
      } catch (error) {
        console.error('Failed to change language:', error);
      }
    }
  };

  const translateCurrentSign = () => {
    if (prediction && socketRef.current && isConnected) {
      setIsTranslating(true);
      setShowTranslation(false);
      socketRef.current.emit('translate_text', { text: prediction });
    } else {
      // Fallback to HTTP API
      translateTextHTTP(prediction);
    }
  };

  const translateTextHTTP = async (text) => {
    try {
      setIsTranslating(true);
      setShowTranslation(false);
      const response = await fetch('http://localhost:5000/api/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });
      const data = await response.json();
      setTranslatedText(data.translated_text);
      setShowTranslation(true);
    } catch (error) {
      console.error('Translation failed:', error);
      setError('Translation service unavailable');
    } finally {
      setIsTranslating(false);
    }
  };

  const speakText = () => {
    if (prediction && socketRef.current) {
      socketRef.current.emit('speak_text', { text: prediction });
    }
  };

  const speakTranslatedText = () => {
    if (translatedText && socketRef.current) {
      socketRef.current.emit('speak_text', { text: translatedText });
    }
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      alert('Text copied to clipboard!');
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const refreshAvailableSigns = () => {
    loadAvailableSigns();
  };

  const startRecognition = () => {
    if (socketRef.current && isConnected) {
      setError('');
      setPrediction('');
      setConfidence(0);
      setHandsDetected(false);
      setTranslatedText('');
      setShowTranslation(false);
      socketRef.current.emit('start_recognition');
    } else {
      setError('Not connected to server');
    }
  };

  const stopRecognition = () => {
    if (socketRef.current) {
      socketRef.current.emit('stop_recognition');
    }
  };

  const testConnection = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/health');
      const data = await response.json();
      console.log('🏥 Health check:', data);
      setError(data.status === 'healthy' ? '' : 'Server unhealthy');
      return data.status === 'healthy';
    } catch (error) {
      console.error('Health check failed:', error);
      setError('Server not responding');
      return false;
    }
  };

  // Video Feed with Detection Area
  const VideoFeed = () => {
    if (!videoFrame) {
      return (
        <div className="camera-placeholder">
          <div className="placeholder-content">
            <div className="camera-icon">📷</div>
            <p>{isProcessing ? 'Starting camera...' : 'Camera not active'}</p>
            {isProcessing && <div className="loading-spinner"></div>}
          </div>
          {/* Detection area overlay for placeholder */}
          <div className="detection-area-overlay">
            <div className="detection-box">
              <div className="corner top-left"></div>
              <div className="corner top-right"></div>
              <div className="corner bottom-left"></div>
              <div className="corner bottom-right"></div>
              <div className="detection-label">
                👆 Hand Detection Area
              </div>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="video-container" ref={containerRef}>
        <img 
          ref={videoRef}
          src={videoFrame} 
          alt="Live Recognition Feed" 
          className="video-feed"
        />
        {/* Detection area overlay */}
        <div className="detection-area-overlay">
          <div className={`detection-box ${handsDetected ? 'hands-detected' : ''}`}>
            <div className="corner top-left"></div>
            <div className="corner top-right"></div>
            <div className="corner bottom-left"></div>
            <div className="corner bottom-right"></div>
            <div className="detection-label">
              {handsDetected ? '✅ Hand Detected' : '👆 Show Hand Here'}
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🌐 Sign Language Recognition System</h1>
        <div className="status-indicators">
          <span className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? '✅ Connected' : '❌ Disconnected'}
          </span>
          <span className={`processing-status ${isProcessing ? 'processing' : 'idle'}`}>
            {isProcessing ? '🔴 Live' : '⚫ Idle'}
          </span>
          <div className="language-selector">
            <select 
              value={targetLanguage}
              onChange={(e) => changeLanguage(e.target.value)}
              className="language-dropdown"
            >
              {Object.entries(languages).map(([code, name]) => (
                <option key={code} value={code}>
                  {name}
                </option>
              ))}
            </select>
          </div>
        </div>
      </header>

      {error && (
        <div className="error-banner">
          ⚠️ {error}
          <button onClick={() => setError('')} className="close-error">×</button>
          <button onClick={testConnection} className="retry-button">Test Connection</button>
        </div>
      )}

      <div className="main-content">
        {/* Left Section - Video and Available Signs */}
        <div className="left-section">
          <div className="video-section">
            <div className="video-container-wrapper">
              <VideoFeed />
            </div>

            <div className="controls">
              <button 
                onClick={startRecognition} 
                disabled={!isConnected || isProcessing}
                className="control-button start-button"
              >
                {isProcessing ? '⏳ Starting...' : '🎬 Start Recognition'}
              </button>
              <button 
                onClick={stopRecognition} 
                disabled={!isConnected || !isProcessing}
                className="control-button stop-button"
              >
                ⏹️ Stop Recognition
              </button>
              <button 
                onClick={speakText} 
                disabled={!prediction}
                className="control-button speak-button"
              >
                🔊 Speak
              </button>
            </div>
          </div>

          {/* Available Signs Section - Below buttons */}
          <div className="available-signs-section">
            <div className="section-header">
              <h3>Available Signs</h3>
              <button 
                onClick={refreshAvailableSigns}
                disabled={isLoadingSigns}
                className="refresh-button"
                title="Refresh signs"
              >
                {isLoadingSigns ? '🔄' : '🔄'}
              </button>
            </div>
            
            {isLoadingSigns ? (
              <div className="loading-signs">Loading signs...</div>
            ) : (
              <div className="signs-grid">
                {availableSigns.map((sign, index) => (
                  <div 
                    key={index}
                    className={`sign-item ${prediction === sign ? 'active' : ''}`}
                  >
                    {sign}
                  </div>
                ))}
              </div>
            )}
            
            {availableSigns.length === 0 && !isLoadingSigns && (
              <div className="no-signs">
                No signs available. Train the model first.
              </div>
            )}
          </div>
        </div>

        {/* Right Section - Recognition Results and Translation */}
        <div className="recognition-section">
          <div className="recognition-info">
            <h2>Recognition Results</h2>
            
            <div className="detection-status">
              <div className={`status-indicator ${handsDetected ? 'detected' : 'not-detected'}`}>
                {handsDetected ? '✅ Hand Detected' : '❓ No Hand Detected'}
              </div>
            </div>

            <div className="prediction-display">
              <h3>Current Sign:</h3>
              <div className="prediction-text">
                {prediction || '---'}
              </div>
            </div>

            <div className="confidence-display">
              <h3>Confidence:</h3>
              <div className="confidence-value">
                {(confidence * 100).toFixed(1)}%
              </div>
              <div className="confidence-bar">
                <div 
                  className="confidence-fill"
                  style={{ width: `${confidence * 100}%` }}
                ></div>
              </div>
            </div>

            {/* Translation Section */}
            <div className="translation-section">
              <h3>
                Translation 
                <span className="language-badge">
                  {languages[targetLanguage] || 'English'}
                </span>
              </h3>
              
              <div className="translation-controls">
                <button 
                  onClick={translateCurrentSign} 
                  disabled={!prediction || isTranslating}
                  className="control-button translate-button"
                >
                  {isTranslating ? '🔄 Translating...' : '🌐 Translate'}
                </button>
                
                {showTranslation && translatedText && (
                  <div className="translation-result">
                    <div className="translation-text">
                      {translatedText}
                    </div>
                    <div className="translation-actions">
                      <button 
                        onClick={() => copyToClipboard(translatedText)} 
                        className="control-button copy-translation-button"
                      >
                        📋 Copy
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="performance-tips">
              <h3>Tips for Better Detection:</h3>
              <ul>
                <li>✅ Place your hand inside the green detection area</li>
                <li>✅ Ensure good lighting on your hand</li>
                <li>✅ Keep background simple and uncluttered</li>
                <li>✅ Make clear, distinct signs</li>
                <li>✅ Hold each sign steady for 1-2 seconds</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
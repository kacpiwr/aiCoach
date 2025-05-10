import React, { useState } from 'react';
import '../styles/Home.css';

function Home() {
  const [file, setFile] = useState(null);
  const [userName, setUserName] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type.startsWith('video/')) {
      setFile(selectedFile);
      setError(null);
    } else {
      setError('Please select a valid video file');
      setFile(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !userName) {
      setError('Please provide both a video file and your name');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('user', userName);

    try {
      const response = await fetch('http://127.0.0.1:8000/analyze-shot', { // Updated URL
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError('Failed to analyze video. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="home-container">
      <h1>Shot Analysis Tool</h1>
      <div className="upload-section">
        <form onSubmit={handleSubmit} className="upload-form">
          <div className="form-group">
            <label htmlFor="userName">Enter Your Name:</label>
            <input
              type="text"
              id="userName"
              value={userName}
              onChange={(e) => setUserName(e.target.value)}
              placeholder="Your Name"
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="videoFile">Upload Video:</label>
            <input
              type="file"
              id="videoFile"
              accept="video/*"
              onChange={handleFileChange}
              required
            />
          </div>
          <button type="submit" disabled={loading}>
            {loading ? 'Processing...' : 'Submit'}
          </button>
        </form>
        {error && <p className="error-message">{error}</p>}
      </div>
      {results && (
  <div className="results-section">
    <h2>Analysis Results</h2>
    <div className="results-content">
      {(() => {
        const similarity = results.analysis.overall_similarity.toFixed(2);
        const similarityMessage =
          similarity > 70
            ? "Perfect"
            : similarity > 60
            ? "Great"
            : similarity > 50
            ? "Good"
            : similarity > 40
            ? "Could be better"
            : similarity > 30
            ? "Bad"
            : "Is that even a shot?";
        return (
          <>
            <h3>Overall Similarity</h3>
            <p>{similarity}% - {similarityMessage}</p>
          </>
        );
      })()}
      <h3>Stage Recommendations</h3>
      {Object.entries(results.recommendations.stages).map(([stage, data]) => (
        <div key={stage} className="recommendation-section">
          <h3>{stage.charAt(0).toUpperCase() + stage.slice(1)} Improvements</h3>
          <p>Similarity: {data.similarity.toFixed(2)}%</p>
          {data.improvements.map((improvement, index) => (
            <p key={index}>
              <strong>{improvement.feature}:</strong> {improvement.description} (Score: {improvement.score.toFixed(2)})
            </p>
          ))}
        </div>
      ))}

      <h3>Practice Drills</h3>
      {results.recommendations.practice_drills.map((drill, index) => (
        <div key={index}>
          <h3>{drill.name}</h3>
          {drill.steps.map((step, stepIndex) => (
            <p key={stepIndex}>{step}</p>
          ))}
        </div>
      ))}

      <h3>Progress Tracking</h3>
      {results.recommendations.progress_tracking.map((item, index) => (
        <p key={index}>{item}</p>
      ))}
    </div>
  </div>
)}
    </div>
  );
}

export default Home;


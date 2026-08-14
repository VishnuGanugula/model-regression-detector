import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import EmailForm from './components/EmailForm';
import ResultBadge from './components/ResultBadge';
import HistoryTable from './components/HistoryTable';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8080/api/emails';

export default function App() {
  const [history, setHistory] = useState([]);
  const [currentResult, setCurrentResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch classification history from Spring Boot backend
  const fetchHistory = useCallback(async () => {
    setHistoryLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_BASE_URL}/history`);
      setHistory(response.data || []);
    } catch (err) {
      console.error('Failed to fetch history:', err);
      setError('Could not connect to Spring Boot backend (http://localhost:8080). Make sure the server is running.');
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  // Handle submit classification form
  const handleClassify = async (emailText) => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_BASE_URL}/classify`, {
        email_text: emailText
      });
      
      const newRecord = response.data;
      setCurrentResult(newRecord);
      
      // Prepend to history list immediately
      setHistory((prev) => [newRecord, ...prev]);
    } catch (err) {
      console.error('Classification error:', err);
      const msg = err.response?.data?.message || err.response?.data || err.message || 'Error processing email classification.';
      setError(`Classification failed: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 pb-16">
      {/* Top Navbar */}
      <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-xl bg-indigo-600 flex items-center justify-center text-xl shadow-lg shadow-indigo-600/30">
              🤖
            </div>
            <div>
              <h1 className="text-lg font-bold text-white tracking-tight">AI Email Classifier Platform</h1>
              <p className="text-xs text-slate-400">MLOps Architecture: React + Spring Boot + MySQL + PyTorch FastAPI</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span>
              Live Pipeline Active
            </span>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-5xl mx-auto px-4 pt-8">

        {/* Global Error Banner */}
        {error && (
          <div className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-300 text-sm flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span>⚠️</span>
              <span>{error}</span>
            </div>
            <button 
              onClick={() => setError(null)}
              className="text-xs font-bold hover:text-white"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Form Component */}
        <EmailForm onSubmit={handleClassify} loading={loading} />

        {/* Real-time Result Badge */}
        {currentResult && (
          <ResultBadge 
            result={currentResult} 
            onClose={() => setCurrentResult(null)} 
          />
        )}

        {/* Classification History Table */}
        <HistoryTable 
          history={history} 
          loading={historyLoading} 
          onRefresh={fetchHistory} 
        />
      </main>

      {/* Footer */}
      <footer className="mt-16 text-center text-xs text-slate-600 border-t border-slate-900 pt-8">
        <p>Real-Time MLOps Microservice Architecture • PyTorch GPT-2 124M Base Model</p>
      </footer>
    </div>
  );
}
